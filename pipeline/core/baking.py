"""high-poly Mesh + UVMesh -> HeightMap: per-texel displacement of the high-poly
surface along the low-poly normal, baked by ray-casting the UV atlas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt

from pipeline.core.uv_geometry import GUTTER_TEXELS, dilate_radius_for
from pipeline.schemas.mesh import Mesh
from pipeline.schemas.texture_maps import HeightMap
from pipeline.schemas.uv_mesh import UVMesh

__all__ = ["BakingConfig", "run_baking"]

_RAY_BATCH_SIZE = 8192

_SELF_HIT_EPS_SCALE = 1e-6


@dataclass
class BakingConfig:
    search_distance: float | None = None  # None -> 5% of the high-poly mesh's bbox extent
    # Gutter width in texels; keep equal to UVUnwrapConfig.gutter_texels.
    gutter_texels: int = GUTTER_TEXELS
    # A height map of zeros over the same coverage, without casting a ray.
    flat: bool = False
    # Clip baked heights to +- this percentile of their magnitudes (None: keep all).
    clip_percentile: float | None = None


class _Raster(NamedTuple):
    positions: np.ndarray  # (H, W, 3) float64 — object-space position per texel
    normals: np.ndarray  # (H, W, 3) float64 — interpolated low-poly normal
    valid: np.ndarray  # (H, W) bool — core coverage, BEFORE dilation




def run_baking(highpoly: Mesh, uv_mesh: UVMesh, config: BakingConfig | None = None) -> HeightMap:
    config = config or BakingConfig()
    raster = _rasterize_uv_mesh(uv_mesh)
    h, w = raster.valid.shape

    if config.flat:
        (_, coverage) = _dilate_nearest([np.zeros((h, w))], raster.valid, dilate_radius_for(w, h, config.gutter_texels))
        return HeightMap(data=np.zeros((h, w), dtype=np.float32), valid=coverage, units="scene")

    highpoly_tm = highpoly.to_trimesh()
    extent = float(highpoly_tm.bounding_box.extents.max())
    search_distance = config.search_distance or extent * 0.05
    self_hit_eps = extent * _SELF_HIT_EPS_SCALE

    height_map = np.zeros((h, w), dtype=np.float32)

    ys, xs = np.where(raster.valid)
    if len(ys) == 0:
        return HeightMap(data=height_map, valid=np.zeros((h, w), dtype=bool), units="scene")

    tex_pos = raster.positions[ys, xs]
    tex_nrm = raster.normals[ys, xs]

    found, loc = _cast_rays_bidirectional(highpoly_tm, tex_pos, tex_nrm, search_distance, self_hit_eps)

    # Signed distance along the low-poly normal: positive = high-poly outside.
    height_vals = np.zeros(len(tex_pos), dtype=np.float64)
    height_vals[found] = np.einsum("ij,ij->i", loc[found] - tex_pos[found], tex_nrm[found])
    if config.clip_percentile is not None and found.any():
        limit = float(np.percentile(np.abs(height_vals[found]), config.clip_percentile))
        height_vals = np.clip(height_vals, -limit, limit)

    ht_core = np.zeros((h, w), dtype=np.float64)
    ht_core[ys, xs] = height_vals

    hit_mask = np.zeros((h, w), dtype=bool)
    hit_mask[ys[found], xs[found]] = True

    # Texels whose ray found nothing inherit height from the nearest hit texel.
    if hit_mask.any() and not hit_mask[raster.valid].all():
        src = _nearest_source_indices(hit_mask)
        holes = raster.valid & ~hit_mask
        ht_core[holes] = ht_core[src[0][holes], src[1][holes]]

    # Dilate into the gutter by copying the whole result from the nearest core
    (ht_dilated, coverage) = _dilate_nearest([ht_core], raster.valid, dilate_radius_for(w, h, config.gutter_texels))

    height_map[coverage] = ht_dilated[coverage].astype(np.float32)
    return HeightMap(data=height_map, valid=coverage, units="scene")




def _rasterize_uv_mesh(uv_mesh: UVMesh) -> _Raster:
    """Scanline-rasterize each UV-space triangle, interpolating 3D position / normal per texel.
    `valid` is the core coverage; the gutter is added later in `run_baking`."""
    w, h = uv_mesh.texture_resolution
    positions = np.zeros((h, w, 3), dtype=np.float64)
    normals = np.zeros((h, w, 3), dtype=np.float64)
    valid = np.zeros((h, w), dtype=bool)

    verts, faces, uv = uv_mesh.vertices, uv_mesh.faces, uv_mesh.uv
    vertex_normals = uv_mesh.vertex_normals
    if vertex_normals is None:
        vertex_normals = Mesh(vertices=verts, faces=faces).to_trimesh().vertex_normals

    # Must match uv_geometry.uv_to_atlas_xy and slf_shaders.uv_to_xy exactly.
    px = uv[:, 0] * w - 0.5
    py = (1.0 - uv[:, 1]) * h - 0.5  # UV origin is bottom-left; image row 0 is the top

    for i0, i1, i2 in faces:
        x0, y0, x1, y1, x2, y2 = px[i0], py[i0], px[i1], py[i1], px[i2], py[i2]

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue

        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))), w - 1)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))), h - 1)
        if xmin > xmax or ymin > ymax:
            continue

        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1

        eps = -1e-6
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)

        if not np.any(inside):
            # Sliver triangle containing no texel center: claim the texel nearest its centroid.
            cx = int(round((x0 + x1 + x2) / 3.0))
            cy = int(round((y0 + y1 + y2) / 3.0))
            if 0 <= cx < w and 0 <= cy < h and not valid[cy, cx]:
                positions[cy, cx] = (verts[i0] + verts[i1] + verts[i2]) / 3.0
                n = vertex_normals[i0] + vertex_normals[i1] + vertex_normals[i2]
                normals[cy, cx] = _normalize(n)
                valid[cy, cx] = True
            continue

        b0, b1, b2 = w0[inside], w1[inside], w2[inside]
        pos = b0[:, None] * verts[i0] + b1[:, None] * verts[i1] + b2[:, None] * verts[i2]
        nrm = (
            b0[:, None] * vertex_normals[i0]
            + b1[:, None] * vertex_normals[i1]
            + b2[:, None] * vertex_normals[i2]
        )
        nrm_len = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm_len[nrm_len == 0] = 1.0
        nrm = nrm / nrm_len

        px_idx, py_idx = gx[inside], gy[inside]
        positions[py_idx, px_idx] = pos
        normals[py_idx, px_idx] = nrm
        valid[py_idx, px_idx] = True

    return _Raster(positions, normals, valid)




def _cast_rays_bidirectional(
    mesh: trimesh.Trimesh,
    pos: np.ndarray,
    nrm: np.ndarray,
    max_distance: float,
    self_hit_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast outward (+n) and inward (-n) from each texel up to `max_distance`, keeping the closest
    hit. Bounded so a texel over a hole in the high-poly mesh does not hit the far side."""
    n = len(pos)
    found = np.zeros(n, dtype=bool)
    best_loc = np.zeros((n, 3), dtype=np.float64)
    best_dist = np.full(n, np.inf, dtype=np.float64)

    if n == 0:
        return found, best_loc

    for direction_sign in (1.0, -1.0):
        # Each ray starts `self_hit_eps` behind the texel so a surface through the texel is still ahead.
        origins = pos - nrm * (direction_sign * self_hit_eps)
        directions = nrm * direction_sign
        f, loc = _cast_rays_nearest(mesh, origins, directions, max_distance + self_hit_eps)
        if not np.any(f):
            continue
        dist = np.full(n, np.inf, dtype=np.float64)
        dist[f] = np.linalg.norm(loc[f] - pos[f], axis=1)
        better = f & (dist <= max_distance) & (dist < best_dist)
        best_dist[better] = dist[better]
        best_loc[better] = loc[better]
        found |= better

    return found, best_loc


def _cast_rays_nearest(
    mesh: trimesh.Trimesh,
    origins: np.ndarray,
    directions: np.ndarray,
    max_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-ray nearest hit within `max_distance`, batched to bound peak memory."""
    n = len(origins)
    hit_found = np.zeros(n, dtype=bool)
    hit_location = np.zeros((n, 3), dtype=np.float64)
    if n == 0:
        return hit_found, hit_location

    for start in range(0, n, _RAY_BATCH_SIZE):
        end = min(start + _RAY_BATCH_SIZE, n)
        batch_origins = origins[start:end]
        batch_directions = directions[start:end]

        locations, index_ray, _ = mesh.ray.intersects_location(
            batch_origins, batch_directions, multiple_hits=True
        )
        if len(index_ray) == 0:
            continue

        dists = np.linalg.norm(locations - batch_origins[index_ray], axis=1)

        if max_distance is not None:
            keep = dists <= max_distance
            if not np.any(keep):
                continue
            locations = locations[keep]
            index_ray = index_ray[keep]
            dists = dists[keep]

        order = np.lexsort((dists, index_ray))
        sorted_ray = index_ray[order]
        _, first_pos = np.unique(sorted_ray, return_index=True)
        chosen = order[first_pos]

        rays = start + index_ray[chosen]
        hit_found[rays] = True
        hit_location[rays] = locations[chosen]
    return hit_found, hit_location




def _nearest_source_indices(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every texel, the (row, col) of the nearest True texel in `mask`."""
    _, indices = distance_transform_edt(~mask, return_distances=True, return_indices=True)
    return indices[0], indices[1]


def _dilate_nearest(
    layers: list[np.ndarray], valid: np.ndarray, radius: int
) -> tuple[np.ndarray, ...]:
    """Extend each layer `radius` texels outside `valid` by copying the nearest valid texel's
    value (never averaging). Returns the layers, then the dilated mask."""
    if not valid.any() or radius <= 0:
        return (*layers, valid.copy())

    distance, indices = distance_transform_edt(~valid, return_distances=True, return_indices=True)
    ring = (~valid) & (distance <= radius)
    sy, sx = indices[0], indices[1]

    out = []
    for layer in layers:
        filled = layer.copy()
        filled[ring] = layer[sy[ring], sx[ring]]
        out.append(filled)
    return (*out, valid | ring)




def _normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    return v / length if length > 0.0 else v
