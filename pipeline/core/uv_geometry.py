"""UV-atlas geometry: rasterizes a `UVMesh` into per-texel position/normal/tangent
arrays (same bbox + barycentric scan as `baking._rasterize_uv_mesh`), plus the
per-vertex frame attributes the shaders rebuild that frame from.

Pixel/UV convention, which must match baking.py and slf_shaders.py's
`xy_to_uv`/`uv_to_xy` exactly: (u, v) maps to array coordinates
(x, y) = (u*W - 0.5, (1 - v)*H - 0.5), so row 0 is the top and texel i's center is at
u = (i + 0.5)/W. That half-texel offset is where GL's `texture()` actually samples,
and every texel weight computed here is eventually inverted against a `texture()`
call (sh_math.cg_solve_bilinear_texels) - getting it wrong misregisters training
against rendering by up to half a texel. `uv_to_atlas_xy`/`sample_bilinear` are the
only place the conversion happens on the numpy side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

from pipeline.schemas.mesh import Mesh
from pipeline.schemas.uv_mesh import UVMesh

__all__ = [
    "UVAtlasGeometry",
    "compute_vertex_tangents",
    "orthonormal_frame",
    "vertex_frame_attributes",
    "vertex_normals_for",
    "uv_to_atlas_xy",
    "rasterize_uv_atlas",
    "sample_bilinear",
    "dilate_valid",
    "dilate_radius_for",
    "GUTTER_TEXELS",
]

# Gutter width in texels. A gutter has to cover bilinear's 2x2 neighbourhood (1 texel)
# plus the sub-texel overshoot of the POM march, both counted in texels rather than UV
# area, and it does not scale with resolution. Swept 1..16 at 1024^2 on a synthetic
# benchmark set (no longer in this repo): render coverage is flat to 0.05%, i.e. no march lands
# outside a 1-texel gutter - so 1 is the minimum and buys the most chart area.
GUTTER_TEXELS = 1


def dilate_radius_for(width: int, height: int, ref_texels: int = GUTTER_TEXELS) -> int:
    """Gutter radius in texels - a fixed count. `width`/`height` are ignored, and kept
    only so every call site reads the same."""
    return max(1, int(ref_texels))


def compute_vertex_tangents(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex raw (un-normalized) tangent/bitangent from the UV parametrization
    (Lengyel's method), accumulated over incident faces. The magnitude is world units
    per UV unit: direction gives the frame, length converts a world step to a UV step.
    """
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    uv0, uv1, uv2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    edge1, edge2 = v1 - v0, v2 - v0
    duv1, duv2 = uv1 - uv0, uv2 - uv0

    denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    degenerate = np.abs(denom) < 1e-12
    r = np.where(degenerate, 0.0, 1.0 / np.where(degenerate, 1.0, denom))

    face_tangent = (edge1 * duv2[:, 1:2] - edge2 * duv1[:, 1:2]) * r[:, None]
    face_bitangent = (edge2 * duv1[:, 0:1] - edge1 * duv2[:, 0:1]) * r[:, None]

    tangent = np.zeros_like(vertices)
    bitangent = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(tangent, faces[:, k], face_tangent)
        np.add.at(bitangent, faces[:, k], face_bitangent)
    return tangent, bitangent


def orthonormal_frame(raw_tangent: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gram-Schmidt `raw_tangent` against `normal` (both (..., 3)) -> unit (tangent,
    bitangent). Degenerate tangents fall back to an arbitrary orthogonal vector, which
    only affects that texel's frame orientation. GLSL twin: slf_shaders.uv_frame.
    """
    n = normal / np.clip(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12, None)
    t = raw_tangent - n * np.sum(raw_tangent * n, axis=-1, keepdims=True)
    t_len = np.linalg.norm(t, axis=-1, keepdims=True)
    degenerate = t_len[..., 0] < 1e-8

    fallback_axis = np.where(np.abs(n[..., :1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    fallback = fallback_axis - n * np.sum(fallback_axis * n, axis=-1, keepdims=True)
    t = np.where(degenerate[..., None], fallback, t)
    t_len = np.clip(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12, None)
    t_unit = t / t_len
    b_unit = np.cross(n, t_unit)
    return t_unit, b_unit


def vertex_normals_for(uv_mesh: UVMesh) -> np.ndarray:
    """Per-vertex normals, falling back to trimesh's smooth ones when the .obj carried
    none. The single source: the atlas and the shaders' vertex attributes only agree
    because both start here."""
    normals = uv_mesh.vertex_normals
    if normals is None:
        normals = Mesh(vertices=uv_mesh.vertices, faces=uv_mesh.faces).to_trimesh().vertex_normals
    return np.asarray(normals, dtype=np.float64)


def vertex_frame_attributes(uv_mesh: UVMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(normal, raw tangent, raw bitangent) per vertex - everything a shader needs to
    rebuild the local frame with no frame texture at all.

    These are the same arrays `rasterize_uv_atlas` interpolates barycentrically, so a
    fragment interpolating them gets the identical value, not an approximation. The
    tangents stay un-normalized because their length is the `xy_per_height` scale.
    """
    normals = vertex_normals_for(uv_mesh)
    tangent, bitangent = compute_vertex_tangents(uv_mesh.vertices, uv_mesh.faces, uv_mesh.uv)
    return normals, tangent, bitangent


def uv_to_atlas_xy(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """(N, 2) UV in [0, 1] (UVMesh convention) -> (N, 2) continuous atlas (x, y).
    Numpy twin of slf_shaders.py's GLSL `uv_to_xy` - see this module's docstring for
    why the half-texel offset is mandatory rather than cosmetic."""
    x = uv[:, 0] * width - 0.5
    y = (1.0 - uv[:, 1]) * height - 0.5
    return np.stack([x, y], axis=-1)


def uv_to_atlas_xy_torch(uv, width: int, height: int):
    """`uv_to_atlas_xy` for a torch tensor, on whatever device it lives on."""
    import torch

    return torch.stack([uv[:, 0] * width - 0.5, (1.0 - uv[:, 1]) * height - 0.5], dim=-1)


def dilate_valid(
    valid: np.ndarray, fields: list[np.ndarray], radius: int, target_valid: np.ndarray | None = None
) -> np.ndarray:
    """Texture-padding: every invalid texel within `radius` of a valid one takes that
    nearest valid texel's values (in place, per array in `fields`) and becomes valid.
    Closes thin-triangle rasterization gaps and bleed-pads chart edges so bilinear
    sampling near a seam never blends in an untouched zero. Returns the new valid mask;
    `valid` is not modified.

    `target_valid`, if given, is used as the fill region instead of the `radius` ball -
    for a caller that already has an authoritative coverage mask (baking.py's gutter)
    and wants these fields padded to exactly it.
    """
    if target_valid is None:
        if not valid.any() or (~valid).sum() == 0:
            return valid.copy()
        dist, (iy, ix) = ndi.distance_transform_edt(~valid, return_indices=True)
        to_fill = (~valid) & (dist <= radius)
        for field in fields:
            field[to_fill] = field[iy[to_fill], ix[to_fill]]
        return valid | to_fill

    if not valid.any():
        return target_valid.copy()
    _, (iy, ix) = ndi.distance_transform_edt(~valid, return_indices=True)
    to_fill = target_valid & ~valid
    for field in fields:
        field[to_fill] = field[iy[to_fill], ix[to_fill]]
    return target_valid.copy()


@dataclass
class UVAtlasGeometry:
    positions: np.ndarray  # (H, W, 3) low-poly flat surface position
    normals: np.ndarray  # (H, W, 3) low-poly interpolated normal (unit)
    tangent_raw: np.ndarray  # (H, W, 3) raw tangent (world units per UV-u unit)
    bitangent_raw: np.ndarray  # (H, W, 3) raw bitangent (world units per UV-v unit)
    valid: np.ndarray  # (H, W) bool


def rasterize_uv_atlas(uv_mesh: UVMesh, target_valid: np.ndarray | None = None) -> UVAtlasGeometry:
    """Per-texel position/normal/tangent arrays, rasterized with the same bbox +
    barycentric scan as `baking._rasterize_uv_mesh` so texel alignment matches the
    already-baked HeightMap.

    Pass a HeightMap's `valid` as `target_valid` to get back exactly baking's gutter
    mask: the two rasterizers' pre-dilate cores differ by a few texels at slivers
    (baking.py has a fallback for a sliver too thin to hold a texel center), so without
    it the two masks are not guaranteed identical. Omit it to use this rasterizer's own
    radius dilate.
    """
    w, h = uv_mesh.texture_resolution
    positions = np.zeros((h, w, 3), dtype=np.float64)
    normals = np.zeros((h, w, 3), dtype=np.float64)
    tangent_raw = np.zeros((h, w, 3), dtype=np.float64)
    bitangent_raw = np.zeros((h, w, 3), dtype=np.float64)
    valid = np.zeros((h, w), dtype=bool)

    verts, faces, uv = uv_mesh.vertices, uv_mesh.faces, uv_mesh.uv
    vertex_normals = vertex_normals_for(uv_mesh)
    vertex_tangent, vertex_bitangent = compute_vertex_tangents(verts, faces, uv)

    # Same map as uv_to_atlas_xy / slf_shaders.uv_to_xy: in this space texel i's center
    # is at x == i, so testing integer coordinates tests texel centers.
    px = uv[:, 0] * w - 0.5
    py = (1.0 - uv[:, 1]) * h - 0.5

    for i0, i1, i2 in faces:
        x0, y0, x1, y1, x2, y2 = px[i0], py[i0], px[i1], py[i1], px[i2], py[i2]
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))), w - 1)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))), h - 1)
        if xmin > xmax or ymin > ymax:
            continue

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue

        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1

        eps = -1e-6
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not np.any(inside):
            continue

        b0, b1, b2 = w0[inside], w1[inside], w2[inside]
        pos = b0[:, None] * verts[i0] + b1[:, None] * verts[i1] + b2[:, None] * verts[i2]
        nrm = b0[:, None] * vertex_normals[i0] + b1[:, None] * vertex_normals[i1] + b2[:, None] * vertex_normals[i2]
        nrm_len = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm_len[nrm_len == 0] = 1.0
        nrm = nrm / nrm_len
        tan = b0[:, None] * vertex_tangent[i0] + b1[:, None] * vertex_tangent[i1] + b2[:, None] * vertex_tangent[i2]
        bit = b0[:, None] * vertex_bitangent[i0] + b1[:, None] * vertex_bitangent[i1] + b2[:, None] * vertex_bitangent[i2]

        px_idx, py_idx = gx[inside], gy[inside]
        positions[py_idx, px_idx] = pos
        normals[py_idx, px_idx] = nrm
        tangent_raw[py_idx, px_idx] = tan
        bitangent_raw[py_idx, px_idx] = bit
        valid[py_idx, px_idx] = True

    valid = dilate_valid(
        valid, [positions, normals, tangent_raw, bitangent_raw], dilate_radius_for(w, h), target_valid=target_valid
    )
    return UVAtlasGeometry(positions, normals, tangent_raw, bitangent_raw, valid)


def sample_bilinear(raster: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear-samples a (H, W, C) raster at continuous atlas coordinates xy (N, 2);
    edges clamp rather than wrap."""
    h, w = raster.shape[:2]
    x0f = np.floor(xy[:, 0])
    y0f = np.floor(xy[:, 1])
    fx = (xy[:, 0] - x0f)[:, None]
    fy = (xy[:, 1] - y0f)[:, None]
    x0 = np.clip(x0f, 0, w - 1).astype(np.int64)
    x1 = np.clip(x0f + 1, 0, w - 1).astype(np.int64)
    y0 = np.clip(y0f, 0, h - 1).astype(np.int64)
    y1 = np.clip(y0f + 1, 0, h - 1).astype(np.int64)

    c00, c10 = raster[y0, x0], raster[y0, x1]
    c01, c11 = raster[y1, x0], raster[y1, x1]
    top = c00 * (1 - fx) + c10 * fx
    bot = c01 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bot * fy
