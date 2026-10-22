"""low-poly Mesh -> UVMesh via automatic UV unwrapping (xatlas).

`whole_charts` names sets of faces that must each come out as ONE UV island (a face to be edited
in UV space, say). xatlas has no such constraint - it cuts a chart wherever its own cost says,
and no option reaches it: on a statue's face `max_cost` from 2 to 64 gave 4 and then 7 islands.
So the sets are unwrapped on their own, where a clean disk comes out as one chart, the rest of the
mesh is unwrapped as usual, and the two are packed together in a final pass (see `run_uv_unwrap`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xatlas

from pipeline.core.chart_flatten import flatten_disk
from pipeline.core.chart_region import make_disk
from pipeline.core.uv_geometry import GUTTER_TEXELS
from pipeline.schemas.mesh import Mesh
from pipeline.schemas.uv_mesh import UVMesh

__all__ = ["UVUnwrapConfig", "run_uv_unwrap"]


@dataclass
class UVUnwrapConfig:
    texture_resolution: tuple[int, int] = (2048, 2048)
    # These three are xatlas's own defaults, kept as knobs because chart count is the
    # thing that hurts here: a decimated marching-cubes surface unwraps into thousands
    # of tiny islands (lego 4,841 charts, chair 1,125 at ~50,000 faces), and every
    # island costs a gutter. max_cost is the per-chart distortion budget a chart may
    # not exceed; max_iterations is how many grow/merge passes xatlas runs (at 1 it
    # never gets to merge charts it split early); straightness_weight penalises ragged
    # boundaries, which on staircase-shaped triangles is cheapest to satisfy by
    # starting a new chart. Sweeping all three failed to reduce chart count usefully.
    max_cost: float = 2.0
    max_iterations: int = 1
    straightness_weight: float = 6.0
    # Chart-to-chart gap in texels (xatlas PackOptions.padding). Two adjacent charts
    # each dilate R texels into it, so this must be at least 2R or their gutters
    # overlap and each gets filled with the other's colour (measured at 1024^2 with
    # padding == R: 95-100% of charts merged into one blob for every R in 1..8).
    # Anything above 2R is slack that only costs chart area.
    margin_texels: int = 2


def _xatlas_options(config: UVUnwrapConfig, chart_overrides: dict | None = None):
    w, h = config.texture_resolution
    # The low-level Atlas API rather than xatlas.parametrize(): that wrapper packs with
    # padding=0 and ignores the target resolution, so charts end up closer together than
    # the gutter baking dilates by, and unrelated charts' gutters collide.
    pack_options = xatlas.PackOptions()
    pack_options.resolution = max(w, h)
    pack_options.padding = config.margin_texels
    pack_options.bilinear = True

    chart_options = xatlas.ChartOptions()
    chart_options.max_cost = config.max_cost
    chart_options.max_iterations = config.max_iterations
    chart_options.straightness_weight = config.straightness_weight
    for name, value in (chart_overrides or {}).items():
        setattr(chart_options, name, value)
    return chart_options, pack_options


def _unwrap_faces(vertices: np.ndarray, faces: np.ndarray, config: UVUnwrapConfig, chart_overrides: dict | None = None):
    """xatlas on a subset: (positions of the input vertices used, vmapping into the subset's
    own vertex list, faces into the output vertices, uvs)."""
    used, compact = np.unique(faces, return_inverse=True)
    chart_options, pack_options = _xatlas_options(config, chart_overrides)
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices[used], compact.reshape(-1, 3).astype(np.uint32))
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    vmapping, indices, uvs = atlas.get_mesh(0)
    return used[vmapping], indices.astype(np.int64), uvs.astype(np.float64), atlas.chart_count


def _density(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> float:
    """Median UV area per unit of surface area over the faces (UV units, so unitless)."""
    a3 = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
    e1, e2 = uv[faces[:, 1]] - uv[faces[:, 0]], uv[faces[:, 2]] - uv[faces[:, 0]]
    a2 = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    return float(np.median(a2[a3 > 0] / a3[a3 > 0]))


def _unwrap_with_whole_charts(mesh: Mesh, config: UVUnwrapConfig, whole_charts: list[np.ndarray]) -> UVMesh:
    """Each set of faces in `whole_charts` becomes one chart.

    1. A set is repaired into a disk (`make_disk`) and flattened on its own (`flatten_disk`: a
       fold-free map, so it is one chart by construction). Not by xatlas: given a statue's face
       (550 faces) alone it still cuts it into five charts, whatever its options.
    2. Everything else is unwrapped as usual.
    3. The two sets of UVs are joined into one mesh in the INPUT's face order - so a face keeps
       its index and downstream stages see the mesh they expect - with each whole chart scaled
       to the texel density of the rest (so it is not blown up to fill an atlas of its own) and
       moved clear of it.
    4. xatlas packs that mesh a last time, from the UVs it is given (`use_input_mesh_uvs`): charts
       are the UV-connected pieces, and they are only placed, not re-cut.
    """
    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    # welded by position first: a mesh read from OBJ may repeat a vertex, which would look like a boundary
    _, _, weld = np.unique(np.round(verts, 7), axis=0, return_index=True, return_inverse=True)
    welded = weld[faces]

    in_chart = np.full(len(faces), -1, dtype=np.int64)
    for k, selected in enumerate(whole_charts):
        disk = make_disk(welded, np.asarray(selected))
        if (in_chart[disk] >= 0).any():
            raise ValueError("two whole charts share faces")
        in_chart[disk] = k

    rest = np.nonzero(in_chart < 0)[0]
    r_src, r_idx, r_uv, _ = _unwrap_faces(verts, faces[rest], config)
    rest_density = _density(verts[r_src], r_idx, r_uv)

    # combined vertex list: the rest's, then each chart's
    c_src, c_uv = [r_src], [r_uv.copy()]
    face_vertices = np.zeros((len(faces), 3), dtype=np.int64)
    face_vertices[rest] = r_idx
    offset = len(r_src)
    for k in range(len(whole_charts)):
        members = np.nonzero(in_chart == k)[0]
        flat = flatten_disk(verts, faces[members])
        src, idx, uv = flat["vertex_ids"], flat["faces"], flat["uv"]
        uv = uv * np.sqrt(rest_density / _density(verts[src], idx, uv))
        uv = uv - uv.min(axis=0) + np.array([2.0 + 2.0 * k, 0.0])  # clear of the rest, which is in [0, 1]^2
        c_src.append(src)
        c_uv.append(uv)
        face_vertices[members] = idx + offset
        offset += len(src)
    src_all, uv_all = np.concatenate(c_src), np.concatenate(c_uv)

    chart_options, pack_options = _xatlas_options(config, {"use_input_mesh_uvs": True})
    atlas = xatlas.Atlas()
    atlas.add_mesh(verts[src_all], face_vertices.astype(np.uint32), uvs=uv_all.astype(np.float32))
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    vmapping, indices, uvs = atlas.get_mesh(0)
    source = src_all[vmapping]
    return UVMesh(
        vertices=verts[source],
        faces=indices.astype(np.int64),
        uv=uvs.astype(np.float32),
        vertex_normals=mesh.vertex_normals[source] if mesh.vertex_normals is not None else None,
        texture_resolution=config.texture_resolution,
        source_vertex_index=source.astype(np.int64),
    )


def run_uv_unwrap(mesh: Mesh, config: UVUnwrapConfig | None = None, whole_charts: list[np.ndarray] | None = None) -> UVMesh:
    config = config or UVUnwrapConfig()
    if whole_charts:
        return _unwrap_with_whole_charts(mesh, config, whole_charts)

    chart_options, pack_options = _xatlas_options(config)

    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices, mesh.faces)
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    # vmapping[i] = the original vertex that produced new vertex i (xatlas duplicates
    # vertices along seam cuts).
    vmapping, indices, uvs = atlas.get_mesh(0)

    new_vertices = mesh.vertices[vmapping]
    new_normals = mesh.vertex_normals[vmapping] if mesh.vertex_normals is not None else None

    return UVMesh(
        vertices=new_vertices,
        faces=indices.astype(np.int64),
        uv=uvs.astype(np.float32),
        vertex_normals=new_normals,
        texture_resolution=config.texture_resolution,
        source_vertex_index=vmapping.astype(np.int64),
    )
