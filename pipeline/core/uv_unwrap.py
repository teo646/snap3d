"""low-poly Mesh -> UVMesh via automatic UV unwrapping (xatlas)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xatlas

from pipeline.schemas.mesh import Mesh
from pipeline.schemas.uv_mesh import UVMesh

__all__ = ["UVUnwrapConfig", "run_uv_unwrap"]


@dataclass
class UVUnwrapConfig:
    texture_resolution: tuple[int, int] = (2048, 2048)
    # xatlas's own defaults, kept as knobs because chart count matters: max_cost is the
    # per-chart distortion budget, max_iterations the grow/merge passes, straightness_weight
    # penalises ragged boundaries.
    max_cost: float = 2.0
    max_iterations: int = 1
    straightness_weight: float = 6.0
    # Chart-to-chart gap in texels (xatlas PackOptions.padding). Adjacent charts each dilate R
    # texels into it, so it must be at least 2R.
    margin_texels: int = 2


def _xatlas_options(config: UVUnwrapConfig):
    w, h = config.texture_resolution
    # The low-level Atlas API, since xatlas.parametrize() packs with padding=0 and ignores the target resolution.
    pack_options = xatlas.PackOptions()
    pack_options.resolution = max(w, h)
    pack_options.padding = config.margin_texels
    pack_options.bilinear = True

    chart_options = xatlas.ChartOptions()
    chart_options.max_cost = config.max_cost
    chart_options.max_iterations = config.max_iterations
    chart_options.straightness_weight = config.straightness_weight
    return chart_options, pack_options


def run_uv_unwrap(mesh: Mesh, config: UVUnwrapConfig | None = None) -> UVMesh:
    config = config or UVUnwrapConfig()
    chart_options, pack_options = _xatlas_options(config)

    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices, mesh.faces)
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    # vmapping[i] = the original vertex that produced new vertex i (xatlas duplicates seam vertices).
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
