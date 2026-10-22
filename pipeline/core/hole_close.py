"""high-poly Mesh -> hole-closed high-poly Mesh (PyMeshLab `meshing_close_holes`).

Sits between surface_recon and decimation. PGSR's TSDF mesh is never watertight (on
scan6: 187,960 boundary edges, 85% of the holes under 6 vertices - sub-triangle-scale
at 7.4M faces, so invisible until decimation). Decimation needs
`preserve_boundary=False`, and a boundary collapse merges two hole edges instead of
closing the hole, so as the interior thins ~16x the hole perimeters barely shrink and
thousands of micro-holes chain into a few large tears (scan6: one loop spanning 34% of
the low-poly vertices) that read as cracks in the render. Triangulating the small holes
shut beforehand removes the raw material for that chaining.

`max_hole_size` is PyMeshLab's `maxholesize` in boundary edges, so a genuinely
unobserved region (the object's underside) is left open rather than covered with
invented geometry.

It is a stage rather than a decimation option because baking ray-casts the low-poly
atlas against the *high-poly* mesh: both must see the same closed surface, or every
texel over a closed hole finds no ray hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymeshlab

from pipeline.schemas.mesh import Mesh

__all__ = ["HoleCloseConfig", "run_hole_close"]


@dataclass
class HoleCloseConfig:
    # In boundary edges (PyMeshLab's `maxholesize`). 0 disables hole closing
    # entirely and makes this stage a pass-through. 500 is the level the
    # hole_test sweep settled on for scan6.
    max_hole_size: int = 500


def run_hole_close(mesh: Mesh, config: HoleCloseConfig | None = None) -> Mesh:
    config = config or HoleCloseConfig()
    if config.max_hole_size <= 0:
        return mesh

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    # newfaceselected=False: the default leaves the patch faces selected, and
    # meshing_decimation_quadric_edge_collapse honors that leftover selection even when
    # called with selected=False - which made decimation collapse only inside the
    # patches (7,449,385 -> 7,449,347 against a 20,000 target).
    ms.meshing_close_holes(maxholesize=config.max_hole_size, newfaceselected=False)
    result = ms.current_mesh()
    return Mesh(vertices=np.asarray(result.vertex_matrix()), faces=np.asarray(result.face_matrix()))
