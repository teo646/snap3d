"""high-poly Mesh -> hole-closed high-poly Mesh (PyMeshLab `meshing_close_holes`).

Sits between surface_recon and decimation: the TSDF mesh is never watertight, and decimation
without boundary preservation chains micro-holes into large tears. It is a stage rather than a
decimation option because baking ray-casts against the high-poly mesh, which must be the same
closed surface. `max_hole_size` is PyMeshLab's `maxholesize` in boundary edges, so a large
unobserved region stays open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymeshlab

from pipeline.schemas.mesh import Mesh

__all__ = ["HoleCloseConfig", "run_hole_close"]


@dataclass
class HoleCloseConfig:
    # In boundary edges (PyMeshLab's `maxholesize`). 0 makes this stage a pass-through.
    max_hole_size: int = 5000


def run_hole_close(mesh: Mesh, config: HoleCloseConfig | None = None) -> Mesh:
    config = config or HoleCloseConfig()
    if config.max_hole_size <= 0:
        return mesh

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    # newfaceselected=False: otherwise decimation honors the leftover selection and collapses only inside the patches.
    ms.meshing_close_holes(maxholesize=config.max_hole_size, newfaceselected=False)
    result = ms.current_mesh()
    return Mesh(vertices=np.asarray(result.vertex_matrix()), faces=np.asarray(result.face_matrix()))
