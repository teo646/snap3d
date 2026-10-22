"""high-poly Mesh -> low-poly Mesh via quadric edge-collapse decimation (PyMeshLab)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import pymeshlab

from pipeline.schemas.mesh import Mesh

__all__ = ["DecimationConfig", "run_decimation"]

# Absolute floor matching PGSR's post_process_mesh: a component this small is noise.
_MIN_COMPONENT_FACES = 50


def _keep_largest_component(mesh: Mesh) -> Mesh:
    """Drops every connected component but the largest, then removes the degenerate
    triangles and unreferenced vertices the removal leaves behind. Mirrors PGSR's own
    post_process_mesh (num_cluster=1)."""
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    clusters, counts, _ = o3d_mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    if len(counts) <= 1:
        return mesh
    threshold = max(_MIN_COMPONENT_FACES, int(counts.max()))
    o3d_mesh.remove_triangles_by_mask(counts[np.asarray(clusters)] < threshold)
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_unreferenced_vertices()
    o3d_mesh.compute_vertex_normals()
    return Mesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        vertex_normals=np.asarray(o3d_mesh.vertex_normals),
    )


@dataclass
class DecimationConfig:
    target_face_count: int = 20_000
    # False because the PGSR/TSDF mesh is never watertight: scan6 has 187,960 boundary
    # edges scattered over the whole surface, and preserve_boundary=True treats each as
    # an uncollapsable loop - it stalls at 230,771 faces against a 20,000 target and
    # shatters the surface into 5,135 components (largest 1.16%), which is what baking's
    # ray-cast picked up as bogus heights. False reaches the target (19,984 faces, 626
    # components, largest 73%). Its own cost - micro-holes chaining into large tears -
    # is handled upstream by the hole_close stage.
    preserve_boundary: bool = False
    preserve_normal: bool = True


def run_decimation(mesh: Mesh, config: DecimationConfig | None = None) -> Mesh:
    config = config or DecimationConfig()
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=config.target_face_count,
        preserveboundary=config.preserve_boundary,
        preservenormal=config.preserve_normal,
    )
    result = ms.current_mesh()
    vertices, faces = result.vertex_matrix(), result.face_matrix()

    # Collapse still leaves stray sliver islands near the input's small holes (626
    # components on scan6 vs 1 in the input), and baking's ray-cast wants coherent
    # surface, not speckle. Only the largest component is kept, matching PGSR's own
    # post_process_mesh.
    cleaned = _keep_largest_component(Mesh(vertices=vertices, faces=faces))
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cleaned.vertices),
        o3d.utility.Vector3iVector(cleaned.faces),
    )
    o3d_mesh.compute_vertex_normals()
    return Mesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        vertex_normals=np.asarray(o3d_mesh.vertex_normals),
    )
