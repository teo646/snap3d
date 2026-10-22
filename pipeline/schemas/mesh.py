"""``mesh`` schema shared by surface_recon -> decimation -> baking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3) float64
    faces: np.ndarray  # (F, 3) int64
    vertex_normals: np.ndarray | None = None  # (V, 3)
    vertex_colors: np.ndarray | None = None  # (V, 3 or 4) uint8

    def __len__(self) -> int:
        return len(self.vertices)

    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        # copy=True: trimesh sometimes hands back read-only arrays, which Vector3dVector rejects.
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(self.vertices, dtype=np.float64, copy=True))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(self.faces, dtype=np.int32, copy=True))
        if self.vertex_normals is not None:
            mesh.vertex_normals = o3d.utility.Vector3dVector(
                np.array(self.vertex_normals, dtype=np.float64, copy=True)
            )
        if self.vertex_colors is not None:
            colors = np.array(self.vertex_colors[:, :3], dtype=np.float64, copy=True) / 255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh

    @classmethod
    def from_open3d(cls, mesh: o3d.geometry.TriangleMesh) -> "Mesh":
        return cls(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None,
            vertex_colors=(
                (np.asarray(mesh.vertex_colors) * 255.0).astype(np.uint8) if mesh.has_vertex_colors() else None
            ),
        )

    def to_trimesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.vertex_normals,
            vertex_colors=self.vertex_colors,
            process=False,
        )

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh) -> "Mesh":
        return cls(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.faces),
            vertex_normals=np.asarray(mesh.vertex_normals) if mesh.vertex_normals is not None else None,
            vertex_colors=(
                np.asarray(mesh.visual.vertex_colors)
                if getattr(mesh.visual, "vertex_colors", None) is not None
                else None
            ),
        )

    def save(self, path: str | Path) -> None:
        self.to_trimesh().export(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "Mesh":
        loaded = trimesh.load(str(path), process=False, force="mesh")
        return cls.from_trimesh(loaded)
