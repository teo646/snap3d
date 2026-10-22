"""``uv_mesh`` schema: output of uv_unwrap, input to baking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class UVMesh:
    vertices: np.ndarray  # (V, 3) float64 - may be re-indexed vs. the source mesh (seam cuts duplicate verts)
    faces: np.ndarray  # (F, 3) int64
    uv: np.ndarray  # (V, 2) float32 in [0, 1], one UV per vertex (post seam-cut, so 1:1 is valid)
    vertex_normals: np.ndarray | None = None
    texture_resolution: tuple[int, int] = (1024, 1024)
    # Maps each post-unwrap vertex back to its source-mesh vertex index, so baking can
    # fetch high-poly data (or just re-run raycasts) per UV vertex. Produced by xatlas.
    source_vertex_index: np.ndarray | None = None  # (V,) int64

    def __len__(self) -> int:
        return len(self.vertices)

    def to_trimesh(self) -> trimesh.Trimesh:
        visual = trimesh.visual.TextureVisuals(uv=self.uv)
        return trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.vertex_normals,
            visual=visual,
            process=False,
        )

    def save(self, path: str | Path) -> None:
        """Writes an OBJ with `vt` UV coordinates."""
        self.to_trimesh().export(str(path))
        if self.source_vertex_index is not None:
            np.save(Path(path).with_suffix(".source_vertex_index.npy"), self.source_vertex_index)

    @classmethod
    def load(cls, path: str | Path, texture_resolution: tuple[int, int] = (1024, 1024)) -> "UVMesh":
        loaded = trimesh.load(str(path), process=False, force="mesh")
        uv = np.asarray(loaded.visual.uv) if hasattr(loaded.visual, "uv") else None
        if uv is None:
            raise ValueError(f"{path} has no UV coordinates; not a valid uv_mesh")
        source_idx_path = Path(path).with_suffix(".source_vertex_index.npy")
        source_vertex_index = np.load(source_idx_path) if source_idx_path.exists() else None
        return cls(
            vertices=np.asarray(loaded.vertices),
            faces=np.asarray(loaded.faces),
            uv=uv,
            vertex_normals=np.asarray(loaded.vertex_normals) if loaded.vertex_normals is not None else None,
            texture_resolution=texture_resolution,
            source_vertex_index=source_vertex_index,
        )
