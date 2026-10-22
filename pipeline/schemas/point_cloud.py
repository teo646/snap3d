"""``point_cloud`` schema: ingest's sparse points."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class PointCloud:
    points: np.ndarray  # (N, 3) float64
    colors: np.ndarray | None = None  # (N, 3) float in [0, 1]
    normals: np.ndarray | None = None  # (N, 3)
    # Per-point observations: `track_image_ids[i]` lists the PosedImage.image_id values that observe
    # point i, `track_pixels[i]` the matching (x, y) pixels.
    track_image_ids: list[list[int]] | None = field(default=None)
    track_pixels: list[list[tuple[float, float]]] | None = field(default=None)

    def __len__(self) -> int:
        return len(self.points)

    def to_open3d(self) -> o3d.geometry.PointCloud:
        # copy=True guards against read-only arrays (e.g. from trimesh).
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.points, dtype=np.float64, copy=True))
        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(np.array(self.colors, dtype=np.float64, copy=True))
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(np.array(self.normals, dtype=np.float64, copy=True))
        return pcd

    @classmethod
    def from_open3d(
        cls,
        pcd: o3d.geometry.PointCloud,
        track_image_ids: list[list[int]] | None = None,
        track_pixels: list[list[tuple[float, float]]] | None = None,
    ) -> "PointCloud":
        return cls(
            points=np.asarray(pcd.points),
            colors=np.asarray(pcd.colors) if pcd.has_colors() else None,
            normals=np.asarray(pcd.normals) if pcd.has_normals() else None,
            track_image_ids=track_image_ids,
            track_pixels=track_pixels,
        )

    def save(self, path: str | Path) -> None:
        """Writes a .ply next to a `<stem>.tracks.json` sidecar (if tracks are present); point order is preserved."""
        path = Path(path)
        o3d.io.write_point_cloud(str(path), self.to_open3d())
        if self.track_image_ids is not None:
            sidecar = path.with_suffix(".tracks.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "track_image_ids": self.track_image_ids,
                        "track_pixels": self.track_pixels,
                    }
                )
            )

    @classmethod
    def load(cls, path: str | Path) -> "PointCloud":
        path = Path(path)
        pcd = o3d.io.read_point_cloud(str(path))
        sidecar = path.with_suffix(".tracks.json")
        track_image_ids = track_pixels = None
        if sidecar.exists():
            data = json.loads(sidecar.read_text())
            track_image_ids = data["track_image_ids"]
            track_pixels = data["track_pixels"]
        return cls.from_open3d(pcd, track_image_ids=track_image_ids, track_pixels=track_pixels)
