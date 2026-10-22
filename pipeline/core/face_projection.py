"""Facial landmarks in photographs -> points on a UV-mapped mesh.

A detector (MediaPipe's FaceLandmarker) says where the eyes, nose and mouth are *in a photo*.
The mesh is what the rest of the pipeline holds, so each landmark is carried onto it:

  pixel --(camera pose + intrinsics)--> ray --(first hit on the mesh)--> 3D point --> UV

Several photos see the same face, and each places the landmarks a little differently - or, from
an angle the detector handles badly, wrongly. The 3D points are combined by a median, and a
photo whose points sit far from the rest is dropped (`consensus_landmarks`); the UV of the
consensus comes from the mesh surface nearest it, so a point that lands on a UV seam still gets
one coordinate.

Conventions: poses are COLMAP's (`x_cam = R X + t`, +z forward, +y down); a pixel (u, v) is in
the image's own frame with (0, 0) its top-left *corner* (what MediaPipe's `x * width` gives and
what COLMAP's principal point assumes). UVs are the OBJ's: origin bottom-left, so a texel's
image row is `(1 - v) * height`.
"""

from __future__ import annotations

import numpy as np
import trimesh

from pipeline.schemas.posed_images import CameraIntrinsics, CameraPose
from pipeline.schemas.uv_mesh import UVMesh

__all__ = [
    "pinhole",
    "pixel_rays",
    "project_points",
    "UVSurface",
    "consensus_landmarks",
    "hull_mask",
]


def pinhole(camera: CameraIntrinsics) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy). The photos handed to this are undistorted, so only the pinhole models
    are accepted - a distorted camera would put every ray in the wrong place without a word."""
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    elif camera.model == "SIMPLE_PINHOLE":
        fx = fy = camera.params[0]
        cx, cy = camera.params[1], camera.params[2]
    else:
        raise ValueError(f"camera model {camera.model} has distortion; project from the undistorted photos")
    return fx, fy, cx, cy


def pixel_rays(pose: CameraPose, camera: CameraIntrinsics, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(origin (3,), unit directions (N, 3)) in world space for `pixels` (N, 2)."""
    fx, fy, cx, cy = pinhole(camera)
    pixels = np.asarray(pixels, dtype=np.float64)
    d_cam = np.stack([(pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy, np.ones(len(pixels))], axis=1)
    d_world = d_cam @ pose.rotation_matrix()  # R^T d, as rows
    d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
    return pose.camera_center(), d_world


def project_points(pose: CameraPose, camera: CameraIntrinsics, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pixels (N, 2), depth (N,)) of world points; depth <= 0 is behind the camera."""
    fx, fy, cx, cy = pinhole(camera)
    p = np.asarray(xyz, dtype=np.float64) @ pose.rotation_matrix().T + np.array([pose.tx, pose.ty, pose.tz])
    z = p[:, 2]
    safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    return np.stack([fx * p[:, 0] / safe + cx, fy * p[:, 1] / safe + cy], axis=1), z


class UVSurface:
    """A mesh with one UV per vertex, that rays can be cast at and points snapped to."""

    def __init__(self, uv_mesh: UVMesh):
        self.vertices = np.asarray(uv_mesh.vertices, dtype=np.float64)
        self.faces = np.asarray(uv_mesh.faces, dtype=np.int64)
        self.uv = np.asarray(uv_mesh.uv, dtype=np.float64)
        self.resolution = tuple(uv_mesh.texture_resolution)
        self.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)

    def _uv_of(self, xyz: np.ndarray, tri: np.ndarray) -> np.ndarray:
        bary = trimesh.triangles.points_to_barycentric(self.vertices[self.faces[tri]], xyz)
        return np.einsum("nk,nkd->nd", bary, self.uv[self.faces[tri]])

    def cast(self, origin: np.ndarray, directions: np.ndarray) -> dict[str, np.ndarray]:
        """First hit of each ray: `xyz` (N, 3), `uv` (N, 2), `face` (N,), `distance` (N,); a
        ray that hits nothing has NaN / -1."""
        n = len(directions)
        xyz = np.full((n, 3), np.nan)
        uv = np.full((n, 2), np.nan)
        face = np.full(n, -1, dtype=np.int64)
        dist = np.full(n, np.nan)
        if n:
            hit_xyz, ray_idx, tri_idx = self.mesh.ray.intersects_location(
                np.tile(origin, (n, 1)), directions, multiple_hits=False
            )
            if len(ray_idx):
                xyz[ray_idx] = hit_xyz
                face[ray_idx] = tri_idx
                uv[ray_idx] = self._uv_of(hit_xyz, tri_idx)
                dist[ray_idx] = np.linalg.norm(hit_xyz - origin, axis=1)
        return {"xyz": xyz, "uv": uv, "face": face, "distance": dist}

    def snap(self, xyz: np.ndarray) -> dict[str, np.ndarray]:
        """The mesh surface point nearest each of `xyz`, with its UV: `xyz`, `uv`, `face`, `gap`
        (how far the input was from the surface)."""
        closest, gap, tri = trimesh.proximity.closest_point(self.mesh, xyz)
        return {"xyz": closest, "uv": self._uv_of(closest, tri), "face": tri, "gap": gap}

    def uv_to_pixel(self, uv: np.ndarray) -> np.ndarray:
        """UV -> continuous (column, row) in the texture, texel i covering [i, i + 1)."""
        w, h = self.resolution
        return np.stack([uv[..., 0] * w, (1.0 - uv[..., 1]) * h], axis=-1)


def consensus_landmarks(
    points: np.ndarray, scale_pair: tuple[int, int], max_error: float = 0.25, min_hits: float = 0.6, rounds: int = 4,
) -> dict[str, np.ndarray]:
    """One 3D position per landmark from many views.

    `points` is (V, K, 3), NaN where a view's ray missed the mesh. A view needs at least
    `min_hits` of its landmarks on the mesh to count at all. The consensus is the per-landmark
    median over the views still counted; a view's error is the median distance of its
    landmarks from it, in units of the distance between the two landmarks `scale_pair`
    (the eyes' outer corners - the face's own ruler, so the threshold means the same on any
    mesh); views over `max_error` are dropped and the consensus recomputed, `rounds` times.

    Returns `position` (K, 3), `spread` (K,) - the robust spread (1.4826 MAD) of the kept views'
    distances from it, `error` (V,) per view, `kept` (V,) and `n_views` (K,) used per landmark."""
    v, k, _ = points.shape
    hit_share = np.mean(np.isfinite(points[..., 0]), axis=1)
    kept = hit_share >= min_hits
    if kept.sum() < 2:
        raise ValueError(f"only {int(kept.sum())} views have {min_hits:.0%} of their landmarks on the mesh")
    error = np.full(v, np.inf)
    position = np.full((k, 3), np.nan)
    for _ in range(rounds):
        position = np.nanmedian(points[kept], axis=0)
        ruler = np.linalg.norm(position[scale_pair[0]] - position[scale_pair[1]])
        dist = np.linalg.norm(points - position[None], axis=2)  # (V, K)
        error = np.where(hit_share >= min_hits, np.nanmedian(dist, axis=1) / ruler, np.inf)
        new_kept = error <= max_error
        if new_kept.sum() < 2 or np.array_equal(new_kept, kept):
            kept = new_kept if new_kept.sum() >= 2 else kept
            break
        kept = new_kept
    position = np.nanmedian(points[kept], axis=0)
    dist = np.linalg.norm(points[kept] - position[None], axis=2)
    med = np.nanmedian(dist, axis=0)
    spread = 1.4826 * np.nanmedian(np.abs(dist - med[None]), axis=0)
    return {
        "position": position, "spread": spread, "error": error, "kept": kept,
        "n_views": np.sum(np.isfinite(points[kept][..., 0]), axis=0),
    }


def hull_mask(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    """uint8 mask (255 inside) of the convex hull of `pixels` (N, 2) in a `width` x `height`
    image - a region of landmarks (an eye, the lips) as the area they enclose."""
    import cv2

    mask = np.zeros((height, width), np.uint8)
    pts = np.round(np.asarray(pixels, dtype=np.float64)).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    return mask
