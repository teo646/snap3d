"""``posed_images`` schema: the contract between ingest and everything downstream.

Any ingest implementation only has to produce a directory containing::

    <stage_dir>/images/<file_name>   # the (possibly undistorted) source frames
    <stage_dir>/cameras.json         # a serialized PosedImageSet

Poses are COLMAP world-to-camera: ``X_cam = R(q) @ X_world + t``, camera looking down
+Z with +Y down, same as COLMAP's own images.txt/images.bin. Those poses describe the
image *as stored*, which is not always the image as shot - see `exif_up_orientation`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

# Below this, the mean of the per-camera up vectors has cancelled itself out and its
# direction carries no information - see PosedImageSet.mean_up_vector.
_UP_MEAN_MIN_NORM = 1e-3

# EXIF orientation -> (row of R, sign) giving the world direction that is "up" in the
# *displayed* photo. A phone shot in portrait stores a landscape buffer plus an
# orientation tag; COLMAP poses the stored buffer, so for those the up axis is a row of
# R the untagged case never touches. Derived by composing each tag's transform:
#   1,2 (no rotation)      display up = -y_buffer = -R[1]   (the plain case)
#   3,4 (180 / flip)       display up = +y_buffer = +R[1]
#   5,6 (90 CW)            display up = -x_buffer = -R[0]
#   7,8 (270 CW)           display up = +x_buffer = +R[0]
# Mirroring never moves the up axis, only the handedness, which an up vector ignores.
_EXIF_UP_AXIS = {1: (1, -1), 2: (1, -1), 3: (1, 1), 4: (1, 1),
                 5: (0, -1), 6: (0, -1), 7: (0, 1), 8: (0, 1)}
_EXIF_ORIENTATION_TAG = 0x0112


def exif_up_orientation(path: str | Path) -> int:
    """This image's EXIF orientation tag, or 1 ("as stored") when it has none or cannot
    be read. Only the tag is read - the pixels are never rotated, because COLMAP solved
    its poses against the stored buffer and re-encoding would invalidate them."""
    try:
        with Image.open(path) as img:
            value = img.getexif().get(_EXIF_ORIENTATION_TAG)
    except (OSError, UnidentifiedImageError, ValueError):
        return 1
    return value if value in _EXIF_UP_AXIS else 1


CameraModel = Literal[
    "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"
]


class CameraIntrinsics(BaseModel):
    model: CameraModel
    width: int
    height: int
    params: list[float]  # semantics depend on `model`, matches COLMAP's convention

    def as_matrix(self) -> np.ndarray:
        """3x3 intrinsic matrix K, valid for PINHOLE-family models (fx, fy, cx, cy, ...)."""
        if self.model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "RADIAL"):
            fx, fy, cx, cy = self.params[0], self.params[1], self.params[2], self.params[3]
        elif self.model == "SIMPLE_PINHOLE" or self.model == "SIMPLE_RADIAL":
            fx = fy = self.params[0]
            cx, cy = self.params[1], self.params[2]
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported camera model: {self.model}")
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


class CameraPose(BaseModel):
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float

    def rotation_matrix(self) -> np.ndarray:
        q = np.array([self.qw, self.qx, self.qy, self.qz], dtype=np.float64)
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ],
            dtype=np.float64,
        )

    def world_to_camera(self) -> np.ndarray:
        """4x4 extrinsic matrix mapping world-space points into the camera frame."""
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = self.rotation_matrix()
        M[:3, 3] = [self.tx, self.ty, self.tz]
        return M

    def camera_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_camera())

    def camera_center(self) -> np.ndarray:
        R = self.rotation_matrix()
        t = np.array([self.tx, self.ty, self.tz])
        return -R.T @ t

    def up_vector(self, exif_orientation: int = 1) -> np.ndarray:
        """This camera's up as a unit vector in *world* space. COLMAP's +Y is **down**,
        so for an untagged image the up axis is -Y_cam, which through X_world = R^T
        X_cam is the negated second row of R. `exif_orientation` is the photo's EXIF
        tag: a rotated one was shot with a different edge of the sensor pointing up, so
        the up axis moves to another row of R (see _EXIF_UP_AXIS)."""
        row, sign = _EXIF_UP_AXIS.get(exif_orientation, (1, -1))
        return sign * self.rotation_matrix()[row, :]


class PosedImage(BaseModel):
    image_id: int
    file_name: str  # relative to `images_dir`
    camera_id: int
    pose: CameraPose


class PosedImageSet(BaseModel):
    source: Literal["photo_sfm"]
    images_dir: str = "images"
    cameras: dict[int, CameraIntrinsics]
    images: list[PosedImage]

    def save(self, stage_dir: str | Path) -> Path:
        out_path = Path(stage_dir) / "cameras.json"
        out_path.write_text(self.model_dump_json(indent=2))
        return out_path

    @classmethod
    def load(cls, stage_dir: str | Path) -> "PosedImageSet":
        path = Path(stage_dir) / "cameras.json"
        return cls.model_validate(json.loads(path.read_text()))

    def image_path(self, stage_dir: str | Path, image: PosedImage) -> Path:
        return Path(stage_dir) / self.images_dir / image.file_name

    def mean_up_vector(self, stage_dir: str | Path | None = None) -> np.ndarray:
        """The capture's world up: the normalized mean of every camera's own up vector.
        Derived on demand, so it stays correct if poses are rewritten. This is the
        scene-global "up" a reconstruction's world frame does not record; the viewers
        orbit about it.

        Pass the ingest `stage_dir` (the one `image_path` takes) whenever the frames are
        on disk: each image's EXIF orientation is then read and its up axis corrected.
        Without it every photo is assumed stored as shot, which silently rotates the
        result by 90 degrees for a portrait phone capture.

        A capture whose cameras roll all the way around their view axis cancels the
        mean out and leaves its direction as noise. Below `_UP_MEAN_MIN_NORM` this
        falls back to the dominant axis of the up vectors (top eigenvector of
        sum(u u^T)), whose sign - being sign-free - is fixed by convention.
        """
        if not self.images:
            raise ValueError("cannot derive an up vector from an empty PosedImageSet")
        orientations = (
            [exif_up_orientation(self.image_path(stage_dir, img)) for img in self.images]
            if stage_dir is not None
            else [1] * len(self.images)
        )
        ups = np.stack([img.pose.up_vector(o) for img, o in zip(self.images, orientations)])
        mean = ups.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm >= _UP_MEAN_MIN_NORM:
            return mean / norm
        axis = np.linalg.eigh(ups.T @ ups)[1][:, -1]
        return axis * np.sign(axis[np.argmax(np.abs(axis))])
