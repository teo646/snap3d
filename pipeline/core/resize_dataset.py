"""posed_images -> a flat, training-resolution COLMAP `dense/` dataset (`images/` +
`sparse/*.bin`), what surface_recon and masking read.

Downscales every photo so its longer side is at most `max_image_size` (intrinsics scaled to
match; smaller photos are copied) and writes the model in COLMAP's binary layout.
SIMPLE_PINHOLE cameras leave as PINHOLE because the per-axis rounding makes x and y scales differ.
Sparse point tracks are kept, with 2D coordinates scaled; the point loader drops points with
tracks shorter than 3.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from pipeline.core._colmap_utils import (
    convert_model_to_bin,
    find_colmap_binary,
    write_colmap_text_model,
)
from pipeline.schemas.point_cloud import PointCloud
from pipeline.schemas.posed_images import CameraIntrinsics, PosedImageSet

__all__ = ["ResizeDatasetConfig", "run_resize_dataset", "scaled_pinhole", "scaled_size"]


@dataclass
class ResizeDatasetConfig:
    colmap_binary: str = "colmap"  # converts the text model to binary
    max_image_size: int = 2000


def scaled_size(width: int, height: int, max_image_size: int) -> tuple[int, int]:
    """(width, height) with the longer side brought down to `max_image_size`; never enlarges."""
    scale = max_image_size / max(width, height)
    if scale >= 1.0:
        return width, height
    return max(1, round(width * scale)), max(1, round(height * scale))


def scaled_pinhole(camera: CameraIntrinsics, width: int, height: int) -> CameraIntrinsics:
    """`camera` as a PINHOLE camera of `width` x `height`, intrinsics scaled per axis."""
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        fx = fy = f
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    else:
        raise RuntimeError(
            f"camera model {camera.model} is not PINHOLE or SIMPLE_PINHOLE - ingest should have "
            "undistorted it already"
        )
    sx, sy = width / camera.width, height / camera.height
    return CameraIntrinsics(
        model="PINHOLE", width=width, height=height, params=[fx * sx, fy * sy, cx * sx, cy * sy]
    )


def run_resize_dataset(
    posed_images: PosedImageSet,
    sparse_points: PointCloud | None,
    images_dir: Path,
    workspace_dir: Path,
    config: ResizeDatasetConfig | None = None,
) -> Path:
    """Returns `workspace_dir/dense` with `images/` + `sparse/{cameras,images,points3D}.bin`.
    `images_dir` is the stage directory `posed_images` was loaded from.
    """
    config = config or ResizeDatasetConfig()
    binary = find_colmap_binary(config.colmap_binary)
    dense_dir = workspace_dir / "dense"
    shutil.rmtree(workspace_dir, ignore_errors=True)
    out_images = dense_dir / "images"
    out_images.mkdir(parents=True)

    # One output size per camera, taken from its photos.
    size_of_camera: dict[int, tuple[int, int]] = {}
    for img in posed_images.images:
        src = posed_images.image_path(images_dir, img)
        frame = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if frame is None:
            raise FileNotFoundError(f"could not read {src}")
        h, w = frame.shape[:2]
        camera = posed_images.cameras[img.camera_id]
        if (w, h) != (camera.width, camera.height):
            raise RuntimeError(
                f"{img.file_name} is {w}x{h} but camera {img.camera_id} says "
                f"{camera.width}x{camera.height}"
            )
        new_w, new_h = scaled_size(w, h, config.max_image_size)
        if size_of_camera.setdefault(img.camera_id, (new_w, new_h)) != (new_w, new_h):
            raise RuntimeError(f"photos of camera {img.camera_id} come out at different sizes")

        dest = out_images / img.file_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if (new_w, new_h) == (w, h):
            shutil.copy2(src, dest)
        else:
            cv2.imwrite(str(dest), cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA))

    cameras = {
        camera_id: scaled_pinhole(posed_images.cameras[camera_id], *size_of_camera[camera_id])
        for camera_id in size_of_camera
    }
    for camera_id, cam in cameras.items():
        old = posed_images.cameras[camera_id]
        if (old.width, old.height) != (cam.width, cam.height) or old.model != cam.model:
            print(
                f"resize_dataset: camera {camera_id} {old.model} {old.width}x{old.height} -> "
                f"PINHOLE {cam.width}x{cam.height} (fx={cam.params[0]:.2f} fy={cam.params[1]:.2f})"
            )
    resized = PosedImageSet(
        source=posed_images.source, images_dir="images", cameras=cameras, images=posed_images.images
    )
    points = sparse_points
    if sparse_points is not None and sparse_points.track_image_ids is not None:
        # Each pixel scales by the factors of its image's camera.
        factor = {}
        for img in posed_images.images:
            old, new = posed_images.cameras[img.camera_id], cameras[img.camera_id]
            factor[img.image_id] = (new.width / old.width, new.height / old.height)
        points = PointCloud(
            points=sparse_points.points, colors=sparse_points.colors, normals=sparse_points.normals,
            track_image_ids=sparse_points.track_image_ids,
            track_pixels=[
                [(x * factor[i][0], y * factor[i][1]) for i, (x, y) in zip(ids, pixels)]
                for ids, pixels in zip(sparse_points.track_image_ids, sparse_points.track_pixels)
            ],
        )

    model_txt = workspace_dir / "sparse_txt"
    write_colmap_text_model(model_txt, resized, points)
    convert_model_to_bin(binary, model_txt, dense_dir / "sparse")
    return dense_dir
