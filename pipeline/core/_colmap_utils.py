"""Internal helpers shared by photo_sfm.py and resize_dataset.py: run the `colmap`
CLI as a subprocess and parse its TXT model format. Not a public core module on its
own - both wrap COLMAP but produce different schema outputs (posed_images vs.
undistorted images + camera model), so the shared plumbing lives here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pipeline.schemas.point_cloud import PointCloud
from pipeline.schemas.posed_images import CameraIntrinsics, CameraPose, PosedImage, PosedImageSet


class ColmapError(RuntimeError):
    pass


def find_colmap_binary(configured: str = "colmap") -> str:
    resolved = shutil.which(configured)
    if resolved is None:
        raise ColmapError(
            f"COLMAP binary '{configured}' not found on PATH. Install it (e.g. "
            f"`conda install -c conda-forge colmap`) or pass an explicit path via config.colmap_binary."
        )
    return resolved


def run_colmap(binary: str, args: list[str]) -> None:
    cmd = [binary, *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ColmapError(
            f"`{' '.join(cmd)}` failed with exit code {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def convert_model_to_txt(binary: str, model_dir: Path, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        binary,
        [
            "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(out_dir),
            "--output_type", "TXT",
        ],
    )
    return out_dir


def pick_largest_model(sparse_dir: Path) -> Path:
    """COLMAP's mapper may reconstruct several disconnected models (0/, 1/, ...)
    when the image set doesn't fully connect; we pick the one with the largest
    points3D.bin as a cheap proxy for "most complete reconstruction"."""
    model_dirs = [p for p in sparse_dir.iterdir() if p.is_dir() and (p / "points3D.bin").exists()]
    if not model_dirs:
        raise ColmapError(f"No reconstructed model found under {sparse_dir}")
    return max(model_dirs, key=lambda p: (p / "points3D.bin").stat().st_size)


def parse_cameras_txt(path: Path) -> dict[int, CameraIntrinsics]:
    cameras: dict[int, CameraIntrinsics] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]
        cameras[camera_id] = CameraIntrinsics(model=model, width=width, height=height, params=params)
    return cameras


def parse_images_txt(path: Path) -> tuple[list[PosedImage], dict[int, list[tuple[float, float, int]]]]:
    """Returns (posed_images, points2d_by_image_id). points2d entries are
    (x, y, point3d_id); point3d_id == -1 means unmatched to any 3D point."""
    images: list[PosedImage] = []
    points2d_by_image: dict[int, list[tuple[float, float, int]]] = {}
    # Only comment lines are dropped here - NOT blank lines, since an image with zero
    # POINTS2D observations legitimately has a blank second line, and dropping it would
    # desync the header/points2d alternation for every image after it.
    lines = [l for l in path.read_text().splitlines() if not l.lstrip().startswith("#")]
    for i in range(0, len(lines), 2):
        header = lines[i].split()
        image_id = int(header[0])
        qw, qx, qy, qz = (float(x) for x in header[1:5])
        tx, ty, tz = (float(x) for x in header[5:8])
        camera_id = int(header[8])
        file_name = header[9]
        images.append(
            PosedImage(
                image_id=image_id,
                file_name=file_name,
                camera_id=camera_id,
                pose=CameraPose(qw=qw, qx=qx, qy=qy, qz=qz, tx=tx, ty=ty, tz=tz),
            )
        )
        raw_points = lines[i + 1].split()
        points: list[tuple[float, float, int]] = []
        for j in range(0, len(raw_points), 3):
            x, y, pid = float(raw_points[j]), float(raw_points[j + 1]), int(raw_points[j + 2])
            points.append((x, y, pid))
        points2d_by_image[image_id] = points
    return images, points2d_by_image


def parse_points3d_txt(
    path: Path, points2d_by_image: dict[int, list[tuple[float, float, int]]]
) -> PointCloud:
    import numpy as np

    xyz, rgb = [], []
    track_image_ids: list[list[int]] = []
    track_pixels: list[list[tuple[float, float]]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb.append([int(parts[4]) / 255.0, int(parts[5]) / 255.0, int(parts[6]) / 255.0])
        track = parts[8:]
        image_ids, pixels = [], []
        for j in range(0, len(track), 2):
            image_id, point2d_idx = int(track[j]), int(track[j + 1])
            image_ids.append(image_id)
            px, py, _ = points2d_by_image[image_id][point2d_idx]
            pixels.append((px, py))
        track_image_ids.append(image_ids)
        track_pixels.append(pixels)

    return PointCloud(
        points=np.array(xyz, dtype=np.float64),
        colors=np.array(rgb, dtype=np.float64),
        track_image_ids=track_image_ids,
        track_pixels=track_pixels,
    )


def convert_model_to_bin(binary: str, txt_dir: Path, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or txt_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        binary,
        [
            "model_converter",
            "--input_path", str(txt_dir),
            "--output_path", str(out_dir),
            "--output_type", "BIN",
        ],
    )
    return out_dir


def write_colmap_text_model(
    model_dir: Path, posed_images: PosedImageSet, point_cloud: PointCloud | None
) -> None:
    """Serializes a PosedImageSet (+ optional sparse PointCloud) back into COLMAP's
    TXT model format, so downstream stages (resize_dataset.py) can feed them into
    COLMAP tools like `image_undistorter` without depending on any ingest-internal
    artifacts.

    POINTS2D[]/TRACK[] entries are populated from `point_cloud.track_image_ids` /
    `track_pixels` when available. This matters even though MVS itself only reads
    poses/intrinsics: `patch_match_stereo`'s `__auto__` neighbor selection determines
    each image's "source images" from shared 3D point covisibility, so an empty
    points model makes every image believe it has no neighbors and silently produces
    zero depth maps.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    cameras_lines = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for camera_id, cam in posed_images.cameras.items():
        params_str = " ".join(str(p) for p in cam.params)
        cameras_lines.append(f"{camera_id} {cam.model} {cam.width} {cam.height} {params_str}")
    (model_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n")

    # Invert point_cloud's per-point tracks into per-image POINTS2D lists, and record
    # each point's TRACK[] as (image_id, point2d_idx) pairs referencing those lists.
    points2d_by_image: dict[int, list[tuple[float, float, int]]] = {
        img.image_id: [] for img in posed_images.images
    }
    point_tracks: list[list[tuple[int, int]]] = []
    if point_cloud is not None:
        has_tracks = point_cloud.track_image_ids is not None
        for point_id in range(len(point_cloud)):
            track_entries: list[tuple[int, int]] = []
            if has_tracks:
                image_ids = point_cloud.track_image_ids[point_id]
                pixels = point_cloud.track_pixels[point_id]
                for image_id, (x, y) in zip(image_ids, pixels):
                    if image_id not in points2d_by_image:
                        continue
                    point2d_idx = len(points2d_by_image[image_id])
                    points2d_by_image[image_id].append((x, y, point_id))
                    track_entries.append((image_id, point2d_idx))
            point_tracks.append(track_entries)

    images_lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "# POINTS2D[]"]
    for img in posed_images.images:
        p = img.pose
        images_lines.append(
            f"{img.image_id} {p.qw} {p.qx} {p.qy} {p.qz} {p.tx} {p.ty} {p.tz} "
            f"{img.camera_id} {img.file_name}"
        )
        points2d_str = " ".join(f"{x} {y} {pid}" for x, y, pid in points2d_by_image[img.image_id])
        images_lines.append(points2d_str)
    (model_dir / "images.txt").write_text("\n".join(images_lines) + "\n")

    points_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]"]
    if point_cloud is not None:
        for i in range(len(point_cloud)):
            x, y, z = point_cloud.points[i]
            if point_cloud.colors is not None:
                r, g, b = (int(c * 255) for c in point_cloud.colors[i])
            else:
                r, g, b = 128, 128, 128
            track_str = " ".join(f"{img_id} {idx}" for img_id, idx in point_tracks[i])
            points_lines.append(f"{i} {x} {y} {z} {r} {g} {b} 1.0 {track_str}")
    (model_dir / "points3D.txt").write_text("\n".join(points_lines) + "\n")


def load_posed_images_and_points(
    model_dir: Path, source: str, images_dir_name: str = "images"
) -> tuple[PosedImageSet, PointCloud]:
    cameras = parse_cameras_txt(model_dir / "cameras.txt")
    images, points2d_by_image = parse_images_txt(model_dir / "images.txt")
    point_cloud = parse_points3d_txt(model_dir / "points3D.txt", points2d_by_image)
    posed_images = PosedImageSet(source=source, images_dir=images_dir_name, cameras=cameras, images=images)
    return posed_images, point_cloud
