"""Photo set -> (PosedImageSet, PointCloud) via COLMAP structure-from-motion.

Shells out to the `colmap` CLI (feature extraction -> matching -> mapping -> undistortion) and
parses the sparse model into this project's schemas. Cameras come out PINHOLE, except
`camera_model: SIMPLE_PINHOLE` captures, which stay SIMPLE_PINHOLE until `resize_dataset`.
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.core._colmap_utils import (
    ColmapError,
    convert_model_to_txt,
    find_colmap_binary,
    load_posed_images_and_points,
    pick_largest_model,
    run_colmap,
)
from pipeline.schemas.point_cloud import PointCloud
from pipeline.schemas.posed_images import PosedImageSet

__all__ = ["PhotoSfMConfig", "run_photo_sfm", "run_sfm_model", "ColmapError"]


@dataclass
class PhotoSfMConfig:
    colmap_binary: str = "colmap"
    camera_model: str = "SIMPLE_RADIAL"  # undistorted to PINHOLE afterwards
    single_camera: bool = True  # every photo came from the same camera/lens
    use_gpu: bool = True
    matcher: str = "exhaustive"  # {"exhaustive", "sequential"}
    extra_feature_extractor_args: list[str] = field(default_factory=list)
    extra_mapper_args: list[str] = field(default_factory=list)


def run_photo_sfm(
    images_dir: Path,
    workspace_dir: Path,
    config: PhotoSfMConfig | None = None,
    feature_mask_dir: Path | None = None,
) -> tuple[PosedImageSet, PointCloud]:
    """Runs COLMAP SfM + undistortion.

    `images_dir`: directory of input photos.
    `feature_mask_dir`: optional `--ImageReader.mask_path`, a directory of `<image name>.png`
    masks (0 = ignore) restricting keypoint detection; the returned images are still whole frames.
    `workspace_dir`: scratch + output directory. On success it contains:
        database.db          - COLMAP feature/match database
        sparse_raw/<n>/      - raw reconstructions
        undistorted/images/  - PINHOLE-undistorted images
        undistorted/sparse/  - undistorted sparse model (TXT)
    Returns the parsed (PosedImageSet, PointCloud).
    """
    config = config or PhotoSfMConfig()
    binary = find_colmap_binary(config.colmap_binary)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    database_path = workspace_dir / "database.db"
    undistorted_dir = workspace_dir / "undistorted"

    best_model = run_sfm_model(images_dir, workspace_dir, config, feature_mask_dir)
    _undistort_images(binary, images_dir, best_model, undistorted_dir)

    undistorted_model_txt = convert_model_to_txt(
        binary, undistorted_dir / "sparse", out_dir=undistorted_dir / "sparse_txt"
    )
    posed_images, point_cloud = load_posed_images_and_points(
        undistorted_model_txt, source="photo_sfm", images_dir_name="images"
    )

    n_total = _count_database_images(database_path)
    n_registered = len(posed_images.images)
    note = ""
    if n_registered < n_total:
        note = (
            f" ({n_total - n_registered} dropped - either unregistered, or in a smaller "
            "disconnected model than the one picked; see pick_largest_model)"
        )
    print(f"photo_sfm: registered {n_registered}/{n_total} images{note}")

    return posed_images, point_cloud


def run_sfm_model(
    images_dir: Path,
    workspace_dir: Path,
    config: PhotoSfMConfig,
    feature_mask_dir: Path | None = None,
    image_names: list[str] | None = None,
) -> Path:
    """Feature extraction -> matching -> incremental mapping, returning the directory of the
    largest model (still in the raw camera model; see `_undistort_images`).

    `image_names` restricts the run to those images, relative to `images_dir`.

    Creates `workspace_dir/{database.db, sparse_raw/}`; `workspace_dir` must not hold a previous
    run's database.
    """
    binary = find_colmap_binary(config.colmap_binary)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    database_path = workspace_dir / "database.db"
    sparse_raw_dir = workspace_dir / "sparse_raw"
    sparse_raw_dir.mkdir(exist_ok=True)

    image_list_path = None
    if image_names is not None:
        image_list_path = workspace_dir / "image_list.txt"
        image_list_path.write_text("\n".join(image_names) + "\n")

    _extract_features(binary, config, database_path, images_dir, feature_mask_dir, image_list_path)
    _match_features(binary, config, database_path)
    _run_mapper(binary, config, database_path, images_dir, sparse_raw_dir)
    print(f"photo_sfm: {workspace_dir.name}: {_count_database_images(database_path)} images in database")
    return pick_largest_model(sparse_raw_dir)


def _count_database_images(database_path: Path) -> int:
    """Every image `feature_extractor` saw, registered or not."""
    conn = sqlite3.connect(str(database_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    finally:
        conn.close()


def _extract_features(
    binary: str,
    config: PhotoSfMConfig,
    database_path: Path,
    images_dir: Path,
    feature_mask_dir: Path | None = None,
    image_list_path: Path | None = None,
) -> None:
    run_colmap(
        binary,
        [
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            *(["--image_list_path", str(image_list_path)] if image_list_path is not None else []),
            "--ImageReader.camera_model", config.camera_model,
            "--ImageReader.single_camera", "1" if config.single_camera else "0",
            *(["--ImageReader.mask_path", str(feature_mask_dir)] if feature_mask_dir is not None else []),
            "--FeatureExtraction.use_gpu", "1" if config.use_gpu else "0",
            *config.extra_feature_extractor_args,
        ],
    )


def _match_features(binary: str, config: PhotoSfMConfig, database_path: Path) -> None:
    matcher_cmd = "exhaustive_matcher" if config.matcher == "exhaustive" else "sequential_matcher"
    run_colmap(
        binary,
        [
            matcher_cmd,
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "1" if config.use_gpu else "0",
        ],
    )


def _run_mapper(
    binary: str, config: PhotoSfMConfig, database_path: Path, images_dir: Path, sparse_dir: Path
) -> None:
    run_colmap(
        binary,
        [
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            *config.extra_mapper_args,
        ],
    )
    if not any(sparse_dir.iterdir()):
        raise ColmapError(
            f"COLMAP mapper produced no reconstruction under {sparse_dir}. "
            f"Likely too few images or insufficient visual overlap between views."
        )


def _undistort_images(binary: str, images_dir: Path, model_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    run_colmap(
        binary,
        [
            "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(model_dir),
            "--output_path", str(output_dir),
            "--output_type", "COLMAP",
        ],
    )
