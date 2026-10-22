"""Resize-dataset stage wrapper: posed_images -> a flat, training-resolution dense/ workspace
(images/ + sparse/*.bin) for surface_recon and masking (see pipeline/core/resize_dataset.py).
Stages that need the full-resolution originals read `posed_images: ingest` instead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pipeline.core.resize_dataset import ResizeDatasetConfig, run_resize_dataset
from pipeline.schemas.manifest import StageManifest
from pipeline.schemas.point_cloud import PointCloud
from pipeline.schemas.posed_images import PosedImageSet
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    ingest_dir = input_dirs["posed_images"]

    posed_images = PosedImageSet.load(ingest_dir)
    sparse_points_path = ingest_dir / "sparse_points.ply"
    sparse_points = PointCloud.load(sparse_points_path) if sparse_points_path.exists() else None

    run_resize_dataset(
        posed_images, sparse_points, ingest_dir, output_dir / "_workspace", ResizeDatasetConfig(**params)
    )

    return finalize_stage("resize_dataset", output_dir, params, input_dirs, outputs={}, start_time=t0)


def main() -> None:
    stage_cli("resize_dataset", run)


if __name__ == "__main__":
    main()
