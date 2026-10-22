"""Ingest stage wrapper: raw source -> posed_images + sparse_points.

Dispatches on `params["mode"]` to the matching core implementation. This is the
only stage that knows the ingest paths differ - everything after it consumes the
posed_images schema and nothing else.

An optional `feature_masks` input (photo mode only) is a `masking` stage's output, whose
`masks/` is COLMAP's `--ImageReader.mask_path`, restricting SfM to the pixels those masks
keep. It is wired by photo_pipeline_moving_object.yaml; see that config for why a capture
of a moving object needs it.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from pipeline.core.ingest.photo_sfm import PhotoSfMConfig, run_photo_sfm
from pipeline.schemas.manifest import StageManifest
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = params.get("mode", "photo")
    workspace_dir = output_dir / "_workspace"
    stage_params = {k: v for k, v in params.items() if k != "mode"}

    if mode == "photo":
        source_path = input_dirs["source"]
        feature_masks = input_dirs.get("feature_masks")
        posed_images, sparse_points = run_photo_sfm(
            source_path,
            workspace_dir,
            PhotoSfMConfig(**stage_params),
            feature_mask_dir=(feature_masks / "masks") if feature_masks is not None else None,
        )
        shutil.copytree(workspace_dir / "undistorted" / "images", output_dir / "images", dirs_exist_ok=True)
    else:
        raise ValueError(f"Unknown ingest mode: {mode!r} (expected 'photo')")

    posed_images.save(output_dir)
    sparse_points.save(output_dir / "sparse_points.ply")

    outputs = {"cameras": "cameras.json", "images": "images", "sparse_points": "sparse_points.ply"}
    # Recorded, not consumed (mean_up_vector is derived on demand): having the axis in
    # the manifest makes a capture whose ups cancel out noticeable early.
    up = posed_images.mean_up_vector(output_dir)
    return finalize_stage(
        "ingest",
        output_dir,
        params,
        input_dirs,
        outputs=outputs,
        start_time=t0,
        extra={"mean_up_vector": [round(float(v), 6) for v in up]},
    )


def main() -> None:
    stage_cli("ingest", run)


if __name__ == "__main__":
    main()
