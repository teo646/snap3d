"""Highlight-removal stage: a folder of images -> the same images with the specular
highlights taken out (SHIQ), under the same relative paths and at the same size.

It exists so surface_recon can be fitted to highlight-free pixels while everything that
learns appearance keeps the photographs: view-dependent colour is what `slf` fits, and a
highlight is exactly that signal. In photo_pipeline_specular.yaml the stage sits after `resize_dataset`;
`masking` and `surface_recon` read its images, `slf` still reads ingest's originals.

Its `images` input is a stage output holding images under `images_subdir`, like
`masking`'s. How the result is made (the network's raw output, or the original with the
network's smooth correction applied) is in pipeline/core/highlight_removal_shiq.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pipeline.core.highlight_removal_shiq import SHIQConfig, remove_highlights
from pipeline.core.masking_sam3 import image_names
from pipeline.schemas.manifest import StageManifest
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_params = dict(params)
    images_subdir = stage_params.pop("images_subdir", "")
    config = SHIQConfig(**stage_params)

    images_dir = input_dirs["images"] / images_subdir
    names = image_names(images_dir)
    out_dir = output_dir / "images"
    remove_highlights([images_dir / n for n in names], [out_dir / n for n in names], config)
    print(f"highlight_removal: {len(names)} images ({config.mode}, sigma {config.detail_sigma})")

    return finalize_stage(
        "highlight_removal", output_dir, params, input_dirs,
        outputs={"images": "images"}, start_time=t0, extra={"images": len(names)},
    )


def main() -> None:
    stage_cli("highlight_removal", run)


if __name__ == "__main__":
    main()
