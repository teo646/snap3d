"""Masking stage: a folder of images -> per-image masks (and optionally cut images).

Every image under the `images` input (optionally `images_subdir`) is segmented in file-name
order as one SAM3 video with `text_prompt` (see pipeline.core.masking_sam3).

Output `masks/<image relative path>.png` (255 keep, 0 ignore) at each image's own pixel size.

With `write_masked_images`, also writes an RGBA `images/` folder with everything outside the
(dilated, `mask_dilation_px`) mask replaced by `mask_fill_color` and alpha = the mask, for
surface_recon to train on. `masks/` is always SAM3's boundary as is.

`highlight_threshold` additionally cuts pixels inside the mask whose luminance exceeds it.
It does not combine with `lambda_mask_alpha`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from PIL import Image

from pipeline.core.masking_sam3 import SAM3MaskingConfig, fullres_mask, image_names, segment_views
from pipeline.schemas.manifest import StageManifest
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Controls this stage's own handling of the result, not SAM3 segmentation.
    stage_params = dict(params)
    images_subdir = stage_params.pop("images_subdir", "")
    write_masked_images = stage_params.pop("write_masked_images", False)
    mask_dilation_px = stage_params.pop("mask_dilation_px", 0)
    mask_fill_color = tuple(stage_params.pop("mask_fill_color", (0, 0, 0)))
    highlight_threshold = stage_params.pop("highlight_threshold", None)
    config = SAM3MaskingConfig(**stage_params)

    # `source` is the capture itself (`uses_source_input`).
    images_dir = input_dirs.get("images", input_dirs.get("source")) / images_subdir
    names = image_names(images_dir)
    masks = segment_views(images_dir, names, config)

    masks_dir = output_dir / "masks"
    kept_share = []
    for name in names:
        mask = fullres_mask(masks[name], images_dir / name)
        dest = masks_dir / f"{name}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), mask)
        kept_share.append(float((mask > 0).mean()))
    print(f"masking {config.text_prompt!r}: {len(names)} images, median kept fraction {np.median(kept_share):.3f}")

    outputs = {"masks": "masks"}
    if write_masked_images:
        images_out_dir = output_dir / "images"
        kernel = None
        if mask_dilation_px > 0:
            size = 2 * mask_dilation_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        fill = np.array(mask_fill_color, dtype=np.uint8)
        for name in names:
            src = images_dir / name
            # IMREAD_IGNORE_ORIENTATION: COLMAP posed the stored buffer.
            frame = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if frame is None:
                raise FileNotFoundError(f"could not read {src} to build its masked training image")
            m = (cv2.imread(str(masks_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            if kernel is not None:
                m = cv2.dilate(m, kernel)
            keep = m.astype(bool)
            n_highlight = 0
            if highlight_threshold is not None:
                # BT.601 luma, so a saturated colour is not flagged as a highlight.
                luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bright = luma > highlight_threshold
                n_highlight = int((keep & bright).sum())
                keep = keep & ~bright
            masked = np.where(keep[:, :, None], frame, fill)
            # RGBA: alpha = keep, read by FastGS as Camera.mask. Written under the image's own name
            # because the sparse model references it.
            rgba = np.dstack([masked[:, :, ::-1], (keep.astype(np.uint8) * 255)])  # BGR -> RGB
            dest = images_out_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba, mode="RGBA").save(dest, format="PNG")
            if highlight_threshold is not None and n_highlight:
                print(f"  {name}: {n_highlight} px above luma {highlight_threshold} cut as highlight")
        outputs["images"] = "images"

    return finalize_stage(
        "masking", output_dir, params, input_dirs, outputs=outputs, start_time=t0,
        extra={"images": len(names), "median_kept_fraction": float(np.median(kept_share))},
    )


def main() -> None:
    stage_cli("masking", run)


if __name__ == "__main__":
    main()
