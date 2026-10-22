"""Masking stage: a folder of images -> per-image masks (and optionally cut images).

The one masking stage, used wherever the pipeline segments something. Its `images` input
is a stage output (or the capture itself) holding images, optionally under
`images_subdir`; every image under it is segmented in file-name order, as one SAM3
video, with the text prompt `text_prompt`. A frame the tracker loses is re-segmented on
its own with that same prompt (see pipeline.core.masking_sam3).

Output `masks/<image relative path>.png` - COLMAP's own `--ImageReader.mask_path` layout
(255 keep, 0 ignore), at each image's own pixel size - so the same files serve as COLMAP
feature-detection masks before SfM and as the object silhouette after it. A detection
mask is not a cut: COLMAP still reads the original pixels, it just finds no keypoints
outside the mask. Blacking the background out instead would trade one problem for
another - a hard silhouette edge is exactly the high-contrast corner SIFT likes.

A pipeline can run it before SfM (to keep the background out of feature matching) and again
after `resize_dataset` so SAM3 segments the exact (possibly downscaled)
images surface_recon trains on and no mask-resize step is needed downstream.

With `write_masked_images`, also writes an `images/` folder of the same frames with
everything outside the (dilated) mask replaced by `mask_fill_color` - a literal
background cut, for surface_recon to train on directly (photo_pipeline.yaml). FastGS's
losses are full-frame, so the only way to stop the background competing for them is to
remove it from the pixels before training sees them. These are written as RGBA with
alpha = the mask, which FastGS loads as `Camera.mask`: the FastGS patch supervises
rendered alpha against it under `lambda_mask_alpha`, so the silhouette constrains
geometry and not only colour. See surface_recon_fastpgsr for the stage
that trains on this folder, and its module docstring for the tradeoffs.

`mask_dilation_px` grows the region kept in those cut images only; `masks/` is always
SAM3's boundary as it is.

`highlight_threshold`, given alongside it, additionally cuts any pixel *inside* the
object mask whose luminance exceeds it - a specular highlight. A highlight that slides
across a glossy surface as the camera moves reads as geometry that moves with it;
crudely removing it from the training target sidesteps that the same blunt way
background removal does, at the same cost (another hard edge, now in the middle of the
object). `masks/` is unchanged: a highlight is still real surface. It does not currently
combine with `lambda_mask_alpha`: a cut highlight is `mask_fill_color` in the RGB and 0
in the alpha, which the alpha term reads as "no geometry here" and tries to punch
through the object. Making them work together needs a second mask. Until then use one
or the other.
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

    # Pulled out of params before building SAM3MaskingConfig: they control this stage's
    # own handling of the result, not SAM3 segmentation.
    stage_params = dict(params)
    images_subdir = stage_params.pop("images_subdir", "")
    write_masked_images = stage_params.pop("write_masked_images", False)
    mask_dilation_px = stage_params.pop("mask_dilation_px", 0)
    mask_fill_color = tuple(stage_params.pop("mask_fill_color", (0, 0, 0)))
    highlight_threshold = stage_params.pop("highlight_threshold", None)
    config = SAM3MaskingConfig(**stage_params)

    # `source` is what the runner calls the capture itself (`uses_source_input`).
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
            # IMREAD_IGNORE_ORIENTATION: same convention as every other per-pixel reader
            # in this pipeline - COLMAP posed the stored buffer, not the EXIF-rotated
            # display orientation.
            frame = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if frame is None:
                raise FileNotFoundError(f"could not read {src} to build its masked training image")
            m = (cv2.imread(str(masks_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            if kernel is not None:
                m = cv2.dilate(m, kernel)
            keep = m.astype(bool)
            n_highlight = 0
            if highlight_threshold is not None:
                # Standard BT.601 luma, not a plain channel average, so a saturated red
                # or blue surface isn't flagged as a highlight just for being bright in
                # one channel - a specular highlight is bright in all three.
                luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bright = luma > highlight_threshold
                n_highlight = int((keep & bright).sum())
                keep = keep & ~bright
            masked = np.where(keep[:, :, None], frame, fill)
            # RGBA: alpha = keep, so FastGS's Camera.mask (read from an RGBA image's alpha
            # by scene/cameras.py's process_image) is the same region the RGB was cut to.
            # Written as PNG content under the image's own name: the sparse model
            # references that name, and PIL sniffs the real format from the bytes.
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
