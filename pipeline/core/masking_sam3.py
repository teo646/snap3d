"""SAM3 text-prompted per-view object masking.

`segment_views` prompts SAM3 with a text description of the object on the first image, then
propagates it through the rest of the sequence (file-name order) with its video predictor. A
frame's mask is the union of every tracked instance. A frame the tracker loses is re-segmented
on its own with the same prompt; if that also finds nothing, its mask is left empty with a warning.
"""

from __future__ import annotations

import gc
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

__all__ = ["SAM3MaskingConfig", "fullres_mask", "image_names", "segment_views"]

_VIDEO_FRAME_EXTS = (".jpg", ".jpeg")


@dataclass
class SAM3MaskingConfig:
    text_prompt: str  # open-vocabulary description of the object to segment, e.g. "apple"
    checkpoint_path: str | None = None  # None -> auto-download from Hugging Face (gated)
    # Longest side SAM3 is given; its video predictor's cost scales with pixels x frames.
    max_image_size: int = 2000


_SOURCE_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def image_names(images_dir: Path) -> list[str]:
    """Every image under `images_dir` as a posix path relative to it, in sorted order."""
    names = sorted(
        p.relative_to(images_dir).as_posix()
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _SOURCE_IMAGE_EXTS
    )
    if not names:
        raise RuntimeError(f"No images ({', '.join(_SOURCE_IMAGE_EXTS)}) found under {images_dir}")
    return names


def fullres_mask(mask: np.ndarray, photo_path: Path, dilation_px: int = 0) -> np.ndarray:
    """`mask` (bool, at the resolution `segment_views` ran at) as a 0/255 uint8 image at the
    photo's stored pixel size (EXIF rotation not applied), grown by `dilation_px` pixels."""
    from PIL import Image

    with Image.open(photo_path) as im:
        width, height = im.size
    m = mask.astype(np.uint8) * 255
    if m.shape != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    if dilation_px > 0:
        size = 2 * dilation_px + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return m


def _load_video_predictor(config: SAM3MaskingConfig):
    # SAM3's video predictor hardcodes CUDA. checkpoint_path=None downloads `facebook/sam3`
    # (gated: request access and `hf auth login`).
    from sam3.model_builder import build_sam3_video_predictor

    return build_sam3_video_predictor(checkpoint_path=config.checkpoint_path)


def _stage_video_frames(images_dir: Path, names: list[str], frames_dir: Path, max_image_size: int = 0) -> None:
    """SAM3's video predictor wants a directory of `<frame_index>.jpg` numbered from 0. JPEGs that
    need no downscale are symlinked, anything else is re-encoded."""
    for i, name in enumerate(names):
        src = images_dir / name
        dst = frames_dir / f"{i}.jpg"
        # IMREAD_IGNORE_ORIENTATION keeps the pixel layout COLMAP posed.
        frame = None
        if max_image_size > 0:
            frame = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            scale = max_image_size / max(frame.shape[:2])
            if scale >= 1.0:
                frame = None  # already small enough
            else:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if frame is None and src.suffix.lower() in _VIDEO_FRAME_EXTS:
            dst.symlink_to(src.resolve())
            continue
        if frame is None:
            frame = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv2.imwrite(str(dst), frame)


def segment_views(images_dir: Path, names: list[str], config: SAM3MaskingConfig) -> dict[str, np.ndarray]:
    """Returns {name: bool mask (H, W)}: the union of every instance matching `config.text_prompt`
    on the first frame, tracked through `names`. Masks are at SAM3's (possibly downscaled)
    resolution; see `fullres_mask`."""
    if not names:
        return {}

    predictor = _load_video_predictor(config)
    try:
        with tempfile.TemporaryDirectory(prefix="sam3_video_frames_") as frames_dir_str:
            frames_dir = Path(frames_dir_str)
            _stage_video_frames(images_dir, names, frames_dir, config.max_image_size)

            session_id = predictor.handle_request(
                {"type": "start_session", "resource_path": str(frames_dir)}
            )["session_id"]
            try:
                prompt_response = predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": config.text_prompt,
                    }
                )
                if len(prompt_response["outputs"]["out_obj_ids"]) == 0:
                    raise ValueError(
                        f"SAM3 found no object matching text prompt {config.text_prompt!r} on the first frame"
                    )

                masks: dict[str, np.ndarray] = {}
                lost_frames: list[int] = []  # frames propagation lost track on
                for response in predictor.handle_stream_request(
                    {"type": "propagate_in_video", "session_id": session_id}
                ):
                    frame_idx = response["frame_index"]
                    binary_masks = np.asarray(response["outputs"]["out_binary_masks"])
                    if len(binary_masks) > 0:
                        masks[names[frame_idx]] = np.any(binary_masks > 0, axis=0)
                    else:
                        lost_frames.append(frame_idx)

                # A lost frame is re-segmented on its own with the same text prompt.
                for frame_idx in lost_frames:
                    name = names[frame_idx]
                    retry = predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": frame_idx,
                            "text": config.text_prompt,
                        }
                    )
                    binary_masks = np.asarray(retry["outputs"]["out_binary_masks"])
                    if len(binary_masks) > 0:
                        masks[name] = np.any(binary_masks > 0, axis=0)
                        print(f"  {name}: video propagation lost it, re-segmented standalone")
                    else:
                        # Still nothing: empty mask at the resolution SAM3 was given.
                        h, w = cv2.imread(str(frames_dir / f"{frame_idx}.jpg")).shape[:2]
                        masks[name] = np.zeros((h, w), dtype=bool)
                        print(f"  WARNING {name}: no object found even standalone - empty mask")
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
    finally:
        # Drop the predictor reference before empty_cache() so the weights are returned to the driver.
        predictor.shutdown()
        del predictor
        gc.collect()
        torch.cuda.empty_cache()

    return masks
