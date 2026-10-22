"""Per-view metric depth + surface normal priors via Metric3Dv2 (third_party/Metric3D).

A monocular geometry network is not fooled by what fools a photometric fit: a highlight
that slides as the camera moves, or a surface that gives two views nothing to match. It
predicts the shape of a cup as a cup from one photograph, where a photometric fit will
happily settle on something that explains the pixels and is not the surface. So the prior
supplies a shape signal immune to the ambiguity that corrupts the reconstruction, applied
through FastGS's depth prior term (see photo_pipeline_specular.yaml's header).

Metric3D is a *monocular* predictor: it sees one frame at a time and its depth is not
scale-consistent between views. That is fine here. FastGS's `depth_prior_loss`
(utils/loss_utils.py) solves a per-view weighted least squares for a scale and offset
(`omega * D_model + b ~ D_prior`) and detaches them before taking the residual, so only
the *shape* of the depth is supervised, never its absolute scale.

Loaded through torch.hub against the vendored checkout rather than the GitHub handle, so
the revision is pinned by the submodule and the mmcv patch there is actually used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

METRIC3D_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Metric3D"

# What the ViT checkpoints were trained at, and the focal length of the canonical camera
# their depth is expressed in - see hubconf.py's reference implementation.
_VIT_INPUT_SIZE = (616, 1064)
_CANONICAL_FOCAL = 1000.0
_IMAGENET_MEAN = (123.675, 116.28, 103.53)
_IMAGENET_STD = (58.395, 57.12, 57.375)

__all__ = ["Metric3DConfig", "load_metric3d", "predict_depth_normal"]


@dataclass
class Metric3DConfig:
    # vit_small is the smallest v2 checkpoint, and v2 is what predicts normals at all -
    # the ConvNeXt (v1) variants are depth-only. 62 ms a view, so the larger variants are
    # affordable too if the priors turn out to be worth sharpening.
    model: str = "metric3d_vit_small"
    device: str = "cuda"
    # Priors are kept at the network's own working resolution instead of being upsampled
    # to the training images: the extra pixels carry no information the network produced,
    # and FastGS's loader interpolates them up to the render size anyway. At 2000x1500
    # that is 10 MB a view rather than 60.
    store_at_native_resolution: bool = True


def load_metric3d(config: Metric3DConfig | None = None):
    config = config or Metric3DConfig()
    model = torch.hub.load(
        str(METRIC3D_ROOT), config.model, pretrain=True, source="local", trust_repo=True
    )
    return model.to(config.device).eval()


def _prepare(rgb: np.ndarray, fx: float, fy: float) -> tuple[torch.Tensor, list[int], float]:
    """Keep-ratio resize into the ViT's input size, pad the remainder with the mean
    colour, normalise. Returns the batch, the padding that has to come back off the
    prediction, and the focal length in the resized frame - which is what the
    de-canonical scale is computed from."""
    h, w = rgb.shape[:2]
    scale = min(_VIT_INPUT_SIZE[0] / h, _VIT_INPUT_SIZE[1] / w)
    resized = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    rh, rw = resized.shape[:2]
    pad_h, pad_w = _VIT_INPUT_SIZE[0] - rh, _VIT_INPUT_SIZE[1] - rw
    top, left = pad_h // 2, pad_w // 2
    pad_info = [top, pad_h - top, left, pad_w - left]
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=list(_IMAGENET_MEAN)
    )

    mean = torch.tensor(_IMAGENET_MEAN).float()[:, None, None]
    std = torch.tensor(_IMAGENET_STD).float()[:, None, None]
    batch = torch.from_numpy(padded.transpose(2, 0, 1)).float()
    batch = ((batch - mean) / std)[None]
    return batch, pad_info, fx * scale


def _unpad(x: torch.Tensor, pad_info: list[int]) -> torch.Tensor:
    """Trim the padding off a prediction. `x` is (..., H, W)."""
    top, bottom, left, right = pad_info
    return x[..., top : x.shape[-2] - bottom, left : x.shape[-1] - right]


@torch.no_grad()
def predict_depth_normal(
    model, image_path: Path, fx: float, fy: float, config: Metric3DConfig | None = None
) -> dict[str, torch.Tensor]:
    """Returns {'depth', 'conf', 'normal'} for one view, at the network's own resolution.

    `depth` is metric (metres) - the network predicts in a canonical camera of focal
    length 1000 and the result is rescaled by the view's real focal length. The depth
    prior loss realigns scale per view regardless, so this mostly matters for the depth's *shape*
    being right rather than its units.

    `normal` is a unit vector per pixel in camera space, sign convention unverified here
    - the stage checks it against the cameras rather than assuming it.
    """
    config = config or Metric3DConfig()
    # IMREAD_IGNORE_ORIENTATION: the same convention every other per-pixel reader in this
    # pipeline uses (masking_sam3._stage_video_frames, slf_fit._load_photo,
    # masking/run.py). COLMAP posed the *stored* buffer, so the intrinsics this function
    # is handed - and the images surface_recon renders - are in that frame. Without the
    # flag OpenCV applies the EXIF rotation and a phone capture comes back transposed:
    # cup's frames are 2000x1500 stored and read back 1500x2000, so the prior would be
    # predicted in a frame rotated 90 degrees from the one it is compared against, with
    # fx/fy belonging to the other orientation on top of that.
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise RuntimeError(f"Could not read {image_path}")
    rgb = bgr[:, :, ::-1]

    batch, pad_info, fx_resized = _prepare(rgb, fx, fy)
    depth, conf, out = model.inference({"input": batch.to(config.device)})

    depth = _unpad(depth.squeeze(1), pad_info).squeeze(0)
    conf = _unpad(conf.squeeze(1), pad_info).squeeze(0) if conf is not None else torch.ones_like(depth)
    # De-canonical: the prediction is in a camera of focal length 1000, this view is not.
    depth = depth * (fx_resized / _CANONICAL_FOCAL)

    normal = _unpad(out["prediction_normal"][:, :3], pad_info).squeeze(0)
    normal = torch.nn.functional.normalize(normal, dim=0)

    return {"depth": depth.float(), "conf": conf.float(), "normal": normal.float()}
