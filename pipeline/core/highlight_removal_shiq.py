"""Specular-highlight removal with SHIQ (JSHDR, third_party/SHIQ), keeping the photo's detail.

A highlight that slides across a glossy surface as the camera moves reads, to a photometric
fit, as geometry that moves with it. JSHDR (Fu et al., CVPR 2021) predicts the highlight-free
(diffuse) image from one photo, which is what the reconstruction should be fitted to.

The network's own output is soft, though: on a 2000x1500 frame of `cup` its high-frequency
energy is 0.51 against the photo's 2.53, and print patterns and the tile's grain are smeared.
Training Fast-PGSR on that would take away the texture its multi-view NCC loss matches. What
the network is trusted with is the *change* it makes - the highlight coming out - and that is
smooth, so only its low-frequency part is applied to the sharp original:

    result = original + blur(shiq - original, sigma)

This restores the original's detail exactly (high-frequency energy 2.54) while carrying over
the highlight reduction (the brightest 0.1% of pixels: original 249, SHIQ 210, this 213 at
sigma 10). A highlight's sharpest core is the part a blur of the correction cannot reach, so
sigma trades how much of a narrow highlight comes out against how much of the network's own
softness gets in; `mode="shiq"` returns the network output as it is.

The network runs in its own conda env on the CPU - see `_shiq_worker.py`.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline.core.conda_env import env_python

__all__ = ["SHIQConfig", "merge_shiq_detail", "remove_highlights"]

SHIQ_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "SHIQ"
_WORKER = Path(__file__).with_name("_shiq_worker.py")


@dataclass
class SHIQConfig:
    python_bin: str = env_python("shiq")
    # "hybrid": the original with the network's low-frequency correction (module docstring);
    # "shiq": the network's output as it is.
    mode: str = "hybrid"
    # Blur of the correction, in pixels at the image's own resolution.
    detail_sigma: float = 10.0
    jpeg_quality: int = 98


def merge_shiq_detail(original: np.ndarray, shiq: np.ndarray, sigma: float) -> np.ndarray:
    """`original + blur(shiq - original, sigma)`, uint8 in, uint8 out, same shape. The
    correction is computed in float and the result clipped to [0, 255]."""
    if original.shape != shiq.shape:
        raise ValueError(f"original {original.shape} and SHIQ output {shiq.shape} differ in size")
    correction = shiq.astype(np.float32) - original.astype(np.float32)
    if sigma > 0:
        correction = cv2.GaussianBlur(correction, (0, 0), sigma)
    return np.clip(original.astype(np.float32) + correction, 0, 255).round().astype(np.uint8)


def _save(pixels: np.ndarray, dest: Path, jpeg_quality: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixels)
    if dest.suffix.lower() in (".jpg", ".jpeg"):
        image.save(dest, quality=jpeg_quality, subsampling=0)
    else:
        image.save(dest)


def run_shiq(sources: list[Path], outputs: list[Path], config: SHIQConfig) -> None:
    """The network's raw output for each of `sources`, written to the matching `outputs`
    (PNG files). One subprocess for all of them."""
    with tempfile.TemporaryDirectory(prefix="shiq_job_") as tmp:
        job = Path(tmp) / "job.json"
        job.write_text(json.dumps({
            "shiq_root": str(SHIQ_ROOT),
            "items": [{"image": str(s.resolve()), "out": str(o.resolve())} for s, o in zip(sources, outputs)],
        }))
        subprocess.run([config.python_bin, str(_WORKER), "--job", str(job)], check=True)


def remove_highlights(
    sources: list[Path], dests: list[Path], config: SHIQConfig | None = None
) -> None:
    """Highlight-free versions of `sources`, written to `dests` at the same size, in the
    format their extension names (JPEG for .jpg/.jpeg)."""
    config = config or SHIQConfig()
    if config.mode not in ("hybrid", "shiq"):
        raise ValueError(f"SHIQ mode must be 'hybrid' or 'shiq', got {config.mode!r}")
    if len(sources) != len(dests):
        raise ValueError("one destination per source")
    with tempfile.TemporaryDirectory(prefix="shiq_raw_") as tmp:
        raw = [Path(tmp) / f"{i}.png" for i in range(len(sources))]
        run_shiq(sources, raw, config)
        for src, raw_path, dest in zip(sources, raw, dests):
            shiq = np.asarray(Image.open(raw_path).convert("RGB"))
            if config.mode == "hybrid":
                original = np.asarray(Image.open(src).convert("RGB"))
                shiq = merge_shiq_detail(original, shiq, config.detail_sigma)
            _save(shiq, dest, config.jpeg_quality)

