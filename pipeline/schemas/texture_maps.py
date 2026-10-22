"""``texture_maps`` schema: baking -> height_map, slf -> height_map + sh_texture_map.

These are the final web-facing deliverables. Canonical storage is always a
lossless numpy array (`.npy` / `.npz`) so downstream fitting code never loses
precision; each type also writes a human-viewable PNG preview alongside it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_MARCH_HEADROOM = 1.15


@dataclass
class HeightMap:
    """Scalar displacement field, shape (H, W), in `units`.

    `valid[y, x] == False` marks a texel baking's rasterizer and gutter never reached -
    as opposed to `data` there, which is a zero-init default that looks like a perfectly
    ordinary height. `None` means the HeightMap predates the field, which callers should
    read as "unknown", not "everything valid". rasterize_uv_atlas takes this as its
    `target_valid` so the fit shares baking's exact gutter.
    """

    data: np.ndarray
    units: str = "scene"  # e.g. "scene" (same units as the mesh) or "normalized"
    valid: np.ndarray | None = None  # (H, W) bool

    def march_range(self) -> float:
        """Half-extent of the parallax march, in `units`: the largest displacement plus
        15% headroom, so a ray entering at the top layer starts above every surface.
        Shared because the fit and the renderer have to march the same slab - a
        different range is a different resolved uv, hence a different color."""
        return max(float(np.abs(self.data).max()) * _MARCH_HEADROOM, 1e-6)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.data.astype(np.float32))
        lo, hi = float(self.data.min()), float(self.data.max())
        span = hi - lo if hi > lo else 1.0
        preview = ((self.data - lo) / span * 65535.0).astype(np.uint16)
        cv2.imwrite(str(path), preview)
        path.with_suffix(".range.json").write_text(json.dumps({"min": lo, "max": hi, "units": self.units}))
        if self.valid is not None:
            np.save(path.with_suffix(".valid.npy"), self.valid)

    @classmethod
    def load(cls, path: str | Path) -> "HeightMap":
        path = Path(path)
        valid_path = path.with_suffix(".valid.npy")
        valid = np.load(valid_path) if valid_path.exists() else None
        npy_path = path.with_suffix(".npy")
        if npy_path.exists():
            range_path = path.with_suffix(".range.json")
            units = json.loads(range_path.read_text())["units"] if range_path.exists() else "scene"
            return cls(data=np.load(npy_path), units=units, valid=valid)
        preview = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        range_info = json.loads(path.with_suffix(".range.json").read_text())
        lo, hi = range_info["min"], range_info["max"]
        data = preview.astype(np.float32) / 65535.0 * (hi - lo) + lo
        return cls(data=data, units=range_info.get("units", "scene"), valid=valid)


@dataclass
class SHTextureMap:
    """Per-texel spherical-harmonics reflectance coefficients.

    `data` is (H, W, (degree+1)**2, 3), fit per texel over the view directions that
    observe it. `valid[y, x] == False` marks a texel with no real fit, so a caller
    needing a trustworthy coefficient must check `valid` rather than `data` being
    nonzero: slf_fit nearest-copies a neighbouring valid texel's coefficients into
    those, the same padding baking applies, so bilinear sampling near a real texel
    never blends in black.

    `up_vector` is the capture's mean camera up (PosedImageSet.mean_up_vector). It is
    not part of the fit - the frame is UV-derived and rebuilt from vertex attributes -
    but it is the only record of which way was up, and a viewer has nothing else to
    orbit about. `None` means the npz predates the field; render-time code re-derives
    it from the run's ingest stage.
    """

    data: np.ndarray
    degree: int
    valid: np.ndarray  # (H, W) bool
    up_vector: np.ndarray | None = None  # (3,) world-space, see above

    def __post_init__(self) -> None:
        expected_coeffs = (self.degree + 1) ** 2
        if self.data.shape[-2] != expected_coeffs:
            raise ValueError(
                f"SH degree {self.degree} implies {expected_coeffs} coeffs, "
                f"got data with {self.data.shape[-2]}"
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        arrays = {"data": self.data.astype(np.float32), "degree": self.degree, "valid": self.valid}
        if self.up_vector is not None:
            arrays["up_vector"] = np.asarray(self.up_vector, dtype=np.float32)
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        # DC term (L=0) is the diffuse-albedo-like average response; useful quick preview.
        dc = np.clip(self.data[..., 0, :] * 0.5 / np.sqrt(np.pi), 0, 1)
        preview = (dc * 255.0).astype(np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

    @classmethod
    def load(cls, path: str | Path) -> "SHTextureMap":
        npz_path = Path(path).with_suffix(".npz")
        loaded = np.load(npz_path)
        up = loaded["up_vector"] if "up_vector" in loaded.files else None
        return cls(data=loaded["data"], degree=int(loaded["degree"]), valid=loaded["valid"], up_vector=up)
