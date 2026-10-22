"""Shared plumbing for the interactive viewers: run-dir lookup + capture up axis.

The camera itself lives in pipeline/core/orbit_camera.py (the export stage needs it
too) and is re-exported here, since every viewer imports it from this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.core.orbit_camera import DEFAULT_UP, OrbitCamera, look_at, perspective  # noqa: F401
from pipeline.schemas.manifest import read_manifest
from pipeline.schemas.posed_images import PosedImageSet


def find_stage_dir(run_dir: Path, stage_id: str) -> Path:
    """The subdirectory of `run_dir` whose manifest records this stage. Not
    `run_dir / "NN_<id>"`: that prefix is just the config's stage order, so inserting a
    stage renumbers everything after it and a hardcoded name then points at the wrong
    directory - or at a stale one from a previous pipeline shape."""
    matches = [
        d for d in sorted(run_dir.iterdir())
        if d.is_dir() and (m := read_manifest(d)) is not None and m.stage == stage_id
    ]
    if not matches:
        raise FileNotFoundError(f"no '{stage_id}' stage dir under {run_dir} - run the pipeline through it first")
    if len(matches) > 1:
        raise RuntimeError(f"multiple '{stage_id}' stage dirs under {run_dir}: {matches}")
    return matches[0]


def capture_up_vector(run_dir: Path, fallback: np.ndarray = DEFAULT_UP) -> np.ndarray:
    """This run's world up: the mean of its ingest cameras' up vectors. Tolerant on
    purpose - `run_dir` may be a stage dir or a bare mesh path - and falls back to +Y
    rather than failing, since an orbit axis is a convenience, not correctness."""
    try:
        ingest_dir = find_stage_dir(Path(run_dir), "ingest")
        return PosedImageSet.load(ingest_dir).mean_up_vector(ingest_dir)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError):
        return np.asarray(fallback, dtype=np.float64)
