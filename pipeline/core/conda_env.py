"""Where the separate conda environments the pipeline shells out to (fastgs, da3, shiq, mpface) live."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["env_python"]


def env_python(name: str) -> str:
    """The python of conda env `name`: `$SNAP3D_<NAME>_PYTHON` if set, else the env of that name
    next to the one running this process (conda keeps its envs side by side under `envs/`)."""
    override = os.environ.get(f"SNAP3D_{name.upper()}_PYTHON")
    return override or str(Path(sys.prefix).parent / name / "bin" / "python")
