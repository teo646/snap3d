"""Hole-close stage wrapper: high-poly mesh -> hole-closed high-poly mesh.

Writes `mesh_highpoly.obj`, the same name surface_recon uses; decimation's `mesh` and baking's
`highpoly_mesh` inputs must both point here (see pipeline/core/hole_close.py).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pipeline.core.hole_close import HoleCloseConfig, run_hole_close
from pipeline.schemas.manifest import StageManifest
from pipeline.schemas.mesh import Mesh
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = Mesh.load(input_dirs["mesh"] / "mesh_highpoly.obj")
    closed = run_hole_close(mesh, HoleCloseConfig(**params))
    closed.save(output_dir / "mesh_highpoly.obj")

    return finalize_stage(
        "hole_close", output_dir, params, input_dirs, outputs={"mesh": "mesh_highpoly.obj"}, start_time=t0
    )


def main() -> None:
    stage_cli("hole_close", run)


if __name__ == "__main__":
    main()
