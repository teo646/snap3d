"""Decimation stage wrapper: hole-closed high-poly mesh -> low-poly mesh.

Component filtering happens inside `run_decimation` (largest connected component only,
matching PGSR's post_process_mesh), so there is no separate cleanup stage in front of
this one.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pipeline.core.decimation import DecimationConfig, run_decimation
from pipeline.schemas.manifest import StageManifest
from pipeline.schemas.mesh import Mesh
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_recon_dir = input_dirs["mesh"]

    mesh = Mesh.load(surface_recon_dir / "mesh_highpoly.obj")
    lowpoly = run_decimation(mesh, DecimationConfig(**params))
    lowpoly.save(output_dir / "mesh_lowpoly.obj")

    return finalize_stage(
        "decimation", output_dir, params, input_dirs, outputs={"mesh": "mesh_lowpoly.obj"}, start_time=t0
    )


def main() -> None:
    stage_cli("decimation", run)


if __name__ == "__main__":
    main()
