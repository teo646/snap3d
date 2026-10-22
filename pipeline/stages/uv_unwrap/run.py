"""UV unwrap stage wrapper: low-poly mesh -> uv_mesh."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pipeline.core.uv_unwrap import UVUnwrapConfig, run_uv_unwrap
from pipeline.schemas.manifest import StageManifest
from pipeline.schemas.mesh import Mesh
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    decimation_dir = input_dirs["mesh"]

    mesh = Mesh.load(decimation_dir / "mesh_lowpoly.obj")
    params = dict(params)
    if "texture_resolution" in params:
        params["texture_resolution"] = tuple(params["texture_resolution"])
    uv_mesh = run_uv_unwrap(mesh, UVUnwrapConfig(**params))
    uv_mesh.save(output_dir / "mesh_uv.obj")

    return finalize_stage(
        "uv_unwrap", output_dir, params, input_dirs, outputs={"uv_mesh": "mesh_uv.obj"}, start_time=t0
    )


def main() -> None:
    stage_cli("uv_unwrap", run)


if __name__ == "__main__":
    main()
