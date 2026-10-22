"""UV unwrap stage wrapper: low-poly mesh -> uv_mesh.

Faces that must come out as ONE UV island (a face-detection step can write `face_faces.npy` for a
face) are given as `.npy` files of face indices into the low-poly mesh, either through an optional
`whole_charts` *input* (a directory of them) or the `whole_charts` *param* (a list of paths). The
param is what a runner config can use, since its inputs name stages; the files' content is
recorded in the manifest, because the path alone does not tell a changed file from the old one.
See pipeline/core/uv_unwrap.py for how this is done, and for why xatlas's own options cannot."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
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
    original_params = params  # the manifest records what was asked, not what this stage popped
    params = dict(params)
    if "texture_resolution" in params:
        params["texture_resolution"] = tuple(params["texture_resolution"])
    chart_files = [Path(p) for p in params.pop("whole_charts", [])]
    charts_dir = input_dirs.get("whole_charts")
    if charts_dir is not None:
        chart_files += sorted(charts_dir.glob("*.npy"))
    if charts_dir is not None and not chart_files:
        raise RuntimeError(f"`whole_charts` input {charts_dir} holds no .npy face sets")
    whole_charts = [np.load(f) for f in chart_files] or None
    uv_mesh = run_uv_unwrap(mesh, UVUnwrapConfig(**params), whole_charts=whole_charts)
    uv_mesh.save(output_dir / "mesh_uv.obj")

    return finalize_stage(
        "uv_unwrap", output_dir, original_params, input_dirs, outputs={"uv_mesh": "mesh_uv.obj"}, start_time=t0,
        extra={"whole_charts": {str(f): hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in chart_files}},
    )


def main() -> None:
    stage_cli("uv_unwrap", run)


if __name__ == "__main__":
    main()
