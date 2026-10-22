"""Baking stage wrapper: high-poly mesh + uv_mesh -> baseline height_map.

The height map is a *baseline* (raw high->low displacement), handed to slf as-is; the
final `height_map.png` in the slf stage dir is this same baseline.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pipeline.core.baking import BakingConfig, run_baking
from pipeline.schemas.manifest import StageManifest, read_manifest
from pipeline.schemas.mesh import Mesh
from pipeline.schemas.uv_mesh import UVMesh
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    highpoly_dir = input_dirs["highpoly_mesh"]
    uv_dir = input_dirs["uv_mesh"]

    highpoly = Mesh.load(highpoly_dir / "mesh_highpoly.obj")
    uv_mesh = UVMesh.load(uv_dir / "mesh_uv.obj", texture_resolution=_uv_texture_resolution(uv_dir))

    height_map_baseline = run_baking(highpoly, uv_mesh, BakingConfig(**params))
    height_map_baseline.save(output_dir / "height_map_baseline.png")

    return finalize_stage(
        "baking",
        output_dir,
        params,
        input_dirs,
        outputs={"height_map_baseline": "height_map_baseline.png"},
        start_time=t0,
    )


def _uv_texture_resolution(uv_dir: Path) -> tuple[int, int]:
    manifest = read_manifest(uv_dir)
    if manifest is not None and "texture_resolution" in manifest.params:
        w, h = manifest.params["texture_resolution"]
        return (w, h)
    return (1024, 1024)


def main() -> None:
    stage_cli("baking", run)


if __name__ == "__main__":
    main()
