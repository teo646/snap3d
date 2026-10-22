"""SLF range-bound stage: re-fits slf's SH texture map so no texel's function leaves the colours
it was observed with (see pipeline/core/slf_bound.py). Republishes slf's layout so export reads
it as it would read slf; a separate stage so `range_tolerance` can be retuned without re-running
the fit.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pipeline.core.slf_bound import SLFBoundConfig, bound_sh_texture_map
from pipeline.schemas.manifest import StageManifest, read_manifest
from pipeline.schemas.texture_maps import HeightMap, SHTextureMap
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    slf_dir = input_dirs["slf"]
    stats_path = slf_dir / "fit_stats.npz"
    if not stats_path.exists():
        raise RuntimeError(
            f"{stats_path} is missing - it is written by slf since slf_bound was added; re-run slf"
        )

    config = SLFBoundConfig(**params)
    bounded, report = bound_sh_texture_map(SHTextureMap.load(slf_dir / "sh_texture_map.npz"), stats_path, config)
    print(f"  {report}")
    bounded.save(output_dir / "sh_texture_map.png")
    # Through the schema, not a file copy: the png carries a sidecar.
    HeightMap.load(slf_dir / "height_map.png").save(output_dir / "height_map.png")

    # Carry slf's num_steps forward; export reads it from this stage's manifest.
    slf_manifest = read_manifest(slf_dir)
    extra = {"report": report}
    if slf_manifest is not None and "num_steps" in slf_manifest.params:
        extra["num_steps"] = slf_manifest.params["num_steps"]
    return finalize_stage(
        "slf_bound", output_dir, params, input_dirs,
        outputs={"height_map": "height_map.png", "sh_texture_map": "sh_texture_map.png"},
        start_time=t0, extra=extra,
    )


def main() -> None:
    stage_cli("slf_bound", run)


if __name__ == "__main__":
    main()
