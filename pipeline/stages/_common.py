"""Shared plumbing for every stage wrapper: input hashing + manifest writing.

A stage wrapper reads directories satisfying a `schemas/` contract, calls the matching
`core/` algorithm, writes the result out, and records a manifest; this factors out the
last two parts. Caching (skipping a stage whose manifest matches) is the orchestrator's
job, not a wrapper's - invoked directly, a stage always runs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from pipeline.schemas.manifest import StageManifest, hash_tree, write_manifest

StageRunFn = Callable[[dict[str, Path], Path, dict[str, Any]], StageManifest]


def compute_input_hashes(input_dirs: dict[str, Path]) -> dict[str, str]:
    return {name: hash_tree(path) for name, path in input_dirs.items()}


def finalize_stage(
    stage: str,
    output_dir: Path,
    params: dict[str, Any],
    input_dirs: dict[str, Path],
    outputs: dict[str, str],
    start_time: float,
    extra: dict[str, Any] | None = None,
) -> StageManifest:
    manifest = StageManifest(
        stage=stage,
        params=params,
        input_hashes=compute_input_hashes(input_dirs),
        outputs=outputs,
        duration_seconds=time.time() - start_time,
        extra=extra or {},
    )
    write_manifest(output_dir, manifest)
    return manifest


def stage_cli(stage_module_name: str, run_fn: StageRunFn) -> None:
    """Standard `python -m pipeline.stages.<stage>.run` CLI: parse repeated
    `--input name=path` flags, `--output <dir>`, and an optional `--params <json>`,
    then call `run_fn`. Lets any stage be run/debugged in isolation from the
    orchestrator."""
    parser = argparse.ArgumentParser(description=f"Run the {stage_module_name} stage")
    parser.add_argument(
        "--input", action="append", default=[], metavar="NAME=PATH", help="repeatable: logical_name=path"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--params", default="{}", help="JSON dict of stage params")
    args = parser.parse_args()

    input_dirs = {}
    for entry in args.input:
        name, _, path = entry.partition("=")
        input_dirs[name] = Path(path)
    params = json.loads(args.params)

    manifest = run_fn(input_dirs, args.output, params)
    print(json.dumps(manifest.model_dump(), indent=2))
