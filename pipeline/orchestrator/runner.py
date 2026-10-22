"""Runs a full pipeline from a YAML config: `python -m pipeline.orchestrator.runner \
    --config pipeline/orchestrator/configs/photo_pipeline.yaml --input <photo_dir>`.

Each stage entry in the YAML's `stages` list:
    id: <str>                      # also used for the "NN_<id>" run directory name
    module: <dotted.module.path>   # must expose run(input_dirs, output_dir, params) -> StageManifest
    uses_source_input: <bool>      # if true, gets the CLI --input path as input_dirs["source"]
    inputs: {name: stage_id, ...}  # name -> that earlier stage's output dir
    params: {...}                  # passed through to the stage's core Config

`--text` is the SAM3 prompt, used as the `text_prompt` of every masking stage that sets none. A
masking stage with neither is skipped.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from pipeline.schemas.manifest import read_manifest
from pipeline.stages._common import compute_input_hashes

# Stages that get the CLI --text prompt unless they set their own.
_MASKING_MODULES = ("pipeline.stages.masking.run",)


def run_pipeline(
    config_path: Path,
    source_input: Path,
    runs_dir: Path,
    run_id: str | None,
    force: bool,
    until_stage: str | None,
    text_prompt: str | None = None,
) -> Path:
    start_time = time.time()
    config = yaml.safe_load(config_path.read_text())
    stages = config["stages"]

    run_id = run_id or f"{config.get('run_id_prefix', 'run')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    stage_dirs: dict[str, Path] = {}
    skipped_masking: set[str] = set()
    for i, stage_cfg in enumerate(stages):
        stage_id = stage_cfg["id"]
        stage_dir = run_dir / f"{i + 1:02d}_{stage_id}"
        params = stage_cfg.get("params", {})
        if stage_cfg["module"] in _MASKING_MODULES:
            # No prompt means the capture needs no segmenting; skip the stage.
            prompt = params.get("text_prompt") or text_prompt
            if not prompt:
                print(f"[{i + 1:02d}/{len(stages)}] {stage_id}: skipped (no --text prompt given)")
                skipped_masking.add(stage_id)
                continue
            params = {**params, "text_prompt": prompt}

        input_dirs: dict[str, Path] = {}
        if stage_cfg.get("uses_source_input"):
            input_dirs["source"] = source_input
        for name, ref_stage_id in stage_cfg.get("inputs", {}).items():
            if ref_stage_id not in stage_dirs:
                raise ValueError(
                    f"Stage '{stage_id}' references input from '{ref_stage_id}', which hasn't run yet "
                    f"(check stage ordering in {config_path}"
                    + (", or was skipped - see the masking note above)" if ref_stage_id in skipped_masking else ")")
                )
            input_dirs[name] = stage_dirs[ref_stage_id]

        if not force and _is_cached(stage_dir, params, input_dirs):
            print(f"[{i + 1:02d}/{len(stages)}] {stage_id}: cached, skipping")
        else:
            print(f"[{i + 1:02d}/{len(stages)}] {stage_id}: running ({stage_cfg['module']})...")
            module = importlib.import_module(stage_cfg["module"])
            t0 = time.time()
            module.run(input_dirs=input_dirs, output_dir=stage_dir, params=params)
            print(f"[{i + 1:02d}/{len(stages)}] {stage_id}: done in {time.time() - t0:.1f}s")

        stage_dirs[stage_id] = stage_dir
        if until_stage == stage_id:
            print(f"Stopping after '{stage_id}' (--until).")
            break

    print(f"total pipeline: done in {time.time() - start_time:.1f}s")
    return run_dir


def _is_cached(stage_dir: Path, params: dict, input_dirs: dict[str, Path]) -> bool:
    manifest = read_manifest(stage_dir)
    if manifest is None:
        return False
    return manifest.matches_inputs(params, compute_input_hashes(input_dirs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="the run's source: a photo or scene directory")
    parser.add_argument("--runs-dir", default=Path("data/runs"), type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--force", action="store_true", help="ignore cached manifests, rerun every stage")
    parser.add_argument("--until", default=None, metavar="STAGE_ID", help="stop after this stage id")
    parser.add_argument(
        "--text", default=None, metavar="PROMPT",
        help="object to segment, e.g. --text apple (the text_prompt of every masking stage that sets none). "
        "Omit it to skip masking entirely - for a capture with no background to strip.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input path does not exist: {args.input}", file=sys.stderr)
        raise SystemExit(1)
    run_pipeline(
        config_path=args.config,
        source_input=args.input,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        force=args.force,
        until_stage=args.until,
        text_prompt=args.text,
    )


if __name__ == "__main__":
    main()
