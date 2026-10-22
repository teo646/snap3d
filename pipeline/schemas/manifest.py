"""Shared manifest format written by every stage wrapper.

Every ``data/runs/<run_id>/NN_<stage>/`` directory contains a
``manifest.json`` following :class:`StageManifest`. It records enough
information (input hashes + params) for the orchestrator to decide whether
a stage can be skipped on re-run (simple content-addressed caching).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MANIFEST_FILENAME = "manifest.json"


class StageManifest(BaseModel):
    stage: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    params: dict[str, Any] = Field(default_factory=dict)
    input_hashes: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    upstream_manifests: dict[str, str] = Field(default_factory=dict)
    duration_seconds: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def matches_inputs(self, params: dict[str, Any], input_hashes: dict[str, str]) -> bool:
        """Used by the orchestrator to decide whether a cached stage output is reusable."""
        return self.params == params and self.input_hashes == input_hashes


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(path: Path) -> str:
    """Hash of a file or directory. Directories are hashed as sha256 over the sorted
    list of (relative path, file hash) - i.e. this also picks up added/removed/renamed
    files, not just content changes."""
    if path.is_file():
        return hash_file(path)
    h = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file_path.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(hash_file(file_path).encode("utf-8"))
    return h.hexdigest()


def write_manifest(stage_dir: Path, manifest: StageManifest) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = stage_dir / MANIFEST_FILENAME
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest_path


def read_manifest(stage_dir: Path) -> StageManifest | None:
    manifest_path = stage_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    return StageManifest.model_validate(json.loads(manifest_path.read_text()))
