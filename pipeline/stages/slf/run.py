"""SLF stage wrapper: posed_images + uv_mesh + baseline height_map -> height_map + sh_texture_map.

The optional `masking` input restricts every photo to the object's own silhouette, so
that a frame where something crossed in front of the subject cannot fit the intruder's
colour into the texels behind it. It is optional because a capture that needs no segmenting
skips the masking stage entirely.

Calls `core.slf_fit.run_slf_fit`, which ports the CG + AtA/Atb precision-weighted
SH fit from surface-light-fields-for-planar-paintings@53f5787 (see that module's
docstring for how it's adapted from a single plane to an arbitrary UV-mapped mesh).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from pipeline.core.gaussian_depth import find_model_dir, render_gaussian_depths
from pipeline.core.slf_fit import SLFFitConfig, run_slf_fit
from pipeline.schemas.manifest import StageManifest, read_manifest
from pipeline.schemas.mesh import Mesh
from pipeline.schemas.posed_images import PosedImageSet
from pipeline.schemas.texture_maps import HeightMap
from pipeline.schemas.uv_mesh import UVMesh
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    ingest_dir = input_dirs["posed_images"]
    uv_dir = input_dirs["uv_mesh"]
    baking_dir = input_dirs["height_map"]

    posed_images = PosedImageSet.load(ingest_dir)
    uv_mesh = UVMesh.load(uv_dir / "mesh_uv.obj", texture_resolution=_uv_texture_resolution(uv_dir))
    height_map_baseline = HeightMap.load(baking_dir / "height_map_baseline.png")

    config = SLFFitConfig(**params)
    masks_dir = input_dirs["masking"] / "masks" if "masking" in input_dirs else None
    reference_mesh = None
    reference_depth_dir = None
    if config.depth_test_tolerance > 0 and config.depth_reference == "gaussians":
        if "gaussians" not in input_dirs:
            raise RuntimeError(
                "slf's depth_reference='gaussians' renders depth from the trained Fast-PGSR model; "
                "give the stage a `gaussians` input (the surface_recon stage)."
            )
        recon_dir = input_dirs["gaussians"]
        recon_manifest = read_manifest(recon_dir)
        mult = (recon_manifest.params if recon_manifest else {}).get("mult")
        if mult is None:
            raise RuntimeError(f"no `mult` in {recon_dir}'s manifest params - is it a Fast-PGSR surface_recon stage?")
        # Rendered fresh into the stage dir and deleted after the fit: 12 MP float32 per
        # view is gigabytes, and anything left here is re-hashed by every downstream stage.
        reference_depth_dir = render_gaussian_depths(
            find_model_dir(recon_dir), posed_images, output_dir / "_gaussian_depth", mult=float(mult)
        )
    elif config.depth_test_tolerance > 0:
        if "highpoly_mesh" not in input_dirs:
            raise RuntimeError(
                "slf's depth_test_tolerance > 0 tests each pixel against the high-poly mesh "
                "height_map was baked from; give the stage a `highpoly_mesh` input (the same "
                "stage baking's `highpoly_mesh` comes from)."
            )
        reference_mesh = Mesh.load(input_dirs["highpoly_mesh"] / "mesh_highpoly.obj")

    try:
        sh_texture_map = run_slf_fit(
            posed_images, ingest_dir, uv_mesh, height_map_baseline, config,
            reference_mesh=reference_mesh, reference_depth_dir=reference_depth_dir, masks_dir=masks_dir,
            fit_stats_path=output_dir / "fit_stats.npz",
            debug_status_path=output_dir / "texel_status.png",
        )
    finally:
        if reference_depth_dir is not None:
            shutil.rmtree(reference_depth_dir, ignore_errors=True)

    height_map_baseline.save(output_dir / "height_map.png")
    sh_texture_map.save(output_dir / "sh_texture_map.png")

    return finalize_stage(
        "slf",
        output_dir,
        params,
        input_dirs,
        outputs={"height_map": "height_map.png", "sh_texture_map": "sh_texture_map.png", "fit_stats": "fit_stats.npz", "texel_status": "texel_status.png"},
        start_time=t0,
    )


def _uv_texture_resolution(uv_dir: Path) -> tuple[int, int]:
    manifest = read_manifest(uv_dir)
    if manifest is not None and "texture_resolution" in manifest.params:
        w, h = manifest.params["texture_resolution"]
        return (w, h)
    return (1024, 1024)


def main() -> None:
    stage_cli("slf", run)


if __name__ == "__main__":
    main()
