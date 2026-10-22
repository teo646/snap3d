"""Surface reconstruction stage wrapper (Fast-PGSR), trained on background-masked images.

The training dataset is built from masking's masked `images/`; the final TSDF fusion uses a
separate render dataset with the mask composited in as alpha. `lambda_mask_alpha` supervises the
rendered alpha against that mask.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pipeline.core.colmap_dataset import build_dataset, build_masked_render_dataset, nest_sparse_model
from pipeline.core.surface_recon_fastpgsr import FastPGSRConfig, run_fastpgsr_surface_recon
from pipeline.schemas.manifest import StageManifest
from pipeline.schemas.mesh import Mesh
from pipeline.stages._common import finalize_stage, stage_cli


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_dir = input_dirs["poses"] / "_workspace" / "dense"
    masked_images_dir = input_dirs["masked_images"] / "images"
    if not masked_images_dir.exists():
        raise RuntimeError(
            f"Expected masking's masked training images at {masked_images_dir} - give masking "
            "`write_masked_images: true` (see pipeline/stages/masking/run.py)."
        )
    dataset_dir = output_dir / "_fastpgsr_dataset_masked_train"
    build_dataset(masked_images_dir, dense_dir, dataset_dir)
    nest_sparse_model(dataset_dir)

    masks_dir = input_dirs["masks"] / "masks"
    if not masks_dir.exists():
        raise RuntimeError(f"Expected masking's per-view masks at {masks_dir}")
    config = FastPGSRConfig(**params)
    masked_render_dataset_dir = output_dir / "_fastpgsr_dataset_masked_render"
    build_masked_render_dataset(dense_dir, masks_dir, masked_render_dataset_dir)
    nest_sparse_model(masked_render_dataset_dir)

    mesh_path = run_fastpgsr_surface_recon(
        dataset_dir, output_dir / "_fastpgsr_run", config,
        render_dataset_dir=masked_render_dataset_dir,
    )
    Mesh.load(mesh_path).save(output_dir / "mesh_highpoly.obj")

    return finalize_stage(
        "surface_recon_fastpgsr", output_dir, params, input_dirs,
        outputs={"mesh": "mesh_highpoly.obj"}, start_time=t0,
    )


def main() -> None:
    stage_cli("surface_recon_fastpgsr", run)


if __name__ == "__main__":
    main()
