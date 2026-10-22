"""Surface reconstruction stage wrapper (Fast-PGSR), trained on background-masked
images instead of the full frame - the `surface_recon` of every photo pipeline (see
photo_pipeline.yaml's header comment).

The dataset `train_cmd` reads from is built from masking's masked `images/`, not
resize_dataset's unmasked ones. The final TSDF fusion uses a separate render dataset
with the mask composited in as alpha.

Why this exists: FastGS/PGSR's photometric loss (L1+SSIM), single-view normal loss, and
multi-view NCC loss are all computed over the *entire* frame. On a capture where the
object is a small fraction of each photo (measured on p2: masks cover a median 7.8% of
the frame), that means ~90% of every training signal is spent fitting a background that
usually isn't even geometrically consistent between views (a handheld orbit around a
small object crosses a cluttered room, not a fixed backdrop) - which on p2 produced one
large warped sheet (bounding box ~1.8x1.2x0.6m for a ~15cm stone) rather than the object.
Training on masking's pixel-masked images instead removes that competing signal
entirely.

That fix is not free, and is more likely to bite exactly when it is reached for (a
small object in a cluttered frame, i.e. a high boundary-to-area ratio):

  * The flat fill colour meeting real texture at the silhouette is a hard edge with no
    real-world counterpart. SSIM's windowed comparison straddles it, and the optimiser
    can cheapest-explain the discontinuity with thin "fringe" Gaussians right at the
    boundary - exactly where silhouette accuracy matters most downstream.
  * A flat-coloured region has ~zero reconstruction error at *any* depth, so Gaussians
    seeded there by SfM (which still ran on the original, unmasked photos) have little
    pressure to be culled rather than settle into a flat backdrop plane - trading one
    failure mode (a warped sheet) for a related one (a flat sheet) is possible.
  * The masking stage's own mask (used for SfM point filtering elsewhere) is tightened
    with `mask_dilation_px` - too little margin here clips real object silhouette into
    the fill colour instead of just removing it.

The second of those is what `lambda_mask_alpha` addresses, and it is why the right lever
is alpha rather than the photometric losses. A background-coloured training pixel
constrains rendered *colour* only, which a black opaque Gaussian satisfies exactly as
well as empty space does; nothing in the photometric terms distinguishes the two. Nor do
FastGS's pruning passes, because all of them (multi-view trim's `out_observe`,
`densify_and_prune_fastgs`'s importance and pruning scores) key off whether a Gaussian
was *rendered*, not whether it was penalised, and a backdrop Gaussian is rendered in
every view. Supervising accumulated alpha against the mask penalises the geometry
directly. Restricting L1/SSIM to the mask instead - the obvious-looking alternative -
makes this strictly worse, since it removes the background constraint altogether while
leaving every pruning heuristic equally blind; measured on p3, 26k Gaussians became 107k
and the highpoly mesh 1.5M -> 3.4M verts.

A tighter crop around the mask's bounding box (with adjusted intrinsics) would avoid
the hard-edge/flat-backdrop failure modes entirely, since every remaining pixel would
still be real photographic content - that is the likely next step if masked-background
training plus alpha supervision doesn't fully fix it.
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


def _depth_prior_dir(geometry_prior: Path | None, lambda_depth_prior: float) -> Path | None:
    """Where training reads the monocular depth priors from, or None when it should not.

    With `lambda_depth_prior` at 0 the term is skipped, so the `geometry_prior` input -
    present in every pipeline that can use it, and empty unless it was switched on - is
    not looked at at all. With it positive the priors must exist: a weight asking for a
    prior that is not there is refused here, where the cause is obvious, rather than
    surfacing as a training run that quietly applies nothing.
    """
    if lambda_depth_prior <= 0.0:
        return None
    if geometry_prior is None:
        raise RuntimeError(
            f"lambda_depth_prior={lambda_depth_prior} needs per-view priors - give this stage a "
            "`geometry_prior` input."
        )
    priors = geometry_prior / "priors"
    if not any(priors.glob("*.pth")):
        raise RuntimeError(
            f"lambda_depth_prior={lambda_depth_prior} but {priors} holds no priors: the geometry_prior "
            "stage ran with lambda_depth_prior 0. Give both stages the same weight (the runner's "
            "--lambda-depth-prior does)."
        )
    return priors


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_dir = input_dirs["dataset"] / "_workspace" / "dense"
    masked_images_dir = input_dirs["masking"] / "images"
    if not masked_images_dir.exists():
        raise RuntimeError(
            f"Expected masking's masked training images at {masked_images_dir} - give masking "
            "`write_masked_images: true` (see pipeline/stages/masking/run.py)."
        )
    dataset_dir = output_dir / "_fastpgsr_dataset_masked_train"
    build_dataset(masked_images_dir, dense_dir, dataset_dir)
    nest_sparse_model(dataset_dir)

    masks_dir = input_dirs["masking"] / "masks"
    if not masks_dir.exists():
        raise RuntimeError(f"Expected masking's per-view masks at {masks_dir}")
    config = FastPGSRConfig(**params)
    masked_render_dataset_dir = output_dir / "_fastpgsr_dataset_masked_render"
    build_masked_render_dataset(dense_dir, masks_dir, masked_render_dataset_dir, config.mask_dilation_px)
    nest_sparse_model(masked_render_dataset_dir)

    prior_dir = _depth_prior_dir(input_dirs.get("geometry_prior"), config.lambda_depth_prior)

    mesh_path = run_fastpgsr_surface_recon(
        dataset_dir, output_dir / "_fastpgsr_run", config,
        render_dataset_dir=masked_render_dataset_dir, prior_dir=prior_dir,
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
