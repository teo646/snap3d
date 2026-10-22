"""COLMAP-format dataset (images/ + sparse/) -> high-poly Mesh via Fast-PGSR
(FastGS applied to PGSR, third_party/FastGS on its `fast-pgsr` branch).

PGSR's method, accelerated. FastGS changes how densification
decides which Gaussians to keep - a loss-map threshold plus separate absolute-gradient
and gradient thresholds for splitting and cloning - and trains the two SH bands at
separate learning rates. The reconstruction it produces is still PGSR's: planar
Gaussians fused through TSDF into `mesh/tsdf_fusion_post.ply`, largest connected
component only.

That shared contract is why this wrapper is thin. Fast-PGSR takes every argument PGSR
does plus nine of its own, writes the mesh to the same path under the same name, and
applies the same `post_process_mesh(cluster_to_keep=1)`. The dataset builders and the
masked render dataset are reused from the PGSR stage unchanged.

It runs in its own `fastgs` conda env, cloned from `pgsr` so it inherits the
torch 2.11+cu128 build that works on this GPU. A separate env is not optional: both
repos ship packages called `diff-plane-rasterization`, `fused-ssim` and `simple-knn`,
with different sources behind the same names, so installing one into the other's
environment silently replaces the other's rasterizer.

Patched in the submodule (third_party/patches/FastGS.patch): the missing `<cstdint>` /
`<cfloat>` includes the PGSR submodule also needed; `quaternion_to_matrix`
vendored in place of its pytorch3d import; the alpha-mask depth zeroing in
`scripts/render_tnt.py`; `multi_view_weight_until_iter` actually bounding the multi-view
loss in train.py; and a guard in both rasterizers'
`duplicateToTilesTouched` against opacity below 1/255, which otherwise yields a NaN tile
rect and an illegal memory access shortly after a densification.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.core.conda_env import env_python

FASTGS_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "FastGS"

__all__ = ["FastPGSRConfig", "run_fastpgsr_surface_recon"]


@dataclass
class FastPGSRConfig:
    python_bin: str = env_python("fastgs")
    iterations: int = 30000
    max_depth: float = 8.0
    voxel_size: float = 0.002
    use_depth_filter: bool = True
    # The nine parameters FastGS adds to PGSR. Values are upstream's own run_tnt.py,
    # which is the only invocation its authors publish for this branch.
    ncc_scale: float = 0.5
    opacity_cull_threshold: float = 0.05
    exposure_compensation: bool = True
    densification_interval: int = 500
    highfeature_lr: float = 0.02
    grad_abs_thresh: float = 0.0004
    # Multiplier on the compact box that sets how many tiles each splat covers. Passed
    # to both train and render: the mesh pass rasterises with the same setting.
    mult: float = 0.7
    # Last iteration the multi-view loss is applied (it starts after 7000). Upstream
    # declares 10000 but never reads it; the FastGS patch wires it.
    multi_view_weight_until_iter: int = 10000
    # Weight on |rendered_alpha - mask| in train.py, wired by the FastGS patch. Only does
    # anything when the training images carry an alpha channel, i.e. under
    # surface_recon_fastpgsr. 0 is upstream's behaviour.
    lambda_mask_alpha: float = 0.0
    # Monocular depth prior, masked to the object (see the FastGS patch's
    # utils/loss_utils.depth_prior_loss). Needs `prior_dir` passed to
    # run_fastpgsr_surface_recon - the stage wires that from a geometry_prior input. 0
    # disables the term, which is upstream's behaviour.
    lambda_depth_prior: float = 0.0
    depth_prior_until_iter: int = 15000
    # Nonzero: the depth prior's weight follows how little the render can be trusted at each
    # pixel (FastGS's `depth_prior_render_conf`; 1 = alpha x normal agreement, 2 = multi-view
    # geometric consistency); 0: one flat weight, `lambda_depth_prior`.
    depth_prior_render_conf: int = 2
    # With `depth_prior_render_conf` 2: how many neighbouring views each view's depth is
    # checked against.
    depth_prior_geo_neighbors: int = 2
    # Where FastGS keeps every training view's image, gray copy and mask. "cuda" holds them
    # all in GPU memory, 5 float32 channels a pixel: about 60 MB a view at 3 MP, so ~7.4 GB
    # for 124 views (measured on statue_all) - most of a 16 GB card, before training has
    # allocated anything. "cpu" leaves them in RAM and moves one view's to the GPU per
    # iteration instead (6 ms a view, measured), freeing that memory. Use "cpu" when the
    # view count times the frame size does not leave room for training.
    data_device: str = "cuda"
    # Read by the stage wrapper: dilates the per-view masks before compositing them into
    # the render dataset's alpha, same as the PGSR path.
    mask_dilation_px: int = 1
    extra_train_args: list[str] = field(default_factory=list)


def _run(cmd: list[str], cwd: Path) -> str:
    """Streams the subprocess's output byte-chunked rather than line-buffered, since
    tqdm's `\\r` progress bars never reach a line-by-line reader until they finish.
    Accumulates it too, so a failure can report a tail."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    chunks: list[bytes] = []
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(1024)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        chunks.append(chunk)
    proc.wait()
    output = b"".join(chunks).decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` (cwd={cwd}) failed with exit code {proc.returncode}\n"
            f"--- output (tail) ---\n{output[-4000:]}"
        )
    return output


def run_fastpgsr_surface_recon(
    dataset_dir: Path,
    output_dir: Path,
    config: FastPGSRConfig | None = None,
    render_dataset_dir: Path | None = None,
    prior_dir: Path | None = None,
) -> Path:
    """`dataset_dir` must hold flat `images/` + `sparse/0/{cameras,images,points3D}.bin`.
    Note the `0/`: despite deriving from PGSR, Fast-PGSR's reader is vanilla 3DGS's and
    looks one level deeper than PGSR's flat `sparse/`.

    `render_dataset_dir`, if given, replaces `dataset_dir` for the render/TSDF step only:
    same layout but RGBA images whose alpha is the object mask, which the render pass
    uses to zero background depth before integrating. Training stays on the unmasked
    dataset, exactly as in the PGSR path.

    Returns `mesh/tsdf_fusion_post.ply`: the largest connected component of the fused
    mesh.
    """
    config = config or FastPGSRConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Absolute paths required: the subprocesses run with cwd=FASTGS_ROOT, so a relative
    # -s/-m would resolve there instead of in the caller's directory.
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    render_dataset_dir = render_dataset_dir.resolve() if render_dataset_dir else dataset_dir

    train_cmd = [
        config.python_bin, "train.py",
        "-s", str(dataset_dir),
        "-m", str(output_dir),
        "--iterations", str(config.iterations),
        "--ncc_scale", str(config.ncc_scale),
        "--opacity_cull_threshold", str(config.opacity_cull_threshold),
        "--densification_interval", str(config.densification_interval),
        "--highfeature_lr", str(config.highfeature_lr),
        "--grad_abs_thresh", str(config.grad_abs_thresh),
        "--mult", str(config.mult),
        "--multi_view_weight_until_iter", str(config.multi_view_weight_until_iter),
        "--lambda_mask_alpha", str(config.lambda_mask_alpha),
        "--lambda_depth_prior", str(config.lambda_depth_prior),
        "--depth_prior_until_iter", str(config.depth_prior_until_iter),
        "--depth_prior_render_conf", str(config.depth_prior_render_conf),
        "--depth_prior_geo_neighbors", str(config.depth_prior_geo_neighbors),
        "--data_device", config.data_device,
        *(["--prior_path", str(prior_dir.resolve())] if prior_dir else []),
        *(["--exposure_compensation"] if config.exposure_compensation else []),
        *config.extra_train_args,
    ]
    _run(train_cmd, cwd=FASTGS_ROOT)

    # scripts/render_tnt.py is what upstream's own run_tnt.py calls for this branch (the
    # top-level render.py references an undefined `mesh_path` and cannot finish). It was
    # patched with the two lines PGSR's render.py has that zero depth outside each view's
    # alpha mask before TSDF integration: Tanks and Temples has no masks, so upstream
    # never needed them, but without them our RGBA render dataset fuses every view's
    # whole background - on boots 28 GB RSS plus 12 GB of swap ten views into fusion.
    render_cmd = [
        config.python_bin, "scripts/render_tnt.py",
        "-s", str(render_dataset_dir),
        "-m", str(output_dir),
        "--max_depth", str(config.max_depth),
        "--voxel_size", str(config.voxel_size),
        "--num_cluster", "1",
        "--mult", str(config.mult),
        "--skip_test",
    ]
    if config.use_depth_filter:
        render_cmd.append("--use_depth_filter")
    _run(render_cmd, cwd=FASTGS_ROOT)

    mesh_path = output_dir / "mesh" / "tsdf_fusion_post.ply"
    if not mesh_path.exists():
        raise RuntimeError(f"Fast-PGSR did not produce the expected mesh at {mesh_path}")
    return mesh_path
