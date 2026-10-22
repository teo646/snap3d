"""COLMAP-format dataset (images/ + sparse/) -> high-poly Mesh via Fast-PGSR
(third_party/FastGS, `fast-pgsr` branch): planar Gaussians fused through TSDF into
`mesh/tsdf_fusion_post.ply`, largest connected component only.

Runs in its own `fastgs` conda env. Submodule changes live in third_party/patches/FastGS.patch.
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
    # The nine parameters FastGS adds to PGSR (upstream's run_tnt.py values).
    ncc_scale: float = 0.5
    opacity_cull_threshold: float = 0.05
    exposure_compensation: bool = True
    densification_interval: int = 500
    highfeature_lr: float = 0.02
    grad_abs_thresh: float = 0.0004
    # Multiplier on the compact box that sets how many tiles each splat covers (train and render).
    mult: float = 0.7
    # Last iteration the multi-view loss is applied.
    multi_view_weight_until_iter: int = 10000
    # Weight on |rendered_alpha - mask|; needs training images with an alpha channel. 0 disables.
    lambda_mask_alpha: float = 0.0
    # Where FastGS keeps every training view's image, gray copy and mask. "cuda" is faster;
    # "cpu" frees GPU memory when the view count times the frame size does not fit.
    data_device: str = "cuda"
    # Read by the stage wrapper: dilates the per-view masks before compositing them into the
    # render dataset's alpha.
    mask_dilation_px: int = 1
    extra_train_args: list[str] = field(default_factory=list)


def _run(cmd: list[str], cwd: Path) -> str:
    """Streams the subprocess's output in byte chunks (tqdm's `\\r` bars) and keeps it for failure tails."""
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
) -> Path:
    """`dataset_dir` must hold flat `images/` + `sparse/0/{cameras,images,points3D}.bin`.

    `render_dataset_dir`, if given, replaces `dataset_dir` for the render/TSDF step only: RGBA
    images whose alpha is the object mask, used to zero background depth before integrating.

    Returns `mesh/tsdf_fusion_post.ply`.
    """
    config = config or FastPGSRConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Absolute paths: the subprocesses run with cwd=FASTGS_ROOT.
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
        "--data_device", config.data_device,
        *(["--exposure_compensation"] if config.exposure_compensation else []),
        *config.extra_train_args,
    ]
    _run(train_cmd, cwd=FASTGS_ROOT)

    # scripts/render_tnt.py is patched to zero depth outside each view's alpha mask before TSDF
    # integration; without it the whole background is fused.
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
