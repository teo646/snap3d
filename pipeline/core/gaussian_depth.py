"""Per-view depth maps rendered from a trained Fast-PGSR model, at the posed cameras'
own resolution: the reference slf's depth test compares each observation against.

Why the Gaussians rather than the fused mesh: the mesh is those same depth maps
averaged into 2 mm TSDF voxels, which erases anything thinner - suede fibres, a thin
rim's edge. In a photo those still occlude what is behind them, so a pixel showing a
fibre over the boot's lining depth-tests as "lining" against the mesh and passes. The
Gaussians keep them.

The renderer lives in the `fastgs` env, so this writes the cameras to JSON and runs
_fastgs_depth_worker.py there.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from pipeline.core.surface_recon_fastpgsr import FASTGS_ROOT, FastPGSRConfig, _run
from pipeline.schemas.posed_images import PosedImageSet

__all__ = ["find_fastpgsr_checkpoint", "find_model_dir", "render_gaussian_depths"]


def find_model_dir(recon_dir: Path) -> Path:
    """The trained-model directory inside a surface_recon stage's output.

    Its name ends in `_run` (`_fastpgsr_run` for the Fast-PGSR path); slf takes whichever is there."""
    candidates = [d for d in sorted(recon_dir.glob("*_run")) if (d / "point_cloud").is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"no trained Gaussian model (*_run/point_cloud/) under {recon_dir} - is it a "
            "Fast-PGSR surface_recon stage?"
        )
    if len(candidates) > 1:
        raise RuntimeError(f"more than one trained model under {recon_dir}: {[d.name for d in candidates]}")
    return candidates[0]


def find_fastpgsr_checkpoint(run_dir: Path) -> tuple[Path, int]:
    """(point_cloud.ply of the latest iteration, sh_degree) for a Fast-PGSR model dir."""
    iterations = sorted(
        (int(m.group(1)), d) for d in (run_dir / "point_cloud").glob("iteration_*")
        if (m := re.fullmatch(r"iteration_(\d+)", d.name)) and (d / "point_cloud.ply").exists()
    )
    if not iterations:
        raise FileNotFoundError(f"no point_cloud/iteration_*/point_cloud.ply under {run_dir}")
    cfg = (run_dir / "cfg_args").read_text()
    m = re.search(r"sh_degree=(\d+)", cfg)
    if m is None:
        raise ValueError(f"sh_degree not found in {run_dir / 'cfg_args'}")
    return iterations[-1][1] / "point_cloud.ply", int(m.group(1))


def render_gaussian_depths(
    model_dir: Path,
    posed_images: PosedImageSet,
    out_dir: Path,
    mult: float,
    python_bin: str = FastPGSRConfig.python_bin,
) -> Path:
    """Writes `<out_dir>/<image stem>.npy` (float32 camera-space z, 0 where nothing
    was rendered) for every posed image; returns `out_dir`."""
    ply, sh_degree = find_fastpgsr_checkpoint(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = []
    for img in posed_images.images:
        cam = posed_images.cameras[img.camera_id]
        fx, fy, cx, cy = cam.as_matrix()[0, 0], cam.as_matrix()[1, 1], cam.as_matrix()[0, 2], cam.as_matrix()[1, 2]
        cameras.append({
            "name": Path(img.file_name).stem, "width": cam.width, "height": cam.height,
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "R": img.pose.rotation_matrix().tolist(), "t": [img.pose.tx, img.pose.ty, img.pose.tz],
        })
    cameras_json = out_dir / "cameras.json"
    cameras_json.write_text(json.dumps(cameras))
    _run(
        [
            python_bin, str(Path(__file__).with_name("_fastgs_depth_worker.py")),
            "--fastgs-root", str(FASTGS_ROOT), "--ply", str(ply.resolve()), "--sh-degree", str(sh_degree),
            "--mult", str(mult), "--cameras", str(cameras_json.resolve()), "--out-dir", str(out_dir.resolve()),
        ],
        cwd=FASTGS_ROOT,
    )
    return out_dir
