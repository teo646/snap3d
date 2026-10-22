"""Per-view depth + confidence priors via Depth Anything 3, given COLMAP's poses.

DA3 is a multi-view model: shown several views together with their camera poses and
intrinsics, it predicts depth that is consistent between them and at the scale of those
poses. Metric3D (`geometry_prior_metric3d`) sees one frame at a time and its depth has no
relation from view to view; the loss absorbs that with a per-view scale and offset. With
poses given, DA3's views agree already, so the cross-view shape information a per-view
fit discards is, in principle, left in.

DA3's dependencies (numpy<2, its own torch) are not the pipeline's, so it runs in its own
conda env as a subprocess - see `_da3_worker.py` for what it does and writes. The model
`DA3NESTED-GIANT-LARGE-1.1` is CC BY-NC 4.0 (the giant part); `DA3-LARGE-1.1` is Apache 2.0.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from pipeline.core.conda_env import env_python
from pipeline.schemas.posed_images import PosedImageSet

__all__ = ["DA3Config", "run_da3_priors"]

_WORKER = Path(__file__).with_name("_da3_worker.py")


@dataclass
class DA3Config:
    model: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    python_bin: str = env_python("da3")
    device: str = "cuda"
    # Longest side DA3 works at. Its default; the priors are kept at this resolution and
    # FastGS interpolates them up to the render size.
    process_res: int = 504
    # Views predicted together. Each group is a set of neighbouring cameras, and groups
    # are independent of one another - their agreement comes from sharing COLMAP's poses.
    views_per_group: int = 8


def run_da3_priors(
    posed_images: PosedImageSet, images_dir: Path, out_dir: Path, config: DA3Config | None = None
) -> dict:
    """Writes `out_dir/<stem>.pth` ({depth, conf}) for every view and returns the worker's
    `scale.json`. `images_dir` is where each `file_name` resolves (the images the priors
    must line up with); intrinsics are scaled from the posed set's resolution to theirs."""
    config = config or DA3Config()
    views = []
    for img in posed_images.images:
        path = (images_dir / img.file_name).resolve()
        if not path.exists():
            raise RuntimeError(f"Missing image for {img.file_name} at {path}")
        with Image.open(path) as im:
            width, height = im.size  # the stored buffer, as COLMAP posed it
        camera = posed_images.cameras[img.camera_id]
        K = camera.as_matrix()
        K[0] *= width / camera.width
        K[1] *= height / camera.height
        views.append({
            "image": str(path),
            "stem": Path(img.file_name).stem,
            "extrinsics": img.pose.world_to_camera().tolist(),
            "intrinsics": K.tolist(),
        })
    stems = [v["stem"] for v in views]
    if len(set(stems)) != len(stems):
        raise RuntimeError("two views share a file stem; priors are stored by stem")

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="da3_job_") as tmp:
        job = Path(tmp) / "job.json"
        job.write_text(json.dumps({
            "model": config.model, "device": config.device, "process_res": config.process_res,
            "views_per_group": config.views_per_group, "out_dir": str(out_dir), "views": views,
        }))
        subprocess.run([config.python_bin, str(_WORKER), "--job", str(job)], check=True)
    return json.loads((out_dir / "scale.json").read_text())
