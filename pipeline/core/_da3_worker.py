"""Depth Anything 3 worker: runs in the `da3` conda env, not the pipeline's.

Standalone on purpose - it imports nothing from `pipeline` - because DA3's dependencies
(numpy<2, its own torch build) are not the pipeline's. `pipeline.core.geometry_prior_da3`
starts it as a subprocess with a JSON job file.

The job names every view's image, its COLMAP world-to-camera pose and its intrinsics (at
the image's own resolution). Views are predicted together in small groups of neighbouring
cameras, because DA3 is a multi-view model: given poses it makes the groups' depths agree
with each other and puts them at the scale of those poses, i.e. COLMAP's. The groups are
formed by a nearest-neighbour walk through the camera centres, so each group is a set of
views looking at the same part of the object.

Per view it writes `<out_dir>/<stem>.pth` = {"depth", "conf"}, float16, at DA3's
processed resolution - the layout FastGS's `_load_depth_prior` reads, plus `scale.json`.

Two normalisations, because the depth-prior loss is a confidence-weighted *squared*
residual and so inherits the units of both. `conf` is rescaled to [0, 1] per view
(divided by its 99th percentile, clipped): DA3's own confidence has no fixed range.
`depth` is divided by one constant for the whole capture - its median over every view -
so it is ~1 rather than in COLMAP's arbitrary scene units. One constant, not one per view:
the point of giving DA3 the poses is that depths agree across views, and a per-view
divisor would throw that away. The constant is recorded in `scale.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _groups(centers: np.ndarray, size: int) -> list[list[int]]:
    """Indices in chunks of `size`, walking from camera to nearest unvisited camera."""
    n = len(centers)
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        dist = np.linalg.norm(centers - centers[order[-1]], axis=1)
        dist[visited] = np.inf
        order.append(int(np.argmin(dist)))
        visited[order[-1]] = True
    return [order[i : i + size] for i in range(0, n, size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())

    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(job["model"]).to(device=job["device"])

    views = job["views"]
    ext = np.array([v["extrinsics"] for v in views], dtype=np.float32)  # (N, 4, 4) w2c
    ixt = np.array([v["intrinsics"] for v in views], dtype=np.float32)  # (N, 3, 3)
    centers = np.array([-(e[:3, :3].T @ e[:3, 3]) for e in ext])
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    depths: dict[str, torch.Tensor] = {}
    confs: dict[str, torch.Tensor] = {}
    for group in _groups(centers, job["views_per_group"]):
        pred = model.inference(
            [views[i]["image"] for i in group],
            extrinsics=ext[group],
            intrinsics=ixt[group],
            process_res=job["process_res"],
        )
        for k, i in enumerate(group):
            stem = views[i]["stem"]
            depths[stem] = torch.from_numpy(np.asarray(pred.depth[k], dtype=np.float32))
            conf = torch.from_numpy(np.asarray(pred.conf[k], dtype=np.float32))
            sample = conf.flatten()[:: max(1, conf.numel() // 100000)]
            confs[stem] = (conf / torch.quantile(sample, 0.99)).clamp(0, 1)
        print(f"  {len(depths)}/{len(views)} views", flush=True)
        torch.cuda.empty_cache()

    scale = float(torch.cat([d.flatten()[::97] for d in depths.values()]).median())
    for stem in depths:
        torch.save({"depth": (depths[stem] / scale).half(), "conf": confs[stem].half()}, out_dir / f"{stem}.pth")
    (out_dir / "scale.json").write_text(json.dumps({"depth_divisor": scale}))
    print(f"  depth divided by {scale:.4f} (median over all views)", flush=True)


if __name__ == "__main__":
    main()
