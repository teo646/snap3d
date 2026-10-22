"""Runs inside the `fastgs` conda env (see surface_recon_fastpgsr.py for why it is
separate): loads a trained Fast-PGSR checkpoint and writes, per camera, the
`plane_depth` its renderer produces - the same map train.py's losses use and
scripts/render_tnt.py fuses into the mesh - as float32 .npy (H, W), camera-space z.

Not imported by the pipeline: gaussian_depth.py launches it with the fastgs
interpreter and cwd set to the FastGS checkout.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastgs-root", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--sh-degree", type=int, required=True)
    parser.add_argument("--mult", type=float, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.fastgs_root))
    from argparse import ArgumentParser, Namespace

    from arguments import PipelineParams
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import focal2fov

    pipe = PipelineParams(ArgumentParser()).extract(Namespace(separate_sh=True, convert_SHs_python=False, compute_cov3D_python=False, debug=False))
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(args.ply))
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    render_args = Namespace(mult=args.mult)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cameras = json.loads(args.cameras.read_text())
    with torch.no_grad():
        for i, c in enumerate(cameras):
            W, H = c["width"], c["height"]
            fx, fy, cx, cy = c["fx"], c["fy"], c["cx"], c["cy"]
            # FastGS's Camera fixes the principal point at the image centre; anything else
            # would render a shifted depth map, silently misaligned with the photo.
            if abs(cx - 0.5 * W) > 0.5 or abs(cy - 0.5 * H) > 0.5:
                raise ValueError(f"{c['name']}: principal point ({cx}, {cy}) is not the image centre of {W}x{H}")
            R_w2c = np.asarray(c["R"], dtype=np.float64)
            cam = Camera(
                colmap_id=i, R=R_w2c.T, T=np.asarray(c["t"], dtype=np.float64),  # 3DGS stores R transposed
                FoVx=focal2fov(fx, W), FoVy=focal2fov(fy, H), image_width=W, image_height=H,
                image_path="", image_name=c["name"], uid=i, preload_img=False,
            )
            out = render(cam, gaussians, pipe, background, render_args, return_depth_normal=False)
            depth = out["plane_depth"].squeeze().float().cpu().numpy()
            if depth.shape != (H, W):
                raise RuntimeError(f"{c['name']}: rendered depth {depth.shape}, expected {(H, W)}")
            np.save(args.out_dir / f"{c['name']}.npy", depth.astype(np.float32))
            print(f"  gaussian depth [{i + 1:3d}/{len(cameras)}] {c['name']}", flush=True)


if __name__ == "__main__":
    main()
