"""Geometry-prior stage: posed images -> per-view metric depth + surface normal.

Consumed by surface_recon's depth prior term (third_party/FastGS, `lambda_depth_prior`),
which reads each view's depth and confidence. The normals come out of the same network pass
and are stored with them, but nothing reads them. The stage does nothing unless
`lambda_depth_prior` is positive: at 0, the default, it loads no model and writes an empty
`priors/`.

Two backends (`backend`): `metric3d` (default; per-view monocular depth + normal) and
`depth_anything_3` (depth + confidence predicted jointly from COLMAP's poses, in its own
conda env - pipeline/core/geometry_prior_da3.py). Both write the same `priors/<stem>.pth`.

It is a stage of its own rather than a step inside surface_recon so that iterating on the
reconstruction's parameters does not re-run inference over every view, the same reason
`masking` is separate.

Runs against `resize_dataset`'s pixels, not ingest's, so the priors are in the exact
frame and resolution FastGS trains on - the same coupling the masking stage has, and for
the same reason.

Alongside the `.pth` the consumers read, writes `depth_preview/` and `normal_preview/`
as PNGs. Those are for looking at, not for consuming: the depth is colour-mapped and
normalised per view, so it is the only output here whose pixel values mean nothing
numerically. They exist because a prior that is subtly wrong - rotated, inside out,
predicting the backdrop instead of the object - is obvious in a glance and invisible in
a tensor.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pipeline.core.geometry_prior_metric3d import (
    Metric3DConfig,
    load_metric3d,
    predict_depth_normal,
)
from pipeline.schemas.manifest import StageManifest
from pipeline.schemas.posed_images import PosedImageSet
from pipeline.stages._common import finalize_stage, stage_cli


def _camera_focal(posed_images: PosedImageSet, camera_id: int, width: int, height: int) -> tuple[float, float]:
    """(fx, fy) for an image of `width` x `height`.

    ingest's intrinsics describe its own resolution, which resize_dataset may have
    scaled away from (`max_image_size`). The undistorter preserves the field of view, so
    the focal lengths scale with the image - the same reasoning as
    resize_dataset's `_sync_camera_to_image_size`.
    """
    cam = posed_images.cameras[camera_id]
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fx = fy = cam.params[0]
    else:
        fx, fy = cam.params[0], cam.params[1]
    return fx * (width / cam.width), fy * (height / cam.height)


def _normal_faces_camera(normals: list[torch.Tensor]) -> tuple[bool, float]:
    """Do these camera-space normals point back at the camera?

    A surface you can see faces you, and this project's cameras look down +Z (COLMAP's
    convention, which the renderer inherits), so a visible surface's normal must have
    negative Z. Metric3D's sign convention is not documented, and a flipped prior would
    not fail loudly - it would quietly pull the geometry inside out for the first
    `vggt_until_iter` iterations. So it is measured rather than assumed.
    """
    z = torch.cat([n[2].reshape(-1) for n in normals])
    mean_z = float(z.mean())
    return mean_z < 0, mean_z



def _write_depth_preview(depth_tensor: torch.Tensor, depth_png: Path) -> None:
    """Colour-mapped PNG of one view's depth, normalised over its own finite positive
    pixels (near blue, far red *within this view*)."""
    import cv2

    depth = depth_tensor.float().squeeze().cpu().numpy()
    valid = np.isfinite(depth) & (depth > 0)
    norm = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        lo, hi = float(depth[valid].min()), float(depth[valid].max())
        norm[valid] = (depth[valid] - lo) / max(hi - lo, 1e-9)
    coloured = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    coloured[~valid] = 0
    cv2.imwrite(str(depth_png), coloured)


def _write_previews(pred: dict[str, torch.Tensor], depth_png: Path, normal_png: Path) -> None:
    """Human-readable PNGs of one view's prior. Diagnostic only - nothing reads these.

    Depth is normalised over its own finite positive pixels and colour-mapped, so near is
    blue and far is red *within this view*. It deliberately says nothing across views:
    Metric3D is monocular and its scale is not consistent between them, which is why the
    loss that consumes the real values solves for a per-view scale and offset first.
    Normals are the usual (n + 1) / 2 encoding, in the camera frame the network predicts.
    """
    import cv2

    _write_depth_preview(pred["depth"], depth_png)

    import cv2

    normal = pred["normal"].float().cpu().numpy()
    rgb = ((np.transpose(normal, (1, 2, 0)) + 1.0) * 0.5).clip(0.0, 1.0)
    cv2.imwrite(str(normal_png), (rgb[:, :, ::-1] * 255).astype(np.uint8))


def _run_da3(
    input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any], stage_params: dict[str, Any], t0: float
) -> StageManifest:
    """Depth Anything 3 backend: depth + confidence only (no normals - nothing consumes
    them), predicted jointly from COLMAP's poses. See pipeline/core/geometry_prior_da3.py."""
    from pipeline.core.geometry_prior_da3 import DA3Config, run_da3_priors

    posed_images = PosedImageSet.load(input_dirs["posed_images"])
    dense_images = input_dirs["images"] / "_workspace" / "dense" / "images"
    priors_dir = output_dir / "priors"
    scale = run_da3_priors(posed_images, dense_images, priors_dir, DA3Config(**stage_params))
    for path in sorted(priors_dir.glob("*.pth")):
        _write_depth_preview(torch.load(path, weights_only=True)["depth"], output_dir / "depth_preview" / f"{path.stem}.png")
    n = len(list(priors_dir.glob("*.pth")))
    print(f"  {n} priors from Depth Anything 3 (depth+conf, float16)")
    return finalize_stage(
        "geometry_prior", output_dir, params, input_dirs,
        outputs={"priors": "priors", "depth_preview": "depth_preview"},
        start_time=t0, extra={"enabled": True, "backend": "depth_anything_3", "views": n, **scale},
    )


def run(input_dirs: dict[str, Path], output_dir: Path, params: dict[str, Any]) -> StageManifest:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    priors_dir = output_dir / "priors"
    priors_dir.mkdir(exist_ok=True)

    # The stage is in every pipeline that can use a depth prior, and does nothing unless
    # one was asked for: `lambda_depth_prior` is the run's weight for the term (the
    # runner's --lambda-depth-prior), so 0 - the default - means no inference at all, and
    # an empty `priors/` is what `surface_recon` is handed and ignores.
    stage_params = dict(params)
    lambda_depth_prior = float(stage_params.pop("lambda_depth_prior", 0.0))
    if lambda_depth_prior <= 0.0:
        print("geometry_prior: lambda_depth_prior is 0, no priors computed")
        return finalize_stage(
            "geometry_prior", output_dir, params, input_dirs,
            outputs={"priors": "priors"}, start_time=t0, extra={"enabled": False},
        )

    backend = stage_params.pop("backend", "metric3d")
    depth_preview_dir = output_dir / "depth_preview"
    depth_preview_dir.mkdir(exist_ok=True)
    if backend == "depth_anything_3":
        return _run_da3(input_dirs, output_dir, params, stage_params, t0)
    if backend != "metric3d":
        raise ValueError(f"Unknown geometry_prior backend {backend!r} (expected 'metric3d' or 'depth_anything_3')")
    normal_preview_dir = output_dir / "normal_preview"
    normal_preview_dir.mkdir(exist_ok=True)

    ingest_dir = input_dirs["posed_images"]
    dense_images = input_dirs["images"] / "_workspace" / "dense" / "images"
    posed_images = PosedImageSet.load(ingest_dir)
    config = Metric3DConfig(**stage_params)

    model = load_metric3d(config)
    sample_normals: list[torch.Tensor] = []
    written = 0
    for img in posed_images.images:
        # file_name, not its basename: the undistorter keeps a source's folder structure, so a
        # capture split into subfolders has photos at dense/images/1/IMG_4141.JPG. For a flat capture the
        # two are the same.
        image_path = dense_images / img.file_name
        if not image_path.exists():
            raise RuntimeError(f"Missing undistorted image for {img.file_name} at {image_path}")
        import cv2

        height, width = cv2.imread(str(image_path), cv2.IMREAD_COLOR).shape[:2]
        fx, fy = _camera_focal(posed_images, img.camera_id, width, height)
        pred = predict_depth_normal(model, image_path, fx, fy, config)

        if len(sample_normals) < 16:
            sample_normals.append(pred["normal"].cpu())
        # float16 halves 1.8 GB of priors to 0.9 with no meaningful loss: depth is
        # realigned per view before use and normals are unit vectors.
        stem = Path(img.file_name).stem
        torch.save(
            {k: v.half().cpu() for k, v in pred.items()},
            priors_dir / f"{stem}.pth",
        )
        _write_previews(pred, depth_preview_dir / f"{stem}.png", normal_preview_dir / f"{stem}.png")
        written += 1

    faces_camera, mean_z = _normal_faces_camera(sample_normals)
    print(
        f"  {written} priors at {tuple(sample_normals[0].shape[-2:])} "
        f"(depth+conf+normal, float16); normal mean camera-space z {mean_z:+.3f} "
        f"-> {'faces the camera' if faces_camera else 'FACES AWAY'}"
    )
    if not faces_camera:
        raise RuntimeError(
            f"Metric3D's normals average z={mean_z:+.3f} in camera space, i.e. they point "
            "away from the cameras that saw them. Consuming them unflipped would supervise "
            "the reconstruction towards inside-out geometry. Flip the sign in "
            "geometry_prior_metric3d.predict_depth_normal before using these."
        )

    return finalize_stage(
        "geometry_prior", output_dir, params, input_dirs,
        outputs={
            "priors": "priors",
            "depth_preview": "depth_preview",
            "normal_preview": "normal_preview",
        },
        start_time=t0,
        extra={"enabled": True, "views": written, "normal_mean_camera_z": round(mean_z, 6)},
    )


def main() -> None:
    stage_cli("geometry_prior", run)


if __name__ == "__main__":
    main()
