"""Builders that turn resize_dataset's COLMAP `dense/` dir into the flat dataset layout the
Fast-PGSR / FastGS readers expect. Readers match images by basename only, so nested names are
flattened and duplicate basenames are refused.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image

_SPARSE_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def symlink_sparse_model(dense_dir: Path, dataset_dir: Path) -> None:
    """`sparse/` is rebuilt as a real directory with the model files symlinked in individually,
    because the dataset reader writes a derived `sparse/points3D.ply` into it."""
    sparse_dir = dataset_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    for name in _SPARSE_MODEL_FILES:
        dest = sparse_dir / name
        if not dest.exists():
            dest.symlink_to((dense_dir / "sparse" / name).resolve())


def nest_sparse_model(dataset_dir: Path) -> None:
    """Move `sparse/*.bin` down into `sparse/0/`, where FastGS's reader looks."""
    sparse_dir = dataset_dir / "sparse"
    nested = sparse_dir / "0"
    nested.mkdir(parents=True, exist_ok=True)
    for name in _SPARSE_MODEL_FILES:
        flat, dest = sparse_dir / name, nested / name
        if dest.exists():
            continue
        if not flat.is_symlink() and not flat.exists():
            raise RuntimeError(f"Expected {flat} from the shared dataset builder")
        dest.symlink_to(flat.resolve())
        flat.unlink()


def _link_flat_images(image_source_dir: Path, images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for src in sorted(image_source_dir.rglob("*")):
        if not src.is_file():
            continue
        if src.name in seen:
            raise RuntimeError(f"Duplicate image basename would collide in the flat images/: {src.name}")
        seen.add(src.name)
        dest = images_dir / src.name
        if not dest.exists():
            dest.symlink_to(src.resolve())


def build_dataset(image_source_dir: Path, sparse_dense_dir: Path, dataset_dir: Path) -> None:
    """Flat `images/` (symlinks to `image_source_dir`'s files) + `sparse/` from `sparse_dense_dir`.
    They differ when training on masking's masked images; files are matched by basename."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _link_flat_images(image_source_dir, dataset_dir / "images")
    symlink_sparse_model(sparse_dense_dir, dataset_dir)


def build_masked_render_dataset(
    dense_dir: Path, masks_dir: Path, dataset_dir: Path, mask_dilation_px: int
) -> None:
    """Same layout as `build_dataset`, but each image carries its (dilated) mask as an alpha
    channel, which the reader populates `Camera.mask` from; TSDF fusion uses it to zero background depth.

    `masks_dir` is masking's per-view output, `<image relative path>.png`; masks are resized
    (nearest) to the image. The file is PNG-encoded but keeps the original extension.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(exist_ok=True)
    dense_images_dir = dense_dir / "images"

    kernel = None
    if mask_dilation_px > 0:
        size = 2 * mask_dilation_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    seen: set[str] = set()
    for src in sorted(dense_images_dir.rglob("*")):
        if not src.is_file():
            continue
        if src.name in seen:
            raise RuntimeError(f"Duplicate image basename would collide in the flat images/: {src.name}")
        seen.add(src.name)
        dest = images_dir / src.name
        if dest.exists():
            continue

        rel = src.relative_to(dense_images_dir)
        mask_path = masks_dir / f"{rel.as_posix()}.png"
        if not mask_path.exists():
            raise RuntimeError(f"Missing mask for {rel} at {mask_path}")

        rgb = Image.open(src).convert("RGB")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask.shape[::-1] != rgb.size:
            mask = cv2.resize(mask, rgb.size, interpolation=cv2.INTER_NEAREST)
        if kernel is not None:
            mask = cv2.dilate(mask, kernel)
        alpha = Image.fromarray(mask, mode="L")
        Image.merge("RGBA", (*rgb.split(), alpha)).save(dest, format="PNG")

    symlink_sparse_model(dense_dir, dataset_dir)
