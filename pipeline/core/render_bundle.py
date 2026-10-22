"""Packages a finished run into one self-contained `<run_id>_<storage>.snap3d` folder.

    mesh.glb      POSITION (V,3) f32, TEXCOORD_0 (V,2) f32, indices (F,3) u32
    sh.ktx2       2D array, one layer per SH coefficient, RGBA (a = 0 padding). RGBA16F
                  coefficients for "float16"; RGBA8 UNORM for "unorm8", where
                  coefficient = sh.offset[k][c] + sh.scale[k][c] * sample
    height.ktx2   RG16F; r = displacement in `height.units`, g = 1 inside the atlas coverage
    config.json   up_vector, sh{...}, texture_resolution, height{...}, initial_camera

Conventions a consumer must follow:
  * glTF/KTX2 conventions: TEXCOORD_0 has v flipped relative to UVMesh, textures are top row first.
  * Per-vertex normals and tangents are not shipped. Normals are angle-weighted
    (`uv_geometry.vertex_normals_for`), tangents are Lengyel's, summed over incident faces and
    not normalized (`compute_vertex_tangents`).
  * `height.range` and `height.num_steps` define the parallax slab and must match.
  * `height.ktx2` green is the atlas coverage (chart interiors plus gutter), not the fit's
    validity mask.

"unorm8" quantizes each (coefficient, channel) over its min..max within the coverage. "float16"
zeroes small high-order coefficients (below `sh_zero_threshold_step * k`) and rounds the low
mantissa bits (`sh_mantissa_bits`) so the payload compresses better.

The initial camera is the mean direction of the capture's cameras from their centroid, shipped as
orbit parameters and as a resolved eye/target/view matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.core.gltf import write_glb
from pipeline.core.ktx2 import write_ktx2
from pipeline.core.orbit_camera import DEFAULT_UP, OrbitCamera, orbit_angles
from pipeline.schemas.texture_maps import HeightMap, SHTextureMap
from pipeline.schemas.uv_mesh import UVMesh

__all__ = [
    "RenderBundleConfig",
    "write_render_bundle",
    "sparsify_sh",
    "quantize_sh_unorm8",
    "round_float16_mantissa",
    "BUNDLE_FILES",
    "BUNDLE_SUFFIX",
    "SH_STORAGES",
    "bundle_dir_name",
]

# Marks a directory as a finished bundle.
BUNDLE_SUFFIX = ".snap3d"

BUNDLE_FILES = {
    "mesh": "mesh.glb",
    "sh": "sh.ktx2",
    "height": "height.ktx2",
    "config": "config.json",
}


# One bundle is written per storage.
SH_STORAGES = ("unorm8", "float16")


def bundle_dir_name(run_name: str, sh_storage: str) -> str:
    """`<run_id>_<storage>.snap3d`; `run_name` is the run directory's name."""
    if sh_storage not in SH_STORAGES:
        raise ValueError(f"sh_storage must be one of {SH_STORAGES}, got {sh_storage!r}")
    return f"{run_name}_{sh_storage}{BUNDLE_SUFFIX}"


@dataclass
class RenderBundleConfig:
    # POM layers; must match the slf fit's `num_steps`.
    num_steps: int = 7
    # Coefficient k is zeroed below `sh_zero_threshold_step * k`; 0 disables.
    sh_zero_threshold_step: float = 0.02
    # Mantissa bits kept of float16's 10; 10 disables the rounding.
    sh_mantissa_bits: int = 5
    # zlib level for the KTX2 payloads.
    compression_level: int = 9
    # Fallback view when no camera centers are given.
    fov_deg: float = 45.0
    azimuth_deg: float = 45.0
    elevation_deg: float = 20.0
    radius_scale: float = 1.2  # bounding-box diagonal -> orbit radius
    near_scale: float = 0.02  # near/far as multiples of the radius, as the viewer does
    far_scale: float = 20.0


def sparsify_sh(data: np.ndarray, threshold_step: float) -> tuple[np.ndarray, int, int]:
    """Zeroes coefficient k wherever |value| < threshold_step * k, k >= 1.

    Returns (data, values zeroed, non-zero higher-order values before the cut)."""
    data = np.array(data, dtype=np.float32, copy=True)
    if threshold_step <= 0.0:
        return data, 0, int(np.count_nonzero(data[..., 1:, :]))
    before = int(np.count_nonzero(data[..., 1:, :]))
    zeroed = 0
    for k in range(1, data.shape[-2]):
        kill = np.abs(data[..., k, :]) < threshold_step * k
        zeroed += int(np.count_nonzero(data[..., k, :][kill]))
        data[..., k, :][kill] = 0.0
    return data, zeroed, before


def quantize_sh_unorm8(data: np.ndarray, coverage: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(H, W, K, 3) -> (uint8 (H, W, K, 3), offset (K, 3), scale (K, 3)), coefficient ~= offset + scale * (q / 255).
    Ranges are min..max over `coverage`; texels outside it are clamped."""
    data = np.asarray(data, dtype=np.float32)
    if not coverage.any():
        coverage = np.ones(data.shape[:2], dtype=bool)
    covered = data[coverage]  # (N, K, 3)
    offset = covered.min(axis=0)
    scale = covered.max(axis=0) - offset
    safe = np.where(scale > 0, scale, 1.0)
    q = np.clip(np.round((data - offset) / safe * 255.0), 0, 255).astype(np.uint8)
    return q, offset, scale


def round_float16_mantissa(data: np.ndarray, mantissa_bits: int) -> np.ndarray:
    """Rounds each value to `mantissa_bits` of float16's 10, still stored as float16."""
    half = np.float16(data)
    if not 1 <= mantissa_bits < 10:
        return half
    dropped = 10 - mantissa_bits
    bits = half.view(np.uint16).astype(np.uint32)
    carried = np.minimum(bits + (1 << (dropped - 1)), 0xFFFF)  # no wrap past the NaN patterns
    rounded = (carried & ~np.uint32((1 << dropped) - 1)).astype(np.uint16)
    out = rounded.view(np.float16)
    # Rounding up out of 65504 lands on inf.
    return np.where(np.isfinite(half) & ~np.isfinite(out), half, out)


def write_render_bundle(
    uv_mesh: UVMesh,
    sh_map: SHTextureMap,
    height_map: HeightMap,
    bundle_dir: Path,
    sh_storage: str,
    config: RenderBundleConfig | None = None,
    camera_centers: np.ndarray | None = None,
) -> dict:
    """Writes the four bundle files into `bundle_dir` and returns the config dict.
    `sh_storage` is one of `SH_STORAGES`; `camera_centers` (N, 3) only aim the initial view."""
    config = config or RenderBundleConfig()
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # glTF's UV origin is top-left, this repo's is bottom-left.
    uv_gltf = uv_mesh.uv.astype(np.float32).copy()
    uv_gltf[:, 1] = 1.0 - uv_gltf[:, 1]
    write_glb(bundle_dir / BUNDLE_FILES["mesh"], uv_mesh.vertices, uv_gltf, uv_mesh.faces)

    valid = height_map.valid if height_map.valid is not None else np.ones(height_map.data.shape, bool)
    coeffs, zeroed, nonzero = sparsify_sh(sh_map.data, config.sh_zero_threshold_step)
    height, width, k_coeffs, _ = coeffs.shape
    sh_quant = None
    if sh_storage == "unorm8":
        q, offset, scale = quantize_sh_unorm8(coeffs, valid)
        layers = np.zeros((k_coeffs, height, width, 4), dtype=np.uint8)
        layers[..., :3] = np.transpose(q, (2, 0, 1, 3))
        sh_quant = {"offset": offset.tolist(), "scale": scale.tolist()}
    elif sh_storage == "float16":
        coeffs = round_float16_mantissa(coeffs, config.sh_mantissa_bits)
        layers = np.zeros((k_coeffs, height, width, 4), dtype=np.float16)
        layers[..., :3] = np.transpose(coeffs, (2, 0, 1, 3))
    else:
        raise ValueError(f"sh_storage must be one of {SH_STORAGES}, got {sh_storage!r}")
    write_ktx2(bundle_dir / BUNDLE_FILES["sh"], layers, level=config.compression_level)

    relief = np.stack([height_map.data.astype(np.float32), valid.astype(np.float32)], axis=-1)
    write_ktx2(bundle_dir / BUNDLE_FILES["height"], relief, level=config.compression_level)

    # Displacement is stored raw in scene units.
    layer_step = 2.0 * height_map.march_range() / max(config.num_steps, 1)
    quant_error = float(np.abs(height_map.data.astype(np.float16).astype(np.float32) - height_map.data).max())
    if quant_error > 0.01 * layer_step:
        print(f"  warning: float16 height error {quant_error:.3g} is {quant_error / layer_step:.1%} of a march layer")

    up = sh_map.up_vector if sh_map.up_vector is not None else DEFAULT_UP
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)

    target = uv_mesh.vertices.mean(axis=0)
    extent = uv_mesh.vertices.max(axis=0) - uv_mesh.vertices.min(axis=0)
    radius = max(float(np.linalg.norm(extent)) * config.radius_scale, 1e-3)
    camera = OrbitCamera(origin=target, radius=radius, up=up)
    camera.azimuth, camera.elevation = config.azimuth_deg, config.elevation_deg
    if camera_centers is not None and len(camera_centers):
        # Mean direction, not mean position.
        offsets = np.asarray(camera_centers, dtype=np.float64) - target
        norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        mean_dir = (offsets / np.maximum(norms, 1e-12)).mean(axis=0)
        if np.linalg.norm(mean_dir) > 1e-6:
            camera.azimuth, camera.elevation = orbit_angles(mean_dir, up)

    tex_w, tex_h = uv_mesh.texture_resolution
    bundle = {
        "format": "sh_texture_bundle/3",
        "up_vector": up.tolist(),
        "sh": {
            "degree": int(sh_map.degree),
            "coefficients": int(k_coeffs),  # == layers in sh.ktx2, one per coefficient
            "zero_threshold_step": float(config.sh_zero_threshold_step),
            "zeroed_fraction": float(zeroed / nonzero) if nonzero else 0.0,
            # "unorm8": coefficient = offset[k][c] + scale[k][c] * sample.
            "storage": sh_storage,
            **({"mantissa_bits": int(config.sh_mantissa_bits)} if sh_quant is None else sh_quant),
        },
        "texture_resolution": [int(tex_w), int(tex_h)],
        "height": {
            # Parallax slab, in the mesh's own units.
            "range": height_map.march_range(),
            "num_steps": int(config.num_steps),
            "units": height_map.units,
        },
        "initial_camera": {
            "type": "orbit",
            "target": target.tolist(),
            "radius": radius,
            "azimuth_deg": camera.azimuth,
            "elevation_deg": camera.elevation,
            "position": camera.position.tolist(),
            # Row-major GL view matrix for the pose above.
            "view_matrix": camera.view_matrix().tolist(),
            "fov_deg": config.fov_deg,
            "near": radius * config.near_scale,
            "far": radius * config.far_scale,
        },
        "mesh": {"vertices": int(len(uv_mesh.vertices)), "faces": int(len(uv_mesh.faces))},
        "files": dict(BUNDLE_FILES),
        "rotation": {"center": [0,0,0], "axis": up.tolist()}
    }
    (bundle_dir / BUNDLE_FILES["config"]).write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle
