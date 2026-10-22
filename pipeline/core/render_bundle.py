"""Packages a finished run into one self-contained `.snap3d` folder another project
renders from.

Four files, in formats a web renderer already has loaders for, and nothing that can be
derived from them. No manifests, no stage numbering, no previews, no ingest images, no
dependency on this repo: a consumer reads `config.json`, loads a GLB and two KTX2
textures, and draws.

    <run_id>_<storage>.snap3d/      one per storage in SH_STORAGES: "unorm8" and "float16"
      mesh.glb      POSITION (V,3) f32, TEXCOORD_0 (V,2) f32, indices (F,3) u32
      sh.ktx2       2D array, one layer per SH coefficient; rgb = the coefficient's RGB,
                    a = 0 (padding - see pipeline/core/ktx2.py for why 3-channel is not
                    an option). RGBA16F holding the coefficients themselves when
                    `sh.storage` is "float16"; RGBA8 UNORM when it is "unorm8", where
                    coefficient = sh.offset[k][c] + sh.scale[k][c] * sample, sample the
                    normalized [0, 1] value the GPU returns
      height.ktx2   RG16F; r = displacement in `height.units`, g = 1 inside the atlas
                    coverage and 0 outside
      config.json   up_vector, sh{...}, texture_resolution, height{...}, initial_camera

**`.snap3d` names the folder, not a container.** It is named after the run it came from
(`photo_20260904_000809.snap3d`), so a bundle that has been copied out of this repo's
tree still says which capture it is - the run id is the one piece of provenance the
bundle itself does not otherwise carry. The four files sit in it as ordinary
files, each fetchable on its own - which is the point: a consumer given one directory URL
issues three parallel requests, and each payload is cached, revalidated and served as
itself rather than as a slice of an archive that has to arrive whole before any of it can
be read. The suffix exists so that one directory can be handed over as *the* deliverable
and recognised as one, the way `.app` or `.framework` do it; there is nothing to unpack,
no offsets to parse, and `config.json` is still the entry point a viewer opens.

**Both files are in their format's own convention, not this repo's.** glTF puts the UV
origin at the image's top-left and KTX2 stores row 0 first, so the mesh's `TEXCOORD_0`
is written with v flipped relative to the UVMesh this repo passes around, and the
textures are written top row first. A consumer that just follows glTF and KTX2 is
therefore correct with no special-casing; it is `view_bundle.py`, reusing this repo's
v-up shader, that converts on the way in.

**The per-vertex frame is recomputed, not shipped.** The shader also wants a normal and
the raw (un-normalized) dP/du, dP/dv per vertex, but all three are pure functions of
position+uv+indices - `uv_geometry.vertex_normals_for` and `compute_vertex_tangents`,
28 ms for 65k vertices - so storing them would nearly triple the geometry payload to
save a load-time cost that does not register. A consumer must reproduce those two
functions exactly - a different frame is a different resolved uv, hence different
colors - and "exactly" means:

  * **Normals: angle-weighted, not area-weighted.** `vertex_normals_for` delegates to
    trimesh's smooth vertex normals, which weight each incident face normal by that
    face's *interior angle at the vertex*. Verified against this repo's own meshes:
    angle weighting reproduces the output to 1.3e-14, while area weighting is off by up
    to 1.6 and uniform averaging by 0.83 - i.e. an area-weighted consumer gets exactly
    the subtly-wrong-colours-everywhere this paragraph warns about.

  * **Tangents: Lengyel**, accumulated over incident faces and *not* normalized, since
    their length is what converts a world step into a texel step.

**Coefficients are stored either as 8-bit UNORM with a per-(coefficient, channel) affine
map, or as float16 cut twice before compression.** `unorm8` is the pipeline default:
each (k, channel) is quantized over its own min..max across the atlas coverage, into
256 levels, and `config.json` carries the offset and scale. The dequantization is affine,
so a renderer can upload the bytes as RGBA8, let the GPU filter them bilinearly and apply
offset + scale after the lookup - exactly equivalent to filtering the dequantized values,
at half the GPU memory of RGBA16F. Measured on boots (1024 atlas, 114 views, masked):
sh.ktx2 2.28 MB against float16's 8.97 MB; PSNR 22.89 against 23.18 dB, SSIM 0.762
against 0.799, LPIPS 0.267 against 0.186. The range is a plain min..max, so a few
ill-conditioned texels with extreme coefficients widen every texel's step (13-20 colour
levels on boots); clipping to percentiles was measured too and costs far less quality
(23.14 dB, LPIPS 0.199) at 4.17 MB, if that trade is ever wanted.

The float16 bundle cuts twice before compression - and both cuts
exist only to give the compressor something to find. Neither changes the file's format: it is still
RGBA16F, still sampled by the GPU as half floats.

  * *Near-zero high-order coefficients become exactly zero*, following the reference
    implementation (`surface-light-fields-for-planar-paintings`,
    `PerPixelSH.save_sh_pngs`): coefficient k loses any value under
    `sh_zero_threshold_step * k`, so the higher the order - the smaller its visual
    contribution - the more of it collapses into runs. k=0 is never touched; it carries
    the base colour.

  * *The low mantissa bits are rounded away*, keeping `sh_mantissa_bits` of float16's 10.
    This is the bigger lever by far, because what is left after the zeroing is
    high-entropy noise in the bottom of each mantissa: the values differ in bits that no
    renderer can show. Being an exponent-relative cut, it costs the same fraction of
    every coefficient regardless of magnitude.

Measured over three views on four runs, against the float32 fit: the zeroing alone lands
at 45-50 dB PSNR for 23-31% of the raw float32 size, and each mantissa bit dropped from
there takes off another ~9% of the file. Through 6 bits the quality cost is under 1.1 dB
and the share of drawn pixels off by more than 2/255 moves by less than 0.15 points; the
5-bit default costs at most 2.5 dB more and 0.5 points. 4 bits is where it breaks - the
error suddenly reaches 5-10% of drawn pixels - so the default sits one step above it.

Two things in `config.json` are format, not tuning:

  * **`height.range` and `height.num_steps`** define the slab the parallax march walks.
    A renderer using different ones resolves a different uv per fragment and reads the
    SH coefficients at a texel they were never fit for.

  * **The initial camera looks from where the capture was shot.** A fixed azimuth is not
    safe: on a flat object it lands behind the surface and the bundle opens on a black
    screen (measured on the Holy Family run). Averaging the ingest cameras' directions
    from the centroid always picks a side the object was actually photographed from.
    Shipped as orbit parameters *and* a resolved eye/target/view matrix, so a consumer
    with its own camera rig can skip the orbit.

`height.ktx2`'s green channel is the atlas coverage baking produced (chart interiors
plus their gutter), which is what a renderer discards on - not the SH fit's own validity
mask. Texels inside the coverage that no view observed carry a nearest-neighbour copy of
a real fit, so they shade correctly; masking them out would punch holes in the surface.
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

# The naming convention that marks a directory as a finished, self-contained bundle.
# Nothing reads it back - it is a label for whoever receives the folder.
BUNDLE_SUFFIX = ".snap3d"

BUNDLE_FILES = {
    "mesh": "mesh.glb",
    "sh": "sh.ktx2",
    "height": "height.ktx2",
    "config": "config.json",
}


# Every export writes one bundle per storage; there is no choosing between them. unorm8 is
# the smaller (RGBA8, half the GPU memory), float16 the more exact.
SH_STORAGES = ("unorm8", "float16")


def bundle_dir_name(run_name: str, sh_storage: str) -> str:
    """`<run_id>_<storage>.snap3d` - the folder the four files go in. `run_name` is the
    run directory's name, which the export stage takes from its own parent."""
    if sh_storage not in SH_STORAGES:
        raise ValueError(f"sh_storage must be one of {SH_STORAGES}, got {sh_storage!r}")
    return f"{run_name}_{sh_storage}{BUNDLE_SUFFIX}"


@dataclass
class RenderBundleConfig:
    # POM layers. Must match the slf fit's `num_steps`; the stage wrapper reads it off
    # that stage's manifest, so this default only applies to a direct call.
    num_steps: int = 7
    # Coefficient k is flattened to zero below `sh_zero_threshold_step * k`; 0 disables
    # it. The default is the reference implementation's. See the module docstring.
    sh_zero_threshold_step: float = 0.02
    # Mantissa bits kept of float16's 10; 10 disables the rounding. See the module
    # docstring for the measured curve behind this default.
    sh_mantissa_bits: int = 5
    # zlib level for the KTX2 payloads; see pipeline/core/ktx2.py for why zlib.
    compression_level: int = 9
    # Initial view. azimuth/elevation are only the fallback for when no camera centers
    # are given; otherwise they are measured off the capture (see the module docstring).
    fov_deg: float = 45.0
    azimuth_deg: float = 45.0
    elevation_deg: float = 20.0
    radius_scale: float = 1.2  # bounding-box diagonal -> orbit radius
    near_scale: float = 0.02  # near/far as multiples of the radius, as the viewer does
    far_scale: float = 20.0


def sparsify_sh(data: np.ndarray, threshold_step: float) -> tuple[np.ndarray, int, int]:
    """Zeroes coefficient k wherever |value| < threshold_step * k, k >= 1.

    Returns (data, values zeroed, non-zero higher-order values before the cut) - the
    denominator excludes texels that were already zero, which on a typical atlas is 40%
    of the map and would otherwise flatter the ratio."""
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
    """(H, W, K, 3) coefficients -> (uint8 (H, W, K, 3), offset (K, 3), scale (K, 3)) with
    coefficient ~= offset + scale * (q / 255). The range of each (k, channel) is its
    min..max over `coverage` (the texels a renderer draws); texels outside it are clamped
    into that range, which only matters for bilinear taps across a chart edge's gutter."""
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
    """Rounds each value to `mantissa_bits` of float16's 10, still stored as float16.

    Nothing about the file's format changes - it is still RGBA16F, still sampled by the
    GPU as half floats - only the low mantissa bits are forced to zero, which is what
    lets the compressor find long matches in what is otherwise near-random noise. Because it works
    on the exponent-mantissa split it is a *relative* precision cut, uniform over the
    three orders of magnitude the coefficients span, unlike a fixed-point step that would
    have to be sized for the largest of them.

    Round-to-nearest (ties away from zero) on the sign-magnitude bit pattern, so a carry
    out of the mantissa correctly increments the exponent; the only value that could
    carry all the way to infinity is float16's largest finite one, which is clamped back."""
    half = np.float16(data)
    if not 1 <= mantissa_bits < 10:
        return half
    dropped = 10 - mantissa_bits
    bits = half.view(np.uint16).astype(np.uint32)
    carried = np.minimum(bits + (1 << (dropped - 1)), 0xFFFF)  # no wrap past the NaN patterns
    rounded = (carried & ~np.uint32((1 << dropped) - 1)).astype(np.uint16)
    out = rounded.view(np.float16)
    # Rounding up out of the top normal (65504) lands on inf; nothing else can overflow.
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
    """Writes the four bundle files into `bundle_dir` and returns the config dict it
    wrote. `bundle_dir` is the `.snap3d` folder itself (the export stage names it with
    `bundle_dir_name`), and it is created if missing. `sh_storage` is one of `SH_STORAGES`:
    "float16" is RGBA16F coefficients (zeroing + mantissa rounding above); "unorm8" is
    RGBA8 with a per-(coefficient, channel) min..max affine map in config.json (the zeroing
    still applies first, the mantissa rounding does not). See the module docstring. `camera_centers` (N, 3) are the
    capture's camera positions; they only aim the initial view."""
    config = config or RenderBundleConfig()
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # glTF's UV origin is the image's top-left; this repo's is the bottom-left (see the
    # module docstring and uv_geometry.py). The flip belongs here, once, rather than in
    # every consumer.
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

    # The displacement is stored raw, in scene units, so the shader needs no scale
    # factor - but that makes float16's absolute precision scene-dependent. What it has
    # to stay well under is one march layer, which is what the resolve quantizes to.
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
        # Mean *direction*, not mean position: a capture that arcs around the object has
        # a mean position near its centre, which points nowhere.
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
            # "float16": sh.ktx2 samples are the coefficients. "unorm8": coefficient =
            # offset[k][c] + scale[k][c] * sample, sample in [0, 1] as the GPU returns it.
            "storage": sh_storage,
            **({"mantissa_bits": int(config.sh_mantissa_bits)} if sh_quant is None else sh_quant),
        },
        "texture_resolution": [int(tex_w), int(tex_h)],
        "height": {
            # Parallax slab, in the mesh's own units. See the module docstring.
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
            # Row-major GL view matrix for the pose above, so a consumer needs neither
            # the orbit convention nor a look_at of its own to reproduce it.
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
