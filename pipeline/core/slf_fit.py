"""height baseline + posed images -> per-texel SH reflectance fit (SLF).

Ports the CG + AtA/Atb fitting core of teo646/surface-light-fields-for-planar-
paintings @ 53f5787 from a single flat painting to an arbitrary UV-mapped mesh:

  - The reference's one global (u_axis, v_axis, normal) frame becomes a per-texel
    frame from the UV parametrization (uv_geometry.orthonormal_frame).
  - Its GPU "UV pass" (rasterize, then per-fragment POM march of a height texture to
    resolve which uv a photo pixel really observes) becomes camera_raster's
    `ParallaxRasterContext`, rasterizing the real low-poly mesh with hardware depth
    testing. The march is slf_shaders.RESOLVE_GLSL - literally the same source the
    viewers use, not a port of it.
  - Per-photo texel colors come from the same matrix-free CG solve of
    (B^T B + ridge I) u = B^T b (sh_math.cg_solve_bilinear_texels); SH coefficients
    accumulate into per-texel AtA/Atb across photos, precision-weighted, solved with
    torch.linalg.solve. `d` is `view_local`, already relative to the texel's own
    normal, so sh_math.hsh_remap applies per texel rather than once globally.

Coefficients are fit against raw [0, 1] RGB with no storage centering: the
reference's "-0.5" trick was specific to its uint16 PNG encoding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from pipeline.core.camera_raster import ParallaxRasterContext
from pipeline.core.sh_math import bilinear_neighbors, cg_solve_bilinear_texels, hsh_remap, sh_basis
from pipeline.core.uv_geometry import orthonormal_frame, rasterize_uv_atlas, uv_to_atlas_xy_torch
from pipeline.schemas.posed_images import PosedImageSet
from pipeline.schemas.texture_maps import HeightMap, SHTextureMap
from pipeline.schemas.uv_mesh import UVMesh

__all__ = ["SLFFitConfig", "run_slf_fit", "TEXEL_STATUS_COLORS", "save_texel_status_map"]

# Per-texel fit outcome -> (name, BGR for cv2.imwrite, one-line meaning). Order is
# priority: a texel is classified as the first of these it matches, and the five are
# mutually exclusive and exhaustive over every texel of the atlas.
TEXEL_STATUS_COLORS: dict[str, tuple[tuple[int, int, int], str]] = {
    # Values are (B, G, R) - cv2's channel order, since save_texel_status_map writes
    # them straight into an image via cv2.imwrite. The legend .json reverses each to
    # RGB. Comments below name the colour as it actually displays.
    "background": (
        (0, 255, 255),  # -> RGB (255,255,0): yellow
        "atlas 밖 (rasterize_uv_atlas + baking의 gutter 확장에도 포함 안 됨); 렌더에서 절대 샘플되지 않음",
    ),
    "well_fit": (
        (255, 90, 40),  # -> RGB (40,90,255): blue
        "min_observations_per_texel 이상 관측 + precision 충분히 확보 (well_conditioned); 정상적으로 SH 전체 차수 피팅됨",
    ),
    "under_observed": (
        (30, 30, 230),  # -> RGB (230,30,30): red
        "관측은 됐지만 min_observations_per_texel 미만이라 degree-0 평균(DC항만)으로 대체됨",
    ),
    "rejected_low_precision": (
        (0, 150, 255),  # -> RGB (255,150,0): orange
        "관측은 있었으나 총 precision이 ridge*min_precision_ratio 미만이라 해가 ridge에 지배됨 -> 버려지고 nearest-copy로 채워짐",
    ),
    "never_observed": (
        (200, 40, 160),  # -> RGB (160,40,200): purple
        "이 텍셀을 잡은 관측이 하나도 없음(코어 텍셀이면 마스크/뎁스에 전부 걸렀거나 어떤 사진도 안 봄, gutter면 흔한 경우) -> nearest-copy로 채워짐",
    ),
}


def save_texel_status_map(
    path: Path,
    atlas_valid: np.ndarray,
    well_fit: np.ndarray,
    under_observed: np.ndarray,
    rejected_low_precision: np.ndarray,
    never_observed: np.ndarray,
) -> None:
    """Writes a flat-colour (H, W, 3) PNG at `path`, one TEXEL_STATUS_COLORS entry per
    texel, plus a `<path>.legend.json` mapping each colour to its name and meaning.
    All five boolean masks are (H, W) and are treated in TEXEL_STATUS_COLORS' order,
    so a texel is coloured by the first mask it is True in - background first (nothing
    below it can be true for a texel outside the atlas), well_fit next, and so on."""
    import json

    h, w = atlas_valid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    masks = {
        "background": ~atlas_valid,
        "well_fit": well_fit,
        "under_observed": under_observed,
        "rejected_low_precision": rejected_low_precision,
        "never_observed": never_observed,
    }
    assigned = np.zeros((h, w), dtype=bool)
    for name, (color, _meaning) in TEXEL_STATUS_COLORS.items():
        m = masks[name] & ~assigned
        img[m] = color
        assigned |= m
    stray = atlas_valid & ~assigned
    if stray.any():
        raise AssertionError(f"{int(stray.sum())} valid texels matched none of the five states - a fit_slf bug")

    path = Path(path)
    cv2.imwrite(str(path), img)
    path.with_suffix(".legend.json").write_text(json.dumps(
        {name: {"color_rgb": list(color[::-1]), "meaning": meaning}
         for name, (color, meaning) in TEXEL_STATUS_COLORS.items()},
        ensure_ascii=False, indent=1,
    ))



@dataclass
class SLFFitConfig:
    sh_degree: int = 1  # (degree + 1) ** 2 coefficients per channel, e.g. degree=2 -> 9; supports 0-3
    min_observations_per_texel: int = 4  # below this, fall back to a flat degree-0 average (see run_slf_fit)
    ridge: float = 1e-4  # SH normal-equations ridge (AtA diagonal)
    num_steps: int = 7  # POM ray-march layers - match view_slf.py's --num-steps to be exact
    recon_min_precision: float = 0.0
    min_precision_ridge_ratio: float = 10.0
    # Reject a pixel whose resolved surface depth differs from the reference (high-poly)
    # mesh's depth at that pixel by more than this fraction of the reference depth - it
    # landed on a different surface than the photo shows, typically the back part where
    # two parts of the object overlap. Relative, so it means the same at any capture
    # scale. 0 disables the test and the reference mesh is not needed.
    depth_test_tolerance: float = 0.0
    # What the depth test compares against: "highpoly" renders the fused mesh baking
    # baked from; "gaussians" loads per-view depth maps rendered from the trained
    # Gaussians (pipeline.core.gaussian_depth), which keep structure thinner than the
    # mesh's TSDF voxels - fibres, thin edges - that still occludes in a photo.
    depth_reference: str = "highpoly"
    # Excludes a photo's outer border from sampling: a pixel within this fraction of
    # image width/height of any edge is dropped before the mask/depth tests, same as
    # them. COLMAP's `image_undistorter` (resize_dataset's or ingest's own) has no
    # crop or border-validity mask, so a lens with real distortion can leave black (or
    # otherwise unsourced) pixels at the frame edge that this pipeline never detects -
    # this is a blunt guard against fitting them as real colour, at the cost of ignoring
    # real image content in that margin. 0 disables it (the default: most of this
    # project's captures are phone cameras with negligible distortion - see README).
    border_margin_frac: float = 0.0
    device: str = "cuda"


def _load_photo(path: Path, device: torch.device) -> torch.Tensor | None:
    """(H, W, 3) float32 RGB in [0, 1] on `device`, or None if unreadable. JPEGs decode
    on the GPU (nvJPEG); anything else through cv2. Both ignore EXIF Orientation, as
    COLMAP, PIL and SAM3 do: cv2 would otherwise rotate a phone capture's
    Orientation-6 JPEG and sample every observation from a transposed photo.
    torchvision's decode_jpeg leaves orientation alone unless asked."""
    if device.type == "cuda" and path.suffix.lower() in (".jpg", ".jpeg"):
        from torchvision.io import ImageReadMode, decode_jpeg, read_file

        try:
            rgb = decode_jpeg(read_file(str(path)), mode=ImageReadMode.RGB, device=device)
            return rgb.permute(1, 2, 0).float().div_(255.0)
        except RuntimeError:
            pass  # e.g. a progressive or CMYK JPEG nvJPEG rejects - fall through to cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        return None
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).to(device).float().div_(255.0)


def run_slf_fit(
    posed_images: PosedImageSet,
    images_dir: Path,
    uv_mesh: UVMesh,
    height_map: HeightMap,
    config: SLFFitConfig | None = None,
    reference_mesh=None,  # pipeline.schemas.mesh.Mesh - the high-poly mesh height_map was baked from
    reference_depth_dir: Path | None = None,  # <image stem>.npy per view, for depth_reference="gaussians"
    masks_dir: Path | None = None,  # masking's <image file name>.png per view; None fits every resolved pixel
    fit_stats_path: Path | None = None,  # if given, the .npz the slf_bound stage re-fits from
    debug_status_path: Path | None = None,  # if given, a per-texel fit-status PNG - see TEXEL_STATUS_COLORS
) -> SHTextureMap:
    """Fits the SH texture map. With `fit_stats_path`, also saves what a later re-fit
    needs without re-reading any photo: per observed texel the normal equations (AtA
    with the ridge on, Atb), the observed colour range, and whether the texel fell back
    to a flat average; plus the hole-fill map (each padded texel's source texel), so a
    re-fit can re-pad from its own coefficients. See pipeline/core/slf_bound.py.

    With `debug_status_path`, also saves a flat-colour PNG classifying every texel of
    the atlas into one of TEXEL_STATUS_COLORS - which texels got a real fit, which are
    padding, and why, at a glance."""
    config = config or SLFFitConfig()
    if config.depth_reference not in ("highpoly", "gaussians"):
        raise ValueError(f"depth_reference must be 'highpoly' or 'gaussians', got {config.depth_reference!r}")
    use_mesh_reference = config.depth_test_tolerance > 0 and config.depth_reference == "highpoly"
    use_map_reference = config.depth_test_tolerance > 0 and config.depth_reference == "gaussians"
    if use_mesh_reference and reference_mesh is None:
        raise ValueError("depth_test_tolerance > 0 with depth_reference='highpoly' needs the high-poly mesh")
    if use_map_reference and reference_depth_dir is None:
        raise ValueError("depth_test_tolerance > 0 with depth_reference='gaussians' needs reference_depth_dir")
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    W, H = uv_mesh.texture_resolution
    K = (config.sh_degree + 1) ** 2
    P = H * W

    atlas = rasterize_uv_atlas(uv_mesh, target_valid=height_map.valid)

    if masks_dir is not None and not masks_dir.is_dir():
        raise ValueError(f"masks_dir {masks_dir} is not a directory")

    # One frame for both the POM march and the SH fit: the *flat* low-poly tangent
    # plane, which is what height_map's values are measured against (baking.py casts
    # rays along that normal). It is shipped as vertex attributes, not textures - a
    # renderer rebuilds it per fragment from the same arrays the atlas interpolates
    # (uv_geometry.vertex_frame_attributes, slf_shaders.uv_frame), so the rebuilt
    # frame is this one rather than an approximation of it.
    tangent_flat, bitangent_flat = orthonormal_frame(atlas.tangent_raw, atlas.normals)

    # The atlas-pixel shift per world-unit of height (the reference's
    # `u_uv_per_length`, per texel here) is slf_shaders.xy_per_height now: the raw
    # tangent attributes carry the lengths it is made of.
    height_range = height_map.march_range()

    pos_flat_t = torch.from_numpy(atlas.positions.reshape(P, 3).astype(np.float32)).to(device)
    tangent_flat_t = torch.from_numpy(tangent_flat.reshape(P, 3).astype(np.float32)).to(device)
    bitangent_flat_t = torch.from_numpy(bitangent_flat.reshape(P, 3).astype(np.float32)).to(device)
    normal_flat_t = torch.from_numpy(atlas.normals.reshape(P, 3).astype(np.float32)).to(device)

    AtA = torch.zeros(P, K, K, dtype=torch.float32, device=device)
    Atb = torch.zeros(P, K, 3, dtype=torch.float32, device=device)
    support_count = torch.zeros(P, dtype=torch.int32, device=device)
    total_precision = torch.zeros(P, dtype=torch.float32, device=device)
    # Per-texel, per-channel range of the per-photo colours the fit is fed - saved with
    # the normal equations (fit_stats_path) for the slf_bound stage. Taken over the
    # CG-solved texel colours rather than raw pixels, so one bright speck in one photo
    # does not set a texel's bound.
    obs_min = torch.full((P, 3), float("inf"), dtype=torch.float32, device=device)
    obs_max = torch.full((P, 3), float("-inf"), dtype=torch.float32, device=device)

    # One GPU context/program/texture set for the whole run; only the MVP changes per
    # photo. The same rasterize-then-march GLSL pass the viewers use at render time.
    raster_ctx = ParallaxRasterContext(
        uv_mesh, atlas, height_map.data, height_range, config.num_steps,
        reference_mesh=reference_mesh if use_mesh_reference else None,
    )
    border_masks: dict[tuple[int, int], torch.Tensor] = {}  # (height, width) -> keep-interior mask, built once each

    def _border_mask(height: int, width: int) -> torch.Tensor:
        key = (height, width)
        if key not in border_masks:
            my, mx = round(height * config.border_margin_frac), round(width * config.border_margin_frac)
            m = torch.zeros(height, width, dtype=torch.bool, device=device)
            m[my : height - my if my else height, mx : width - mx if mx else width] = True
            border_masks[key] = m
        return border_masks[key]

    try:
        n_images = len(posed_images.images)
        for i, img in enumerate(posed_images.images):
            t0 = time.time()
            cam = posed_images.cameras[img.camera_id]
            img_path = posed_images.image_path(images_dir, img)
            photo = _load_photo(img_path, device)
            if photo is None:
                print(f"  [{i + 1:3d}/{n_images}] {img.file_name}: could not read image, skipping")
                continue
            if photo.shape[:2] != (cam.height, cam.width):
                raise ValueError(f"{img.file_name}: image is {tuple(photo.shape[:2])}, camera says {(cam.height, cam.width)}")

            K_mat = cam.as_matrix()
            R = img.pose.rotation_matrix()
            t = np.array([img.pose.tx, img.pose.ty, img.pose.tz], dtype=np.float64)
            cam_center = img.pose.camera_center()

            # The object mask, if masking ran. Without it a pixel is fit into the texel
            # its uv resolves to whatever the photo shows there - and where something
            # crosses in front of the object (a hand, a cable), that intruder's colour is
            # what lands in the texel. The depth test only catches an occluder the
            # reference geometry itself contains, which a transient one is not.
            mask = None
            if masks_dir is not None:
                mask_path = masks_dir / f"{img.file_name}.png"
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    raise FileNotFoundError(f"{img.file_name}: no mask at {mask_path}")
                # The masks may be at either resolution - masking runs on
                # resize_dataset's possibly-downscaled frames. The downscale is a pure
                # resize, so nearest-neighbour keeps the mask binary and aligned.
                if mask_img.shape != (cam.height, cam.width):
                    mask_img = cv2.resize(mask_img, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST)
                mask = torch.from_numpy(mask_img).to(device) > 127

            # `keep` already means both "a triangle covers this pixel" and "the
            # resolved uv is real, covered atlas territory" - both discarded in-shader,
            # so no separate filtering is needed here.
            # Everything per-pixel from here stays on the GPU: as numpy on 12 MP arrays
            # the mask/zero/gather steps were most of each photo's time.
            uv_full, keep, surface_depth, reference_depth = raster_ctx.resolve_torch(
                K_mat, R, t, cam_center, cam.width, cam.height, device
            )
            n_border_rejected = 0
            if config.border_margin_frac > 0:
                border = _border_mask(cam.height, cam.width)
                n_border_rejected = int((keep & ~border).sum().item())
                keep = keep & border
            n_mask_rejected = 0
            if mask is not None:
                n_mask_rejected = int((keep & ~mask).sum().item())
                keep = keep & mask
            n_depth_rejected = 0
            if use_map_reference:
                ref_np = np.load(reference_depth_dir / f"{Path(img.file_name).stem}.npy")
                if ref_np.shape != tuple(keep.shape):
                    raise ValueError(f"{img.file_name}: reference depth {ref_np.shape} vs camera {tuple(keep.shape)}")
                reference_depth = torch.from_numpy(ref_np).to(device, non_blocking=True)
            if reference_depth is not None:
                # reference_depth is 0 where the reference covers nothing, which fails
                # this too: a low-poly silhouette overhanging empty space.
                agrees = (surface_depth - reference_depth).abs() <= config.depth_test_tolerance * reference_depth
                n_depth_rejected = int((keep & ~agrees).sum().item())
                keep = keep & agrees
            # Boolean-mask indexing is row-major, the same observation order np.nonzero gave.
            uv_kept = uv_full[keep]
            if uv_kept.shape[0] == 0:
                print(f"  [{i + 1:3d}/{n_images}] {img.file_name}: 0 valid after parallax resolve, skipping")
                continue
            # uv -> texel-index space through the one shared half-texel map, so the
            # weights the CG solve inverts are the ones texture() will apply.
            xy_t = uv_to_atlas_xy_torch(uv_kept, W, H)
            colors_t = photo[keep]
            x0, x1, y0, y1, fx, fy = bilinear_neighbors(xy_t[:, 0], xy_t[:, 1], W, H)
            idx = torch.stack([y0 * W + x0, y0 * W + x1, y1 * W + x0, y1 * W + x1], dim=1)
            wts = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], dim=1)

            texel_color, precision = cg_solve_bilinear_texels(idx, wts, colors_t, P=P)
            owned = torch.nonzero(precision > config.recon_min_precision, as_tuple=True)[0]
            if len(owned) == 0:
                print(f"  [{i + 1:3d}/{n_images}] {img.file_name}: 0 texels above precision threshold, skipping")
                continue

            cam_center_t = torch.from_numpy(cam_center.astype(np.float32)).to(device)
            view_world = cam_center_t.unsqueeze(0) - pos_flat_t[owned]
            view_world = view_world / view_world.norm(dim=1, keepdim=True).clamp_min(1e-10)
            view_local = torch.stack(
                [
                    (view_world * tangent_flat_t[owned]).sum(dim=1),
                    (view_world * bitangent_flat_t[owned]).sum(dim=1),
                    (view_world * normal_flat_t[owned]).sum(dim=1),
                ],
                dim=-1,
            )
            d = view_local / view_local.norm(dim=1, keepdim=True).clamp_min(1e-10)
            Y = sh_basis(config.sh_degree, hsh_remap(d))
            w = precision[owned]
            Y_w = Y * w.unsqueeze(1)

            outer = Y_w.unsqueeze(2) * Y.unsqueeze(1)
            AtA.index_add_(0, owned, outer)
            cross = Y_w.unsqueeze(2) * texel_color[owned].unsqueeze(1)
            Atb.index_add_(0, owned, cross)
            support_count.index_add_(0, owned, torch.ones_like(owned, dtype=torch.int32))
            total_precision.index_add_(0, owned, w)
            owned3 = owned.unsqueeze(1).expand(-1, 3)
            obs_min.scatter_reduce_(0, owned3, texel_color[owned], reduce="amin")
            obs_max.scatter_reduce_(0, owned3, texel_color[owned], reduce="amax")

            border_note = f", {n_border_rejected} px border-rejected" if config.border_margin_frac > 0 else ""
            mask_note = f"{border_note}, {n_mask_rejected} px mask-rejected" if mask is not None else border_note
            depth_note = f"{mask_note}, {n_depth_rejected} px depth-rejected" if reference_depth is not None else mask_note
            print(f"  [{i + 1:3d}/{n_images}] {img.file_name}: {len(owned):7d} texels{depth_note}  ({time.time() - t0:.1f}s)")
    finally:
        raster_ctx.release()

    AtA.diagonal(dim1=1, dim2=2).add_(config.ridge)
    coeffs = torch.linalg.solve(AtA, Atb)

    # Observed = the normal equations carry more weight than the ridge prior. Texels
    # failing this solved toward black; the dilation below overwrites them.
    well_conditioned = total_precision >= config.ridge * config.min_precision_ridge_ratio
    is_observed = (support_count > 0) & well_conditioned

    # Too few views to trust the full-degree solve, but a degree-0 average is still
    # meaningful: overwrite with [avg_color, 0, ...]. Atb[:,0,:]/AtA[:,0,0] is exactly
    # the precision-weighted mean color (the C0 in Y[0]=C0 cancels), so these texels
    # show a real if flat color instead of being flagged invalid.
    under_observed = is_observed & (support_count < config.min_observations_per_texel)
    if under_observed.any():
        k0_avg = Atb[under_observed, 0, :] / AtA[under_observed, 0, 0].unsqueeze(-1).clamp_min(1e-8)
        coeffs[under_observed, 0, :] = k0_avg
        coeffs[under_observed, 1:, :] = 0.0

    coeffs = coeffs.cpu().numpy().reshape(H, W, K, 3)
    observed = is_observed.cpu().numpy().reshape(H, W)
    valid = observed & atlas.valid

    rejected_low_precision = (support_count > 0) & ~well_conditioned
    n_gated = int(rejected_low_precision.sum().item())
    print(
        f"  precision gate (total >= {config.ridge * config.min_precision_ridge_ratio:g}): "
        f"{n_gated} ridge-dominated texels rejected, filled by dilation"
    )

    if debug_status_path is not None:
        save_texel_status_map(
            debug_status_path,
            atlas_valid=atlas.valid,
            well_fit=(is_observed & ~under_observed).cpu().numpy().reshape(H, W),
            under_observed=under_observed.cpu().numpy().reshape(H, W),
            rejected_low_precision=rejected_low_precision.cpu().numpy().reshape(H, W),
            never_observed=((support_count == 0)).cpu().numpy().reshape(H, W),
        )

    # Texture padding, matching baking.py's: the shaders sample u_coeffs with
    # GL_LINEAR gated on atlas.valid, so an unobserved-but-valid texel left at zero
    # would bleed black into its observed neighbours, worst at chart seams. Nearest-
    # copy (never average) so sampling near a seam blends two real fits. `valid`
    # itself stays keyed on real observations; only the backing data is padded.
    holes = atlas.valid & ~observed
    hole_index = np.zeros(0, dtype=np.int64)
    hole_source = np.zeros(0, dtype=np.int64)
    if observed.any() and holes.any():
        _, (iy, ix) = distance_transform_edt(~observed, return_indices=True)
        coeffs[holes] = coeffs[iy[holes], ix[holes]]
        hole_index = np.flatnonzero(holes)
        hole_source = (iy * W + ix).ravel()[hole_index]

    if fit_stats_path is not None:
        obs_idx = torch.nonzero(is_observed, as_tuple=True)[0]
        np.savez_compressed(
            fit_stats_path,
            observed_index=obs_idx.cpu().numpy().astype(np.int64),
            under_observed=under_observed[obs_idx].cpu().numpy(),
            AtA=AtA[obs_idx].cpu().numpy(),
            Atb=Atb[obs_idx].cpu().numpy(),
            obs_min=obs_min[obs_idx].cpu().numpy(),
            obs_max=obs_max[obs_idx].cpu().numpy(),
            hole_index=hole_index,
            hole_source=hole_source,
        )

    # The mean camera up rides along with the coefficients: the fit does not need it,
    # but it is the only record of which way was up. See SHTextureMap.up_vector.
    return SHTextureMap(
        data=coeffs.astype(np.float32),
        degree=config.sh_degree,
        valid=valid,
        up_vector=posed_images.mean_up_vector(images_dir),
    )

