"""Real spherical-harmonics basis + CG bilinear-texel reconstruction.

Ported from teo646/surface-light-fields-for-planar-paintings @ 53f5787 (model.py's
`_sh_basis`, util.py's `bilinear_neighbors`/`cg_solve_bilinear_texels`) - the numerical
core of that commit's stabilized POM-based fit. See slf_fit.py for how they compose.

`hsh_remap` is the reference's hemisphere-to-sphere remap. `view_local` is already
relative to the texel's own normal, so remapping onto the full sphere before fitting
lets a low degree (1 by default) capture more single-sided detail per coefficient. It
clamps z just below zero to the hemisphere boundary rather than discarding it: with a
baked, decimated normal and a capture that circles the object, grazing observations
land there without being real cross-hemisphere ones.
"""

from __future__ import annotations

import torch

__all__ = ["sh_basis", "hsh_remap", "bilinear_neighbors", "cg_solve_bilinear_texels"]

_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
_C3 = [
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
    -0.4570457994644658, 1.445305721320277, -0.5900435899266435,
]  # fmt: skip


def sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """dirs: (B, 3) unit vectors -> (B, (degree+1)**2) real SH basis values, degree 0-3."""
    if not 0 <= degree <= 3:
        raise ValueError(f"sh_basis only supports degree 0-3, got {degree}")
    B = dirs.shape[0]
    K = (degree + 1) ** 2
    Y = torch.zeros(B, K, device=dirs.device, dtype=dirs.dtype)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    Y[:, 0] = _C0
    if degree >= 1:
        Y[:, 1] = -_C1 * y
        Y[:, 2] = _C1 * z
        Y[:, 3] = -_C1 * x
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        Y[:, 4] = _C2[0] * xy
        Y[:, 5] = _C2[1] * yz
        Y[:, 6] = _C2[2] * (2 * zz - xx - yy)
        Y[:, 7] = _C2[3] * xz
        Y[:, 8] = _C2[4] * (xx - yy)
        if degree >= 3:
            Y[:, 9] = _C3[0] * y * (3 * xx - yy)
            Y[:, 10] = _C3[1] * xy * z
            Y[:, 11] = _C3[2] * y * (4 * zz - xx - yy)
            Y[:, 12] = _C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
            Y[:, 13] = _C3[4] * x * (4 * zz - xx - yy)
            Y[:, 14] = _C3[5] * z * (xx - yy)
            Y[:, 15] = _C3[6] * x * (xx - 3 * yy)
    return Y


def hsh_remap(dirs: torch.Tensor) -> torch.Tensor:
    """(B, 3) unit vectors relative to the normal (z = normal) -> (B, 3) on the full
    sphere: z clamped to [0, 1], then `d' = (2xz, 2yz, 2z^2-1)`, which is
    unit-preserving. Feed the result to `sh_basis`, not `dirs`."""
    x, y = dirs[:, 0], dirs[:, 1]
    z = dirs[:, 2].clamp(0.0, 1.0)
    return torch.stack([2 * x * z, 2 * y * z, 2 * z * z - 1], dim=-1)


def bilinear_neighbors(
    x: torch.Tensor, y: torch.Tensor, width: int, height: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuous atlas coordinates (x, y) - uv_geometry.uv_to_atlas_xy's output, NOT
    normalized uv - to the 4 neighbour texel indices and bilinear weights. Same
    convention as sample_bilinear/rasterize_uv_atlas, since this decides which texels a
    photo pixel scatters onto during CG."""
    x0f = torch.floor(x)
    y0f = torch.floor(y)
    fx = x - x0f
    fy = y - y0f
    x0 = x0f.long().clamp(0, width - 1)
    x1 = (x0f + 1).long().clamp(0, width - 1)
    y0 = y0f.long().clamp(0, height - 1)
    y1 = (y0f + 1).long().clamp(0, height - 1)
    return x0, x1, y0, y1, fx, fy


RECON_RIDGE: float = 1e-3
RECON_CG_ITERS: int = 50
RECON_CG_TOL: float = 1e-5


def cg_solve_bilinear_texels(
    idx: torch.Tensor,  # (N, 4) int64 - flat texel index of the 4 bilinear neighbors per observation
    wts: torch.Tensor,  # (N, 4) float32 - bilinear weights for those 4 texels (rows sum to 1)
    b: torch.Tensor,  # (N, C) float32 - observed pixel color per observation
    P: int,  # number of texels (unknowns)
    ridge: float = RECON_RIDGE,
    iters: int = RECON_CG_ITERS,
    tol: float = RECON_CG_TOL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solves (B^T B + ridge*I) u = B^T b by matrix-free CG, where B[n, p] is the
    bilinear weight observation n gives texel p. Recovers the texel values that, when
    bilinearly re-sampled at each observation's coordinate, best reproduce every
    observed pixel - so a texel many photo pixels partially cover is one shared
    unknown rather than each pixel snapping to its nearest texel.

    Returns (u (P, C), precision (P,) = (B^T B)_pp), the latter an inverse-variance
    confidence used to discard barely-observed texels and to weight this photo in the
    SH regression.
    """

    def apply_B(v: torch.Tensor) -> torch.Tensor:
        return (wts.unsqueeze(-1) * v[idx]).sum(dim=1)

    def apply_Bt(r: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(P, r.shape[1], device=r.device, dtype=r.dtype)
        contrib = (wts.unsqueeze(-1) * r.unsqueeze(1)).reshape(-1, r.shape[1])
        out.index_add_(0, idx.reshape(-1), contrib)
        return out

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return apply_Bt(apply_B(v)) + ridge * v

    Atb = apply_Bt(b)
    precision = torch.zeros(P, device=b.device, dtype=b.dtype)
    precision.index_add_(0, idx.reshape(-1), (wts * wts).reshape(-1))

    x = torch.zeros_like(Atb)
    r = Atb - matvec(x)
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = matvec(p)
        denom = (p * Ap).sum(dim=0).clamp_min(1e-20)
        alpha = rs_old / denom
        x = x + alpha.unsqueeze(0) * p
        r = r - alpha.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        if torch.sqrt(rs_new.max()).item() < tol:
            break
        beta = rs_new / rs_old.clamp_min(1e-20)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    return x, precision
