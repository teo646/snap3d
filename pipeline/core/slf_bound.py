"""Keep a fitted SH texture map inside the colours each texel was actually seen with.

The slf fit is least squares and extrapolates beyond the observed view directions, which shows
as glowing or black texels at unseen angles. This re-fits each texel, per channel, so its
function's range lies in [min observed - tol, max observed + tol] clamped to [0, 1], from the
normal equations and observed range slf saved (slf_fit.run_slf_fit's fit_stats_path).

  * Degree 1 (default): the range is exactly [a - |b|, a + |b|]; see _bound_degree1_range.
  * Texels slf flattened to a degree-0 average, and degree 0 maps: clamp a.
  * Degree 2 and up: rejected.

Coefficient padding is redone from the bounded coefficients using slf's recorded source map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pipeline.schemas.texture_maps import SHTextureMap

__all__ = ["SLFBoundConfig", "bound_sh_texture_map"]

_C0 = 0.28209479177387814
_C1 = 0.4886025119029199


@dataclass
class SLFBoundConfig:
    # How far outside its observed colours a texel's function may reach, in [0, 1] colour units.
    range_tolerance: float = 0.02
    device: str = "cuda"


def _bound_degree1_range(
    AtA: torch.Tensor, Atb: torch.Tensor, coeffs: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Re-fit degree-1 SH coefficients so each texel's function stays inside [lo, hi].

    AtA (N, 4, 4) with the ridge on its diagonal, Atb (N, 4, 3), coeffs (N, 4, 3) the
    unconstrained solve, lo/hi (N, 3). Returns (coeffs, texel-channels changed).

    f(d) = a + b.d' with a = C0*c0, b = C1*(-c3, -c1, c2), so the range is [a - |b|, a + |b|].
    The direction of b is held at the unconstrained fit's and only the DC and |b| are re-solved:
    a 2-variable quadratic over the triangle beta >= 0, C0*alpha + C1*beta <= hi,
    C0*alpha - C1*beta >= lo, minimised at the interior optimum or the best edge minimum.
    Texels already in range are unchanged.
    """
    C0, C1 = _C0, _C1
    out = coeffs.clone()
    n_changed = 0
    for ch in range(3):
        c = coeffs[:, :, ch]
        g = Atb[:, :, ch]
        v = c[:, 1:]
        v_norm = v.norm(dim=1)
        a = C0 * c[:, 0]
        r = C1 * v_norm
        lo_c, hi_c = lo[:, ch], hi[:, ch]
        needs = (a + r > hi_c + 1e-6) | (a - r < lo_c - 1e-6)
        if not needs.any():
            continue
        idx = torch.nonzero(needs, as_tuple=True)[0]
        M, gi, vi, vn = AtA[idx], g[idx], v[idx], v_norm[idx]
        lo_i, hi_i = lo_c[idx], hi_c[idx]
        # lo crossed hi: take the midpoint, flat.
        hi_i = torch.maximum(hi_i, lo_i)

        w = torch.zeros(len(idx), 4, device=c.device, dtype=c.dtype)
        tiny = vn < 1e-12
        w[:, 1:] = vi / vn.clamp_min(1e-12).unsqueeze(1)
        w[tiny, 1:] = torch.tensor([0.0, 1.0, 0.0], device=c.device, dtype=c.dtype)
        e0 = torch.zeros_like(w)
        e0[:, 0] = 1.0
        basis = torch.stack([e0, w], dim=2)  # (n, 4, 2)
        Hm = basis.transpose(1, 2) @ M @ basis  # (n, 2, 2)
        h = (basis.transpose(1, 2) @ gi.unsqueeze(2)).squeeze(2)  # (n, 2)

        def objective(p: torch.Tensor) -> torch.Tensor:
            return (p.unsqueeze(1) @ Hm @ p.unsqueeze(2)).squeeze(-1).squeeze(-1) - 2.0 * (p * h).sum(dim=1)

        def feasible(p: torch.Tensor) -> torch.Tensor:
            fa, fr = C0 * p[:, 0], C1 * p[:, 1]
            eps = 1e-6
            return (fr >= -eps) & (fa + fr <= hi_i + eps) & (fa - fr >= lo_i - eps)

        # Triangle corners in (alpha, beta).
        v1 = torch.stack([lo_i / C0, torch.zeros_like(lo_i)], dim=1)
        v2 = torch.stack([hi_i / C0, torch.zeros_like(hi_i)], dim=1)
        v3 = torch.stack([(lo_i + hi_i) / (2 * C0), (hi_i - lo_i) / (2 * C1)], dim=1)

        def edge_min(p0: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
            d = p1 - p0
            dHd = (d.unsqueeze(1) @ Hm @ d.unsqueeze(2)).squeeze(-1).squeeze(-1)
            grad = h - (Hm @ p0.unsqueeze(2)).squeeze(2)
            t = ((grad * d).sum(dim=1) / dHd.clamp_min(1e-20)).clamp(0.0, 1.0)
            return p0 + t.unsqueeze(1) * d

        candidates = [edge_min(v1, v2), edge_min(v2, v3), edge_min(v1, v3)]
        interior = torch.linalg.solve(Hm, h.unsqueeze(2)).squeeze(2)
        best, best_obj = None, None
        for cand in candidates + [interior]:
            obj = objective(cand)
            obj = torch.where(feasible(cand), obj, torch.full_like(obj, float("inf")))
            if best is None:
                best, best_obj = cand.clone(), obj
            else:
                take = obj < best_obj
                best[take], best_obj[take] = cand[take], obj[take]

        out[idx, :, ch] = best[:, 0:1] * e0 + best[:, 1:2] * w
        n_changed += len(idx)
    return out, n_changed


def bound_sh_texture_map(
    sh_map: SHTextureMap, fit_stats_path: Path, config: SLFBoundConfig | None = None
) -> tuple[SHTextureMap, str]:
    """Returns the bounded map and a one-line report. `sh_map` is left untouched."""
    config = config or SLFBoundConfig()
    if sh_map.degree >= 2:
        raise ValueError(
            f"slf_bound bounds the range in closed form, which exists for SH degree 0-1; got {sh_map.degree}"
        )
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    stats = np.load(fit_stats_path)
    H, W, K, _ = sh_map.data.shape
    flat = torch.from_numpy(sh_map.data.reshape(H * W, K, 3).copy()).to(device)
    obs = torch.from_numpy(stats["observed_index"]).to(device)
    under = torch.from_numpy(stats["under_observed"]).to(device)
    tol = config.range_tolerance
    lo = (torch.from_numpy(stats["obs_min"]).to(device) - tol).clamp(0.0, 1.0)
    hi = (torch.from_numpy(stats["obs_max"]).to(device) + tol).clamp(0.0, 1.0)
    hi = torch.maximum(hi, lo)

    coeffs = flat[obs]
    n_fit = 0
    if sh_map.degree == 1:
        fit = ~under
        if fit.any():
            AtA = torch.from_numpy(stats["AtA"]).to(device)[fit]
            Atb = torch.from_numpy(stats["Atb"]).to(device)[fit]
            coeffs[fit], n_fit = _bound_degree1_range(AtA, Atb, coeffs[fit], lo[fit], hi[fit])
        flat_rows = under
    else:
        flat_rows = torch.ones_like(under)
    # Constant functions: clamp a = C0 * c0.
    a = _C0 * coeffs[flat_rows, 0, :]
    a_clamped = torch.minimum(torch.maximum(a, lo[flat_rows]), hi[flat_rows])
    n_flat = int((a_clamped != a).sum())
    coeffs[flat_rows, 0, :] = a_clamped / _C0
    flat[obs] = coeffs

    hole_index = torch.from_numpy(stats["hole_index"]).to(device)
    if len(hole_index):
        flat[hole_index] = flat[torch.from_numpy(stats["hole_source"]).to(device)]

    bounded = SHTextureMap(
        data=flat.cpu().numpy().reshape(H, W, K, 3).astype(np.float32),
        degree=sh_map.degree, valid=sh_map.valid, up_vector=sh_map.up_vector,
    )
    n_rows = int(obs.numel())
    report = (
        f"range bound (observed +- {tol:g}): {n_fit} of {3 * int((~under).sum()) if sh_map.degree == 1 else 0} "
        f"fitted texel-channels re-fit, {n_flat} of {3 * int(flat_rows.sum())} flat ones clamped "
        f"({n_rows} observed texels)"
    )
    return bounded, report
