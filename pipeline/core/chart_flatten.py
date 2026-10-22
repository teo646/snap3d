"""Flattening a disk of triangles into one fold-free UV chart.

xatlas decides for itself where to cut, and on a large curved region (a statue's face, 550 faces)
it cuts it into several charts whatever its options are. When a region must be one chart it is
flattened here instead:

1. **Tutte embedding.** The boundary is laid out on a circle, spaced by its 3D edge lengths, and
   every interior vertex is placed at the weighted mean of its neighbours (mean-value weights,
   which are positive). For a disk this is a guaranteed fold-free map - but a distorted one, as
   the whole boundary has been forced onto a circle.
2. **ARAP** (as-rigid-as-possible, Liu et al. 2008), local / global iterations. Each triangle is
   given the rotation that best fits its current UV edges to its true 3D shape; the UVs are then
   solved so every triangle is as close to that as the others allow. The boundary is free, so the
   chart relaxes towards its natural outline. Each step is shortened until no triangle flips, so
   the map stays fold-free from the Tutte start to the end.

The result is checked rather than trusted: no triangle may be flipped, and the triangles' areas
must add up to the area the boundary encloses (a fold-free map that overlapped itself would not).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

__all__ = ["flatten_disk"]


def _boundary_loop(faces: np.ndarray) -> list[int]:
    """The single boundary loop of a disk, in order."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(e, axis=1)
    uniq, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    on_boundary = counts[inv.ravel()] == 1
    nxt = {int(a): int(b) for a, b in e[on_boundary]}  # directed as the faces wind
    start = next(iter(nxt))
    loop, v = [start], nxt[start]
    while v != start:
        loop.append(v)
        v = nxt[v]
    if len(loop) != len(nxt):
        raise ValueError("the region has more than one boundary loop: not a disk")
    return loop


def _local_frames(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(F, 3, 2): each triangle's corners in its own plane, an isometric 2D copy of it."""
    p0, p1, p2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e1 = p1 - p0
    x1 = np.linalg.norm(e1, axis=1)
    ex = e1 / x1[:, None]
    e2 = p2 - p0
    x2 = np.einsum("ij,ij->i", e2, ex)
    y2 = np.linalg.norm(e2 - x2[:, None] * ex, axis=1)
    out = np.zeros((len(faces), 3, 2))
    out[:, 1, 0] = x1
    out[:, 2, 0] = x2
    out[:, 2, 1] = y2
    return out


def _signed_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _tutte(verts: np.ndarray, faces: np.ndarray, loop: list[int]) -> np.ndarray:
    n = len(verts)
    loop_arr = np.array(loop)
    seg = np.linalg.norm(verts[np.roll(loop_arr, -1)] - verts[loop_arr], axis=1)
    ang = 2 * np.pi * np.concatenate([[0.0], np.cumsum(seg)[:-1]]) / seg.sum()
    uv = np.zeros((n, 2))
    uv[loop_arr] = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    # mean-value weights of each directed edge (i -> j): (tan(a/2) + tan(b/2)) / |ij|, a and b the
    # angles at i of the two triangles sharing the edge
    rows, cols, vals = [], [], []
    for f in faces:
        for k in range(3):
            i, j, l = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
            vij, vil = verts[j] - verts[i], verts[l] - verts[i]
            ang_i = np.arccos(np.clip(vij @ vil / (np.linalg.norm(vij) * np.linalg.norm(vil)), -1, 1))
            rows += [i, i]
            cols += [j, l]
            vals += [np.tan(ang_i / 2) / np.linalg.norm(vij), np.tan(ang_i / 2) / np.linalg.norm(vil)]
    w = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    w = w + sp.csr_matrix((n, n))  # sums duplicates
    deg = np.asarray(w.sum(axis=1)).ravel()
    lap = sp.diags(deg) - w
    interior = np.setdiff1d(np.arange(n), loop_arr)
    if len(interior):
        a = lap[interior][:, interior].tocsc()
        b = -lap[interior][:, loop_arr] @ uv[loop_arr]
        uv[interior] = spl.splu(a).solve(b)
    return uv


def flatten_disk(vertices: np.ndarray, faces: np.ndarray, iterations: int = 200, tol: float = 1e-7) -> dict:
    """UVs for the disk `faces` (rows of indices into `vertices`).

    Returns `vertex_ids` (the vertices used, sorted), `faces` (re-indexed into them), `uv` (one per
    used vertex, oriented so the triangles are counter-clockwise as the input winds them, and
    scaled so that UV area equals surface area) and `report` (iterations, final energy, the
    spread of area stretch)."""
    used, compact = np.unique(faces, return_inverse=True)
    f = compact.reshape(-1, 3)
    v = np.asarray(vertices, dtype=np.float64)[used]
    loop = _boundary_loop(f)

    uv = _tutte(v, f, loop)
    if _signed_areas(uv, f).min() * _signed_areas(uv, f).max() < 0 and np.sign(_signed_areas(uv, f)).sum() < 0:
        uv[:, 0] *= -1.0
    if (_signed_areas(uv, f) <= 0).any():
        uv[:, 0] *= -1.0
    if (_signed_areas(uv, f) <= 0).any():
        raise ValueError("the Tutte start has flipped triangles: the region is not a clean disk")

    ref = _local_frames(v, f)  # (F, 3, 2)
    # cotangent weights of each triangle's three edges
    cot = np.zeros((len(f), 3))
    for k in range(3):
        p, q, r = v[f[:, k]], v[f[:, (k + 1) % 3]], v[f[:, (k + 2) % 3]]
        a, b = q - r, p - r  # the edge (k, k+1) is opposite corner k+2
        cross = np.linalg.norm(np.cross(a, b), axis=1)
        cot[:, k] = np.einsum("ij,ij->i", a, b) / np.maximum(cross, 1e-20)
    n = len(v)
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j = f[:, k], f[:, (k + 1) % 3]
        w = 0.5 * cot[:, k]
        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [-w, -w, w, w]
    lap = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    pin = int(loop[0])  # the Laplacian is singular up to a translation: fix one vertex
    free = np.setdiff1d(np.arange(n), [pin])
    solve = spl.splu(lap[free][:, free].tocsc()).solve

    energy_prev = np.inf
    for it in range(iterations):
        # local step: the best rotation of each triangle's reference shape onto its UV edges
        rhs = np.zeros((n, 2))
        energy = 0.0
        for k in range(3):
            i, j = f[:, k], f[:, (k + 1) % 3]
            e_ref = ref[:, (k + 1) % 3] - ref[:, k]
            e_uv = uv[j] - uv[i]
            cov = 0.5 * cot[:, k][:, None, None] * e_ref[:, :, None] * e_uv[:, None, :]
            if k == 0:
                s = cov
            else:
                s = s + cov
        # s = sum w * e_ref e_uv^T = U S V^T; the rotation taking e_ref to e_uv is R = V U^T
        u, sig, vt = np.linalg.svd(s)
        v_ = np.swapaxes(vt, 1, 2)
        rot = v_ @ np.swapaxes(u, 1, 2)
        flip = np.linalg.det(rot) < 0  # a reflection would flip the triangle: use the other sign of the weaker axis
        v_[flip, :, 1] *= -1.0
        rot = v_ @ np.swapaxes(u, 1, 2)  # (F, 2, 2) maps reference -> uv
        for k in range(3):
            i, j = f[:, k], f[:, (k + 1) % 3]
            e_ref = ref[:, (k + 1) % 3] - ref[:, k]
            target = np.einsum("fab,fb->fa", rot, e_ref)
            w = (0.5 * cot[:, k])[:, None]
            # d/du_i of w |(u_j - u_i) - t|^2 is -2w((u_j - u_i) - t): the system is L u = rhs with
            # the target edge pulling u_j forward and u_i back
            np.add.at(rhs, i, -w * target)
            np.add.at(rhs, j, w * target)
            d = (uv[j] - uv[i]) - target
            energy += float((0.5 * cot[:, k] * (d * d).sum(axis=1)).sum())
        # global step
        new = uv.copy()
        b = rhs[free] - lap[free][:, [pin]] @ uv[[pin]]
        new[free] = solve(b)
        # shorten the step until no triangle is flipped
        step = 1.0
        while (_signed_areas(uv + step * (new - uv), f) <= 0).any():
            step *= 0.5
            if step < 1e-6:
                break
        if step < 1e-6:
            break
        uv = uv + step * (new - uv)
        if abs(energy_prev - energy) < tol * max(energy, 1e-12):
            break
        energy_prev = energy

    areas = _signed_areas(uv, f)
    if (areas <= 0).any():
        raise ValueError("flattening left flipped triangles")
    # the area the boundary encloses must be what the triangles add up to
    bl = uv[np.array(loop)]
    enclosed = 0.5 * abs(np.sum(bl[:, 0] * np.roll(bl[:, 1], -1) - np.roll(bl[:, 0], -1) * bl[:, 1]))
    if abs(enclosed - areas.sum()) > 1e-6 * areas.sum():
        raise ValueError("the flattened chart overlaps itself")
    a3 = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    uv = uv * np.sqrt(a3.sum() / areas.sum())
    stretch = (areas * (a3.sum() / areas.sum())) / a3
    return {
        "vertex_ids": used, "faces": f, "uv": uv,
        "report": {"iterations": it + 1, "area_stretch_p5": float(np.percentile(stretch, 5)), "area_stretch_p95": float(np.percentile(stretch, 95))},
    }
