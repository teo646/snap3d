"""Turning a set of mesh faces into one that can be a single UV chart: a topological disk.

A chart that is to become one UV island must be a disk: connected, with no holes, and no
vertex where two parts of it touch at a point. A region picked by labels (a face found in
photos) is rarely that - a few faces are missed inside it (holes), a few stray ones hang off its
edge (pinches), and a stray piece sits apart from the rest. `make_disk` repairs those from
the surrounding mesh and says plainly when it cannot.

Faces are rows of vertex indices into one mesh; two faces are adjacent when they share an edge.
Vertices that a UV unwrap has duplicated along seams must be welded first (`weld`), or a seam
looks like a boundary.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = ["weld", "disk_report", "make_disk", "close_gaps"]


def weld(faces: np.ndarray, source_vertex_index: np.ndarray | None) -> np.ndarray:
    """`faces` with duplicated (seam-split) vertices mapped back to one index each."""
    return faces if source_vertex_index is None else np.asarray(source_vertex_index)[faces]


def _edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.sort(e, axis=1)


def _face_graph(faces: np.ndarray) -> list[list[int]]:
    """Per face, the faces sharing an edge with it."""
    owner: dict[tuple[int, int], list[int]] = defaultdict(list)
    for f, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            owner[(min(a, b), max(a, b))].append(f)
    adj: list[list[int]] = [[] for _ in range(len(faces))]
    for fs in owner.values():
        for i in fs:
            adj[i].extend(j for j in fs if j != i)
    return adj


def _components(adj: list[list[int]], members: np.ndarray) -> list[np.ndarray]:
    """Edge-connected components among the face indices `members`."""
    inside = np.zeros(len(adj), dtype=bool)
    inside[members] = True
    seen = np.zeros(len(adj), dtype=bool)
    out = []
    for s in members:
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            f = stack.pop()
            comp.append(f)
            for g in adj[f]:
                if inside[g] and not seen[g]:
                    seen[g] = True
                    stack.append(g)
        out.append(np.array(comp))
    return out


def _vertex_fans(faces: np.ndarray) -> dict[int, list[list[int]]]:
    """Per vertex, its incident faces grouped into fans (faces connected around the vertex
    through shared edges). More than one fan is a pinch point."""
    incident: dict[int, list[int]] = defaultdict(list)
    for f, tri in enumerate(faces):
        for v in tri:
            incident[int(v)].append(f)
    fans = {}
    for v, fs in incident.items():
        edges_of = {f: {int(u) for u in faces[f] if u != v} for f in fs}
        left, groups = set(fs), []
        while left:
            f0 = left.pop()
            group, stack = [f0], [f0]
            while stack:
                f = stack.pop()
                for g in list(left):
                    if edges_of[f] & edges_of[g]:
                        left.discard(g)
                        group.append(g)
                        stack.append(g)
            groups.append(group)
        fans[v] = groups
    return fans


def disk_report(faces: np.ndarray) -> dict:
    """Topology of the surface made of `faces`: `euler` (V - E + F; 1 for a disk), `boundary_loops`,
    `pinch_vertices`, `components`, `non_manifold_edges`."""
    faces = np.asarray(faces)
    e = _edges(faces)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    nxt: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        nxt[int(a)].append(int(b))
        nxt[int(b)].append(int(a))
    seen, loops = set(), 0
    for s in nxt:
        if s in seen:
            continue
        loops += 1
        stack = [s]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(nxt[x])
    return {
        "faces": len(faces),
        "euler": int(len(np.unique(faces)) - len(uniq) + len(faces)),
        "boundary_loops": loops,
        "pinch_vertices": sum(1 for g in _vertex_fans(faces).values() if len(g) > 1),
        "components": len(_components(_face_graph(faces), np.arange(len(faces)))),
        "non_manifold_edges": int((counts > 2).sum()),
    }


def close_gaps(adj: list[list[int]], selected: np.ndarray, rings: int) -> np.ndarray:
    """`selected` grown by `rings` faces in every direction and shrunk by as many: a thin gap
    (a strip of faces a vote missed along a seam, a face or two inside a region) fills in,
    while the region's outline stays where it was."""
    n = len(adj)
    keep = np.zeros(n, dtype=bool)
    keep[np.asarray(selected)] = True

    def grow(mask: np.ndarray) -> np.ndarray:
        out = mask.copy()
        for f in np.nonzero(mask)[0]:
            out[adj[f]] = True
        return out

    grown = keep
    for _ in range(rings):
        grown = grow(grown)
    shrunk = grown
    for _ in range(rings):
        shrunk = ~grow(~shrunk)
    return np.nonzero(shrunk | keep)[0]


def _cut_ring(all_faces: np.ndarray, keep: np.ndarray, adj: list[list[int]]) -> bool:
    """If the kept faces have more than one boundary loop (the region goes around a handle of the
    mesh - a finger touching the cheek), drop the shortest chain of kept faces from one loop to
    another, which joins them into one: a slit. One call makes one cut; `make_disk` repeats it
    until a single loop is left. Returns whether it cut."""
    idx = np.nonzero(keep)[0]
    tri = all_faces[idx]
    count: dict[tuple[int, int], int] = defaultdict(int)
    for e in map(tuple, _edges(tri)):
        count[e] += 1
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    boundary = [e for e, c in count.items() if c == 1]
    for a, b in boundary:
        parent[find(a)] = find(b)
    loops = {find(a) for a, _ in boundary}
    if len(loops) < 2:
        return False
    loop_of = {v: find(v) for e in boundary for v in e}
    first, *others = sorted(loops)
    on = {k: [] for k in loops}
    for k, t in enumerate(tri):
        # by a boundary EDGE, not a corner: the chain must end up edge-adjacent to the rest of the
        # mesh, or `make_disk` sees the gap it leaves as an enclosed pocket and fills it again
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if count[(min(a, b), max(a, b))] == 1:
                on[loop_of[a]].append(idx[k])
                break
    goal = {f for k in others for f in on[k]}
    prev = {f: -1 for f in on[first]}
    queue = list(on[first])
    for f in queue:
        if f in goal:
            while f != -1:
                keep[f] = False
                f = prev[f]
            return True
        for g in adj[f]:
            if keep[g] and g not in prev:
                prev[g] = f
                queue.append(g)
    return False


def make_disk(all_faces: np.ndarray, selected: np.ndarray, max_rounds: int = 50, close_rings: int = 2) -> np.ndarray:
    """Indices into `all_faces` of a disk close to `selected`.

    First `close_gaps` bridges thin gaps of up to `2 * close_rings` faces. Then, repeating until
    the topology is right: keep the largest connected piece; add each small
    pocket of mesh surrounded by it (a hole); drop the faces of the smaller fan at any pinch
    vertex. A ring (the region goes round a handle of the mesh) is cut open along its shortest chain of faces. Raises ValueError if it does not converge to a single disk with one boundary loop.

    `all_faces` is the whole mesh's welded faces, which is where holes are filled from -
    the faces that were missed are in the mesh, they are just not selected."""
    adj = _face_graph(all_faces)
    keep = np.zeros(len(all_faces), dtype=bool)
    if len(selected) == 0:
        raise ValueError("no faces selected")
    keep[close_gaps(adj, selected, close_rings) if close_rings else np.asarray(selected)] = True
    for _ in range(max_rounds):
        pieces = _components(adj, np.nonzero(keep)[0])
        if not pieces:
            raise ValueError("no faces selected")
        biggest = max(pieces, key=len)
        keep[:] = False
        keep[biggest] = True
        # pockets: groups of unselected faces that do not reach the rest of the mesh without
        # crossing the selection. The mesh outside the region is one huge group; holes are small.
        outside = _components(adj, np.nonzero(~keep)[0])
        if len(outside) > 1:
            outside.sort(key=len, reverse=True)  # the mesh beyond the region is by far the largest
            keep[np.concatenate(outside[1:])] = True
        faces = all_faces[keep]
        fans = _vertex_fans(faces)
        pinched = [g for g in fans.values() if len(g) > 1]
        if pinched:
            idx = np.nonzero(keep)[0]
            for groups in pinched:
                for g in sorted(groups, key=len)[:-1]:
                    keep[idx[g]] = False
            continue
        rep = disk_report(all_faces[keep])
        if rep["euler"] <= 0 and rep["boundary_loops"] >= 2 and _cut_ring(all_faces, keep, adj):
            continue
        if rep["euler"] == 1 and rep["boundary_loops"] == 1 and rep["components"] == 1:
            return np.nonzero(keep)[0]
    raise ValueError(f"could not make a disk out of the selection: {disk_report(all_faces[keep])}")
