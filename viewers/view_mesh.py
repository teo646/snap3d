#!/usr/bin/env python
"""Interactive viewer for the pipeline's .obj meshes, wrapping open3d's built-in
visualizer (already a base dependency). Reads via o3d.io.read_triangle_mesh, so a
UV-textured mesh shows its texture and a vertex-coloured one its colours.

Usage:
    python viewers/view_mesh.py data/runs/apple_test/06_uv_unwrap/mesh_uv.obj

    # several stages side by side, offset along X:
    python viewers/view_mesh.py \
        data/runs/apple_test/03_surface_recon/mesh_highpoly.obj \
        data/runs/apple_test/05_decimation/mesh_lowpoly.obj --compare

    # headlamp Lambertian shading, to inspect surface detail without texture:
    python viewers/view_mesh.py data/runs/apple_test/06_uv_unwrap/mesh_uv.obj --shade

Hold E and move the mouse to erase vertices under the cursor; press B to undo, up to
10 erase steps back (--erase-key/--undo-key/--undo-depth to change any of that). Pass
--save-edit to overwrite each source .obj with its erased mesh on quit (skips files
nothing was erased from; keeps a one-time .bak of each file it does overwrite).
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same ambient/diffuse split as the SLF viewer's headlamp shading.
_HEADLAMP_AMBIENT = 0.15
_HEADLAMP_DIFFUSE = 0.85


def _apply_headlamp_shading(vis: o3d.visualization.Visualizer, meshes: list[o3d.geometry.TriangleMesh]) -> None:
    """Registers a per-frame callback that recolors each mesh by simple Lambertian
    shading with the light attached to the camera (color = ambient + diffuse *
    max(0, dot(normal, view_dir)), view_dir recomputed from the live camera position
    every frame) - a "headlamp": whatever currently faces the viewer is lit,
    independent of world/mesh orientation. open3d's plain Visualizer has no custom-
    shader hook, so this is per-vertex rather than per-fragment like the SLF viewer's
    GLSL version, but it's the same shading model.
    """
    normals = [np.asarray(m.vertex_normals) for m in meshes]
    verts = [np.asarray(m.vertices) for m in meshes]

    def _update(vis: o3d.visualization.Visualizer) -> bool:
        extrinsic = vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic
        R, t = extrinsic[:3, :3], extrinsic[:3, 3]
        cam_pos = -R.T @ t  # world-space camera position, from the world->camera extrinsic
        for mesh, n, v in zip(meshes, normals, verts):
            view_dir = cam_pos[None, :] - v
            view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-10
            ndotv = np.clip(np.einsum("ij,ij->i", n, view_dir), 0.0, None)
            intensity = _HEADLAMP_AMBIENT + _HEADLAMP_DIFFUSE * ndotv
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(intensity[:, None], 3, axis=1))
            vis.update_geometry(mesh)
        return False

    vis.register_animation_callback(_update)


def _restore_mesh(mesh: o3d.geometry.TriangleMesh, snapshot: o3d.geometry.TriangleMesh) -> None:
    """Overwrites every field `remove_vertices_by_mask` could have touched with
    `snapshot`'s (a `copy.deepcopy` taken before that mutation), in place - `mesh`
    stays the same object, so it's still the one `meshes`, the visualizer, and
    `_apply_headlamp_shading`'s closure all hold a reference to."""
    mesh.vertices = snapshot.vertices
    mesh.triangles = snapshot.triangles
    mesh.vertex_normals = snapshot.vertex_normals
    mesh.vertex_colors = snapshot.vertex_colors
    mesh.triangle_uvs = snapshot.triangle_uvs
    mesh.triangle_material_ids = snapshot.triangle_material_ids
    mesh.textures = snapshot.textures


def _register_vertex_eraser(
    vis: o3d.visualization.VisualizerWithKeyCallback,
    meshes: list[o3d.geometry.TriangleMesh],
    key_code: int,
    radius_px: float,
    undo_key_code: int,
    undo_depth: int,
) -> None:
    """While `key_code` is held, mouse movement erases vertices under the cursor
    (screen-space brush of radius `radius_px`). Only front-facing vertices are
    removed (dot(normal, view_dir) > 0.1) so a drag doesn't reach through the mesh
    to the far side. remove_vertices_by_mask also drops any triangle that
    referenced a removed vertex and keeps colors/normals/UVs in sync.

    `undo_key_code` pops the last erase and restores it, up to `undo_depth` steps
    back. One "step" is one mouse-move callback that erased anything - across
    however many of `meshes` it touched - so undo reverts them together, matching
    what a single drag increment did.

    open3d's built-in left-drag-rotate is wired through the same MouseMoveCallback
    this registers a Python handler for, and registering one *replaces* the default
    rather than layering on top of it - so left-drag orbit and Ctrl+left-drag /
    right-drag pan stop working the moment any mouse-move callback is registered at
    all, `key_code` held or not. `_on_mouse_button`/the non-erase branch of
    `_on_mouse_move` below reimplement both with `ViewControl.rotate`/`.translate`,
    the same calls open3d's own default handler makes, so dragging feels identical
    to a plain (uninstrumented) Visualizer except while `key_code` is held.
    """
    state = {"erase_key_down": False}
    history: deque[list[tuple[int, o3d.geometry.TriangleMesh]]] = deque(maxlen=undo_depth)
    # Left-button drag without the erase key held orbits; Ctrl+left or plain right
    # pans - open3d's own default bindings. `fresh` skips the first delta after a
    # press (nothing to diff against yet) and is re-armed when the erase key is
    # released mid-drag, so a drag interrupted by erasing doesn't jump on resume.
    _MOUSE_LEFT, _MOUSE_RIGHT, _MOD_CONTROL = 0, 1, 0x0002
    drag = {"button": None, "pan": False, "fresh": False, "last": (0.0, 0.0)}

    def _on_key(vis: o3d.visualization.Visualizer, action: int, mods: int) -> bool:
        state["erase_key_down"] = action != 0  # GLFW_RELEASE == 0
        if not state["erase_key_down"]:
            for mesh in meshes:
                if len(mesh.vertices) > 0:
                    mesh.compute_vertex_normals()
                    vis.update_geometry(mesh)
            drag["fresh"] = True
        return False

    def _on_mouse_button(vis: o3d.visualization.Visualizer, button: int, action: int, mods: int) -> bool:
        if button not in (_MOUSE_LEFT, _MOUSE_RIGHT):
            return False
        if action == 0:  # GLFW_RELEASE
            if drag["button"] == button:
                drag["button"] = None
            return False
        drag["button"] = button
        drag["pan"] = button == _MOUSE_RIGHT or bool(mods & _MOD_CONTROL)
        drag["fresh"] = True
        return False

    def _on_mouse_move(vis: o3d.visualization.Visualizer, x: float, y: float) -> bool:
        if not state["erase_key_down"]:
            if drag["button"] is None:
                return False
            if drag["fresh"]:
                drag["last"] = (x, y)
                drag["fresh"] = False
                return False
            dx, dy = x - drag["last"][0], y - drag["last"][1]
            drag["last"] = (x, y)
            ctrl = vis.get_view_control()
            ctrl.translate(dx, dy) if drag["pan"] else ctrl.rotate(dx, dy)
            return False
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        intr = params.intrinsic.intrinsic_matrix
        extr = params.extrinsic
        R, t = extr[:3, :3], extr[:3, 3]
        cam_pos = -R.T @ t
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        step: list[tuple[int, o3d.geometry.TriangleMesh]] = []
        for i, mesh in enumerate(meshes):
            v = np.asarray(mesh.vertices)
            if len(v) == 0:
                continue
            n = np.asarray(mesh.vertex_normals)
            cam_space = (R @ v.T).T + t
            z = cam_space[:, 2]
            in_front = z > 1e-6
            z_safe = np.where(in_front, z, 1.0)
            sx = fx * cam_space[:, 0] / z_safe + cx
            sy = fy * cam_space[:, 1] / z_safe + cy
            within_radius = (sx - x) ** 2 + (sy - y) ** 2 <= radius_px**2
            view_dir = cam_pos[None, :] - v
            view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-10
            facing = np.einsum("ij,ij->i", n, view_dir) > 0.1
            mask = in_front & within_radius & facing
            if mask.any():
                step.append((i, copy.deepcopy(mesh)))  # pre-mutation snapshot, for undo
                mesh.remove_vertices_by_mask(mask)
                vis.update_geometry(mesh)
        if step:
            history.append(step)
            vis.update_renderer()
        return False

    def _on_undo(vis: o3d.visualization.Visualizer, action: int, mods: int) -> bool:
        if action == 0:  # GLFW_RELEASE - undo on press/repeat only
            return False
        if not history:
            print("nothing to undo")
            return False
        step = history.pop()
        for i, snapshot in step:
            _restore_mesh(meshes[i], snapshot)
            vis.update_geometry(meshes[i])
        vis.update_renderer()
        print(f"undid {len(step)} mesh edit(s) ({len(history)} more undo step(s) available)")
        return False

    vis.register_key_action_callback(key_code, _on_key)
    vis.register_mouse_button_callback(_on_mouse_button)
    vis.register_mouse_move_callback(_on_mouse_move)
    vis.register_key_action_callback(undo_key_code, _on_undo)


# Distinct fallback colors applied to each mesh under --compare, since
# side-by-side comparison is only legible if the meshes don't all render the
# same flat gray.
_COMPARE_COLORS = [
    (0.85, 0.15, 0.15),  # red
    (0.15, 0.55, 0.85),  # blue
    (0.20, 0.75, 0.30),  # green
    (0.90, 0.65, 0.10),  # orange
]


def _load(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.vertices) == 0:
        raise ValueError(f"{path} contains no vertices (wrong file / empty stage output?)")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="one or more .obj mesh files")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="offset each mesh along X by its bounding-box width and flat-color each one "
        "distinctly (drops any UV texture), so multiple pipeline stages can be viewed side by side",
    )
    parser.add_argument("--wireframe", action="store_true", help="start in wireframe mode")
    parser.add_argument("--background", choices=["white", "black"], default="white")
    parser.add_argument(
        "--erase-key",
        default="E",
        help="single letter key: hold it and move the mouse to erase vertices under the "
        "cursor (default: %(default)s)",
    )
    parser.add_argument(
        "--erase-radius",
        type=float,
        default=14.0,
        help="brush radius in screen pixels for --erase-key (default: %(default)s)",
    )
    parser.add_argument(
        "--undo-key",
        default="B",
        help="single letter key: undoes the last erase step, up to --undo-depth times (default: %(default)s)",
    )
    parser.add_argument(
        "--undo-depth",
        type=int,
        default=10,
        help="how many erase steps back --undo-key can undo (default: %(default)s)",
    )
    parser.add_argument(
        "--shade",
        action="store_true",
        help="simple headlamp Lambertian shading (light follows the camera) instead of open3d's "
        "default lighting/texture/vertex-colors - combinable with --compare",
    )
    parser.add_argument(
        "--save-edit",
        action="store_true",
        help="on quit, overwrite each source .obj with its erased mesh (keeps a one-time "
        "<name>.obj.bak of the pre-edit file; a mesh nothing was erased from is left alone)",
    )
    args = parser.parse_args()

    for path in args.paths:
        if not path.exists():
            parser.error(f"file not found: {path}")
    if len(args.erase_key) != 1 or not args.erase_key.isalpha():
        parser.error("--erase-key must be a single letter")
    if len(args.undo_key) != 1 or not args.undo_key.isalpha():
        parser.error("--undo-key must be a single letter")
    if args.undo_key.upper() == args.erase_key.upper():
        parser.error("--undo-key and --erase-key must differ")
    if args.undo_depth < 1:
        parser.error("--undo-depth must be >= 1")
    if args.save_edit and args.compare:
        parser.error("--save-edit can't be used with --compare: that offsets and flat-colors "
                      "every mesh in memory, which isn't what should land back in the file")
    if args.save_edit and args.shade:
        parser.error("--save-edit can't be used with --shade: that overwrites vertex_colors with "
                      "camera-relative shading every frame, which isn't the mesh's real color")

    meshes = [_load(path) for path in args.paths]
    # Baseline to detect which meshes an erase actually touched (an equal-count mesh is
    # exactly the loaded one - undo restores original fields, so an erase fully undone
    # is indistinguishable from never erased, and correctly costs no write below).
    original_counts = [(len(m.vertices), len(m.triangles)) for m in meshes]

    if args.compare:
        cursor_x = 0.0
        for i, mesh in enumerate(meshes):
            mesh.paint_uniform_color(_COMPARE_COLORS[i % len(_COMPARE_COLORS)])
            bbox = mesh.get_axis_aligned_bounding_box()
            width = bbox.get_extent()[0]
            mesh.translate((cursor_x - bbox.min_bound[0], 0.0, 0.0))
            cursor_x += width * 1.15 + 1e-6
        for path, mesh in zip(args.paths, meshes):
            print(f"{path}: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris (offset for comparison)")
    else:
        for path, mesh in zip(args.paths, meshes):
            tag = "textured" if mesh.has_textures() else ("vertex-colored" if mesh.has_vertex_colors() else "shaded")
            print(f"{path}: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris ({tag})")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="; ".join(p.name for p in args.paths))
    for mesh in meshes:
        vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0] if args.background == "black" else [1.0, 1.0, 1.0])
    opt.mesh_show_wireframe = args.wireframe
    opt.mesh_show_back_face = True

    if args.shade:
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color  # else a texture would win over our colors
        _apply_headlamp_shading(vis, meshes)

    _register_vertex_eraser(
        vis, meshes, ord(args.erase_key.upper()), args.erase_radius,
        ord(args.undo_key.upper()), args.undo_depth,
    )
    print(f"Hold '{args.erase_key.upper()}' and move the mouse to erase vertices under the cursor.")
    print(f"Press '{args.undo_key.upper()}' to undo, up to {args.undo_depth} step(s) back.")

    vis.run()
    vis.destroy_window()

    if args.save_edit:
        saved = False
        for path, mesh, (orig_v, orig_t) in zip(args.paths, meshes, original_counts):
            if (len(mesh.vertices), len(mesh.triangles)) == (orig_v, orig_t):
                continue
            backup_path = path.with_name(path.name + ".bak")
            if not backup_path.exists():
                shutil.copyfile(path, backup_path)
            o3d.io.write_triangle_mesh(str(path), mesh)
            print(f"saved: {path}  ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris; "
                  f"original backed up to {backup_path})")
            saved = True
        if not saved:
            print("--save-edit given but nothing was erased; files left untouched")


if __name__ == "__main__":
    main()
