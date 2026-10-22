#!/usr/bin/env python
"""Interactive viewer for a finished SLF run: orbits the low-poly mesh and shades each
fragment by evaluating its texel's fitted SH coefficients at the current view
direction - the same math slf_fit's AtA/Atb accumulation was fit against, so this is a
direct readout of the fit rather than a re-approximation.

Adapted from surface-light-fields-for-planar-paintings' render.py, including its
Parallax-Occlusion-Mapping. POM is needed here even though a real mesh is rendered: a
fragment's own uv is the *naive* low-poly address, while training resolved every
observation to the height-map-encoded true-surface uv before fitting, so rendering has
to repeat that resolve or it reads the fit at the wrong address. Only the first march
step uses the naive uv; everything after it uses the resolved one.

Usage:
    pip install -e ".[viewer]"   # moderngl + pygame, not installed by default
    python viewers/view_slf.py data/runs/apple_test
    python viewers/view_slf.py data/runs/apple_test --save-edit  # persist erased vertices on quit

Controls: left-drag orbit, wheel zoom, WASD/arrows pan, hold E + left-drag erase
vertices, ESC quit.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path

import moderngl
import numpy as np
import pygame

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling _common, however this file is launched

from _common import OrbitCamera, find_stage_dir, perspective  # noqa: E402
from pipeline.core.camera_raster import to_gl_texture  # noqa: E402
from pipeline.core.slf_shaders import EVAL_SH_GLSL, RESOLVE_GLSL, VERTEX_SHADER  # noqa: E402
from pipeline.core.uv_geometry import rasterize_uv_atlas, vertex_frame_attributes, vertex_normals_for  # noqa: E402
from pipeline.schemas.manifest import read_manifest  # noqa: E402
from pipeline.schemas.posed_images import PosedImageSet  # noqa: E402
from pipeline.schemas.texture_maps import HeightMap, SHTextureMap  # noqa: E402
from pipeline.schemas.uv_mesh import UVMesh  # noqa: E402

# ─── GLSL shaders ────────────────────────────────────────────────────────────
# VERTEX_SHADER and resolve_parallax_uv come from pipeline/core/slf_shaders.py - the
# same source the training pass compiles, not a separate port of it.

FRAGMENT_HEADER = "#version 330\n"

FRAGMENT_BODY = (
    RESOLVE_GLSL
    + EVAL_SH_GLSL
    + """
in vec2 v_uv;
in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
uniform sampler2D u_valid_tex;  // r = atlas.valid (uv_unwrap coverage), g = sh_map.valid (observation count)
uniform sampler2D u_true_pos_tex;  // atlas.positions + height_map*atlas.normals - matches slf_fit.py's true_pos
out vec4 f_color;

void main(){
    vec3 N = normalize(v_normal);
    vec2 resolved_uv = resolve_parallax_uv(v_uv, v_world_pos, N, v_tangent, v_bitangent);

    vec2 valid = texture(u_valid_tex, resolved_uv).rg;
    if (valid.r < 0.5){
        f_color = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 view_w = normalize(u_cam_pos - v_world_pos);
    vec3 T = normalize(v_tangent - N * dot(v_tangent, N));                                                                                                                                     
    vec3 B = cross(N, T);                                                                                                                                                                      
    vec3 view_l = vec3(dot(view_w, T), dot(view_w, B), dot(view_w, N));
    f_color = vec4(eval_sh(resolved_uv, view_l), 1.0);
}
"""
)


def _fs_header(degree: int, num_steps: int) -> str:
    K = (degree + 1) ** 2
    return FRAGMENT_HEADER + f"#define DEGREE {degree}\n#define K_COEFFS {K}\n#define NUM_STEPS {num_steps}\n"


# ─── data loading ────────────────────────────────────────────────────────────


def _uv_texture_resolution(uv_dir: Path) -> tuple[int, int]:
    manifest = read_manifest(uv_dir)
    if manifest is not None and "texture_resolution" in manifest.params:
        w, h = manifest.params["texture_resolution"]
        return (w, h)
    return (1024, 1024)


def load_slf_result(run_dir: Path):
    uv_dir = find_stage_dir(run_dir, "uv_unwrap")
    # The range-bounded map when the run has one - it is what export ships - else slf's.
    try:
        slf_dir = find_stage_dir(run_dir, "slf_bound")
    except FileNotFoundError:
        slf_dir = find_stage_dir(run_dir, "slf")

    uv_mesh = UVMesh.load(uv_dir / "mesh_uv.obj", texture_resolution=_uv_texture_resolution(uv_dir))
    height_map = HeightMap.load(slf_dir / "height_map.png")  # same baseline height baking.py produced, unchanged by slf
    sh_map = SHTextureMap.load(slf_dir / "sh_texture_map.npz")
    if sh_map.up_vector is None:
        # Run predates SHTextureMap.up_vector. Re-derive it from the same place slf
        # would have: the ingest stage's poses. Doing it here (rather than leaving it
        # None) means render-time code can rely on the axis existing for every run,
        # new or old, without re-running the fit.
        ingest_dir = find_stage_dir(run_dir, "ingest")
        sh_map.up_vector = PosedImageSet.load(ingest_dir).mean_up_vector(ingest_dir)

    # Neither the local frame nor xy_per_height is loaded at all any more: the shader
    # rebuilds both per fragment from the interpolated vertex attributes
    # (slf_shaders.uv_frame / xy_per_height). The atlas is still rasterized here for
    # `valid`, `positions` and the normals `true_pos` is built from.
    atlas = rasterize_uv_atlas(uv_mesh)
    height_range = height_map.march_range()

    # slf_fit.py's `true_pos`: the flat atlas position pushed along the flat normal by
    # the baked height. Training measured the SH view direction from this point, so
    # rendering has to sample it too (not the rasterized `v_world_pos`) for the
    # coefficients to be read at the angle they were fit against.
    true_pos_render = atlas.positions + height_map.data.astype(np.float64)[..., None] * atlas.normals

    return (
        uv_mesh,
        sh_map,
        atlas,
        true_pos_render,
        height_map,
        height_range,
    )


# ─── vertex eraser ───────────────────────────────────────────────────────────
# Hold E + left-drag deletes any vertex the cursor passes over, and every face touching
# it. This only edits the geometry buffer in memory (and on GPU): the uv/sh/height
# textures are untouched, so an erased patch just stops rasterizing and shows the clear
# color through the hole. Nothing reaches disk unless run with --save-edit, which
# overwrites uv_unwrap's mesh_uv.obj on quit (see run_viewer's save_if_edited) -
# without it, quitting discards the edit.


class VertexEraser:
    """Owns the mesh's live vertex/face/uv arrays and the GPU buffers built from them,
    and knows how to drop vertices and re-upload. `uv_mesh` is kept only as a template
    (for `to_trimesh`-adjacent metadata `vertex_frame_attributes` doesn't need per call);
    its own `.vertices`/`.faces`/`.uv` are never read after `__init__`."""

    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program, uv_mesh: UVMesh):
        self._ctx = ctx
        self._prog = prog
        self._template = dataclasses.replace(uv_mesh, vertex_normals=None, source_vertex_index=None)
        self.vertices = uv_mesh.vertices.astype(np.float64)
        self.faces = uv_mesh.faces.astype(np.int64)
        self.uv = uv_mesh.uv.astype(np.float32)
        self.source_vertex_index = (
            None if uv_mesh.source_vertex_index is None else uv_mesh.source_vertex_index.copy()
        )
        self.vao: moderngl.VertexArray | None = None
        self._vbo: moderngl.Buffer | None = None
        self._ibo: moderngl.Buffer | None = None
        self._rebuild()

    def _rebuild(self) -> None:
        mesh = dataclasses.replace(self._template, vertices=self.vertices, faces=self.faces, uv=self.uv)
        normals, tangent_raw, bitangent_raw = vertex_frame_attributes(mesh)
        verts = np.concatenate(
            [
                self.vertices.astype(np.float32),
                self.uv.astype(np.float32),
                normals.astype(np.float32),
                tangent_raw.astype(np.float32),
                bitangent_raw.astype(np.float32),
            ],
            axis=1,
        )
        if self._vbo is not None:
            self._vbo.release()
            self._ibo.release()
            self.vao.release()
        self._vbo = self._ctx.buffer(np.ascontiguousarray(verts).tobytes())
        self._ibo = self._ctx.buffer(np.ascontiguousarray(self.faces.astype(np.int32)).tobytes())
        self.vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, "3f 2f 3f 3f 3f", "in_pos", "in_uv", "in_normal", "in_tangent", "in_bitangent")],
            self._ibo,
        )

    def screen_positions(self, mvp: np.ndarray, width: int, height: int) -> np.ndarray:
        """Current vertices projected to (x, y) pixel coordinates, NaN where a vertex is
        behind the camera (w <= 0 - excluded from hit-testing, not clamped onto screen)."""
        verts_h = np.concatenate([self.vertices, np.ones((len(self.vertices), 1))], axis=1)
        clip = verts_h @ mvp.T.astype(np.float64)
        w = clip[:, 3]
        ndc = clip[:, :2] / np.where(np.abs(w) < 1e-12, 1e-12, w)[:, None]
        screen = np.stack([(ndc[:, 0] + 1.0) * 0.5 * width, (1.0 - ndc[:, 1]) * 0.5 * height], axis=1)
        screen[w <= 0] = np.nan
        return screen

    def erase_near(self, mvp: np.ndarray, width: int, height: int, cursor: tuple[int, int], radius_px: float) -> int:
        """Deletes every vertex within `radius_px` of `cursor` (and the faces touching
        any of them), rebuilds the GPU buffers, and returns how many vertices were
        removed. A no-op (0 returned, buffers untouched) if that would empty the mesh -
        an all-deleted mesh has nothing for `vao.render()` to draw."""
        if len(self.vertices) == 0:
            return 0
        screen = self.screen_positions(mvp, width, height)
        dist2 = np.sum((screen - np.asarray(cursor)) ** 2, axis=1)
        hit = np.where(dist2 <= radius_px * radius_px)[0]
        if len(hit) == 0:
            return 0
        keep = np.ones(len(self.vertices), dtype=bool)
        keep[hit] = False
        new_faces_mask = keep[self.faces].all(axis=1)
        if not new_faces_mask.any():
            return 0  # would erase the whole mesh - refuse rather than render nothing
        old_to_new = np.full(len(self.vertices), -1, dtype=np.int64)
        old_to_new[keep] = np.arange(keep.sum())

        self.vertices = self.vertices[keep]
        self.uv = self.uv[keep]
        if self.source_vertex_index is not None:
            self.source_vertex_index = self.source_vertex_index[keep]
        self.faces = old_to_new[self.faces[new_faces_mask]]
        self._rebuild()
        return int((~keep).sum())

    def to_uv_mesh(self) -> UVMesh:
        """A UVMesh snapshot of the current (possibly edited) geometry, vertex normals
        recomputed fresh - the loaded ones are indexed against the original vertex
        count and would silently misalign after any deletion."""
        mesh = dataclasses.replace(self._template, vertices=self.vertices, faces=self.faces, uv=self.uv)
        mesh.vertex_normals = vertex_normals_for(mesh)
        mesh.source_vertex_index = self.source_vertex_index
        return mesh


# ─── entry point ─────────────────────────────────────────────────────────────


def run_viewer(
    run_dir: Path, width: int, height: int, num_steps: int,
    erase_radius_px: float = 16.0, save_edit: bool = False,
) -> None:
    (
        uv_mesh,
        sh_map,
        atlas,
        true_pos_render,
        height_map,
        height_range,
    ) = load_slf_result(run_dir)
    uv_mesh_path = find_stage_dir(run_dir, "uv_unwrap") / "mesh_uv.obj"
    # The orbit turns about the capture's own up axis, not world +Y: see _common's
    # module docstring for why a reconstruction's world frame has no up of its own.
    capture_up = np.asarray(sh_map.up_vector, dtype=np.float64)
    print(
        f"Mesh: {len(uv_mesh.vertices)} verts, {len(uv_mesh.faces)} faces  |  "
        f"SH degree {sh_map.degree}  |  valid texels: {sh_map.valid.mean() * 100:.1f}%  |  "
        f"capture up: ({capture_up[0]:+.3f}, {capture_up[1]:+.3f}, {capture_up[2]:+.3f})"
    )

    centroid = uv_mesh.vertices.mean(axis=0)
    radius0 = float(np.linalg.norm(uv_mesh.vertices.max(axis=0) - uv_mesh.vertices.min(axis=0))) * 1.2
    camera = OrbitCamera(origin=centroid, radius=max(radius0, 1e-3), up=capture_up)

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption(
        "SLF viewer  [drag: orbit | wheel: zoom | WASD: pan | E+drag: erase vertices | ESC: quit]"
    )

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    prog = ctx.program(
        vertex_shader=VERTEX_SHADER, fragment_shader=_fs_header(sh_map.degree, num_steps) + FRAGMENT_BODY
    )

    tex_w, tex_h = uv_mesh.texture_resolution

    height_tex = to_gl_texture(ctx, height_map.data.astype(np.float32)[..., None])
    valid_rg = np.stack([atlas.valid, sh_map.valid], axis=-1).astype(np.float32)  # r=atlas.valid, g=sh_map.valid
    valid_tex = to_gl_texture(ctx, valid_rg)

    for unit, tex in enumerate(
        [
            height_tex,
            valid_tex,
        ]
    ):
        tex.use(unit)
    prog["u_height_tex"].value = 0
    prog["u_valid_tex"].value = 1
    prog["u_tex_size"].value = (float(tex_w), float(tex_h))
    prog["u_height_range"].value = float(height_range)
    prog["u_layer_step"].value = float(2.0 * height_range / num_steps)

    coeffs_flipped = np.ascontiguousarray(np.flipud(sh_map.data), dtype=np.float32)  # (H, W, K, 3)
    H, W, K, _ = coeffs_flipped.shape
    coeffs_layers = np.ascontiguousarray(np.transpose(coeffs_flipped, (2, 0, 1, 3)))  # (K, H, W, 3)
    coeff_tex = ctx.texture_array((W, H, K), 3, coeffs_layers.tobytes(), dtype="f4")
    coeff_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    coeff_tex.repeat_x = False
    coeff_tex.repeat_y = False
    coeff_tex.use(3)
    prog["u_coeffs"].value = 3

    # The four frame textures this used to upload are gone; these three attributes
    # replace them, and must be the same per-vertex arrays uv_geometry interpolated
    # into the atlas (vertex_frame_attributes) for a fragment's frame to match the fit's.
    # VertexEraser owns the VAO/VBO/IBO from here on and rebuilds them on every deletion.
    eraser = VertexEraser(ctx, prog, uv_mesh)

    clock = pygame.time.Clock()
    dragging = False
    erasing = False
    last_mouse = (0, 0)
    total_erased = 0

    # Last frame's MVP: what's actually on screen when an event arrives this frame, so
    # erase hit-testing picks the vertex under the cursor rather than where it will be
    # after this frame's own camera move.
    eye = camera.position.astype(np.float32)
    mvp = (perspective(45.0, width / height, camera.radius * 0.02, camera.radius * 20.0) @ camera.view_matrix())
    mvp = mvp.astype(np.float32)

    def save_if_edited() -> None:
        if not save_edit:
            return
        if total_erased == 0:
            print("--save-edit given but nothing was erased; leaving mesh_uv.obj untouched")
            return
        # One backup of the pre-edit mesh, kept across repeated edit sessions - only
        # taken if it doesn't already exist, so a second sitting can't overwrite the
        # true original with an already-edited one.
        backup_path = uv_mesh_path.with_suffix(".obj.bak")
        if not backup_path.exists():
            shutil.copyfile(uv_mesh_path, backup_path)
            src_idx = uv_mesh_path.with_suffix(".source_vertex_index.npy")
            if src_idx.exists():
                shutil.copyfile(src_idx, backup_path.with_suffix(".source_vertex_index.npy"))
        eraser.to_uv_mesh().save(uv_mesh_path)
        print(
            f"saved: {uv_mesh_path}  ({len(eraser.vertices)} verts, {len(eraser.faces)} faces; "
            f"original backed up to {backup_path})"
        )

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                save_if_edited()
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.key.get_pressed()[pygame.K_e]:
                    erasing, last_mouse = True, event.pos
                    n = eraser.erase_near(mvp, width, height, event.pos, erase_radius_px)
                    if n:
                        total_erased += n
                        print(f"erased {n} vertices ({total_erased} total, {len(eraser.vertices)} left)")
                else:
                    dragging, last_mouse = True, event.pos
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging, erasing = False, False
            if event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.pos[0] - last_mouse[0], event.pos[1] - last_mouse[1]
                last_mouse = event.pos
                camera.rotate(dx * 0.3, dy * 0.3)
            if event.type == pygame.MOUSEMOTION and erasing:
                last_mouse = event.pos
                n = eraser.erase_near(mvp, width, height, event.pos, erase_radius_px)
                if n:
                    total_erased += n
                    print(f"erased {n} vertices ({total_erased} total, {len(eraser.vertices)} left)")
            if event.type == pygame.MOUSEWHEEL:
                camera.radius = float(np.clip(camera.radius * (0.9 ** event.y), radius0 * 0.05, radius0 * 8.0))

        keys = pygame.key.get_pressed()
        right, up = camera.forward_axes
        pan_speed = camera.radius * 0.02
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            camera.origin = camera.origin + pan_speed * up
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            camera.origin = camera.origin - pan_speed * up
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            camera.origin = camera.origin - pan_speed * right
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            camera.origin = camera.origin + pan_speed * right

        eye = camera.position.astype(np.float32)
        view = camera.view_matrix()
        proj = perspective(45.0, width / height, camera.radius * 0.02, camera.radius * 20.0)
        mvp = (proj @ view).astype(np.float32)

        prog["u_mvp"].write(mvp.flatten(order="F").tobytes())
        prog["u_cam_pos"].value = tuple(eye.tolist())

        ctx.clear(0.05, 0.1, 0.55)  # blue: distinguishes empty viewport background from the red/yellow debug overlays
        eraser.vao.render()
        pygame.display.flip()
        clock.tick(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="e.g. data/runs/apple_test")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--num-steps", type=int, default=7, help="POM ray-march layers - match slf's config.num_steps to be exact"
    )
    parser.add_argument(
        "--erase-radius", type=float, default=16.0,
        help="pixel radius around the cursor the E+drag eraser deletes vertices within",
    )
    parser.add_argument(
        "--save-edit", action="store_true",
        help="on quit, overwrite uv_unwrap's mesh_uv.obj with the erased mesh "
             "(keeps a one-time .obj.bak of the pre-edit file; no-op if nothing was erased)",
    )
    args = parser.parse_args()

    for stage_id in ("uv_unwrap", "slf"):
        try:
            find_stage_dir(args.run_dir, stage_id)
        except (FileNotFoundError, RuntimeError) as exc:
            sys.exit(f"Error: {exc}")

    run_viewer(args.run_dir, args.width, args.height, args.num_steps, args.erase_radius, args.save_edit)


if __name__ == "__main__":
    main()
