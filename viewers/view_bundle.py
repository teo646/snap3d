#!/usr/bin/env python
"""Interactive viewer for an `export` bundle - the four files in a run's
`<run_id>.snap3d/` folder and nothing else.

This is the consumer's view of the deliverable: it never opens the run it came from, so
anything it needs that the bundle does not carry is a hole in the format. What it does
share with the pipeline is *code*, deliberately - the GLSL comes from
`pipeline/core/slf_shaders.py` (the same source the fit compiled) and the per-vertex
frame from `uv_geometry.vertex_frame_attributes`, because reproducing the fit's frame
approximately is the one way to get subtly wrong colors everywhere.

The frame is recomputed here rather than loaded: normals and the raw dP/du, dP/dv are
pure functions of position + uv + indices, and at 65k vertices that costs ~30 ms
against ~1.6 MB of payload they would otherwise add.

The bundle is written in glTF and KTX2's own conventions (UV origin top-left, texture
row 0 first), so the two conversions into this repo's v-up shader happen here, at load:
`v <- 1 - v` on TEXCOORD_0, and the row flip `to_gl_texture` already does. A consumer
writing its own shader against the glTF convention does neither, and instead flips the
sign of the `-H` term in `xy_per_height`.

Usage:
    pip install -e ".[viewer]"   # moderngl + pygame, not installed by default
    python viewers/view_bundle.py data/runs/<run_id>/NN_export/<run_id>_unorm8.snap3d
    python viewers/view_bundle.py data/runs/<run_id>/10_export   # the stage dir works too

Controls: left-drag orbit, wheel zoom, WASD/arrows pan, Q/E roll the up axis,
R reset view (including the up axis), ESC quit.

Pass --modify_initial_cam to have quitting (ESC or closing the window) write the
camera's pose at that moment back into the bundle's config.json as its initial_camera -
a way to pick the opening shot by eye instead of by re-running the export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import moderngl
import numpy as np
import pygame

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.core.camera_raster import to_gl_texture  # noqa: E402
from pipeline.core.gltf import read_glb  # noqa: E402
from pipeline.core.ktx2 import read_ktx2  # noqa: E402
from pipeline.core.orbit_camera import OrbitCamera, horizontal_basis, perspective  # noqa: E402
from pipeline.core.render_bundle import BUNDLE_FILES, BUNDLE_SUFFIX  # noqa: E402
from pipeline.core.slf_shaders import EVAL_SH_GLSL, RESOLVE_GLSL, VERTEX_SHADER  # noqa: E402
from pipeline.core.uv_geometry import vertex_frame_attributes  # noqa: E402
from pipeline.schemas.uv_mesh import UVMesh  # noqa: E402

BACKGROUND = (0.05, 0.05, 0.06)
ROLL_DEG_PER_FRAME = 0.6  # held key, at the ~60 fps clock.tick below targets: 36 deg/s

FRAGMENT_BODY = (
    RESOLVE_GLSL
    + EVAL_SH_GLSL
    + """
in vec2 v_uv;
in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
out vec4 f_color;

void main(){
    vec3 N = normalize(v_normal);
    vec2 resolved_uv = resolve_parallax_uv(v_uv, v_world_pos, N, v_tangent, v_bitangent);
    // g of the relief texture is the atlas coverage - chart interiors plus their
    // gutter. discard, not a debug colour: this viewer shows what a consumer would ship.
    if (texture(u_height_tex, resolved_uv).g < 0.5) discard;

    vec3 view_w = normalize(u_cam_pos - v_world_pos);
    // Gram-Schmidt against N, then B derived by cross product - not the interpolated
    // v_bitangent, whose sign follows the UV chart's handedness and flips on the ~46%
    // of charts xatlas mirrors. The fit used this frame; the march above uses the raw
    // tangents for their lengths.
    vec3 T = normalize(v_tangent - N * dot(v_tangent, N));
    vec3 B = cross(N, T);
    vec3 view_l = vec3(dot(view_w, T), dot(view_w, B), dot(view_w, N));
    f_color = vec4(eval_sh(resolved_uv, view_l), 1.0);
}
"""
)


def resolve_bundle_dir(path: Path) -> Path:
    """Accepts either the `.snap3d` folder or the export stage dir that holds it, so a
    path copied out of a run listing works without knowing which level it points at.

    The folder is named after its run, so it is found by suffix rather than by name; a
    stage dir with no `.snap3d` child is returned as-is, which is what a bundle exported
    before this convention looks like."""
    path = Path(path)
    if path.name.endswith(BUNDLE_SUFFIX):
        return path
    nested = sorted(child for child in path.glob(f"*{BUNDLE_SUFFIX}") if child.is_dir())
    if len(nested) > 1:
        # An export holds one bundle per storage; the unorm8 one is the default to open.
        unorm8 = [child for child in nested if child.name.endswith(f"_unorm8{BUNDLE_SUFFIX}")]
        if len(unorm8) == 1:
            print(f"{path} holds {len(nested)} bundles; opening {unorm8[0].name} "
                  f"(name another .snap3d folder to open that one)")
            return unorm8[0]
        names = ", ".join(child.name for child in nested)
        sys.exit(f"Error: {path} holds more than one bundle ({names}) - name the one to open")
    return nested[0] if nested else path


def load_bundle(bundle_dir: Path):
    """Everything the renderer needs, read from the bundle alone."""
    bundle_dir = Path(bundle_dir)
    config = json.loads((bundle_dir / "config.json").read_text())
    files = config["files"]
    positions, uv, indices = read_glb(bundle_dir / files["mesh"])
    sh = read_ktx2(bundle_dir / files["sh"])  # (K, H, W, 4), a unused; float32, or uint8 for "unorm8"
    relief = read_ktx2(bundle_dir / files["height"])  # (H, W, 2): displacement, coverage

    # v flipped back out of glTF's top-left origin, and the coefficients put back in
    # the (H, W, K, 3) layout the fit and the shader think in.
    uv = np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1)
    # vertex_normals=None on purpose: the bundle ships no frame, so this is the path a
    # consumer takes, and it must be the path the viewer exercises.
    uv_mesh = UVMesh(
        vertices=positions.astype(np.float64),
        faces=indices.astype(np.int64),
        uv=uv,
        vertex_normals=None,
        texture_resolution=tuple(config["texture_resolution"]),
    )
    return config, uv_mesh, np.transpose(sh[..., :3], (1, 2, 0, 3)), relief


def write_back_camera(bundle_dir: Path, config: dict, camera: OrbitCamera) -> None:
    """Overwrite config.json's initial_camera - and up_vector, if rolling moved it -
    with `camera`'s current pose.

    `position` and `view_matrix` are derived fields - render_bundle.py writes them so a
    consumer needs neither the orbit convention nor a look_at of its own to reproduce
    the shot - so they are recomputed here too, rather than left stale. `near`/`far`
    keep whatever multiple of the radius the export used, since view_bundle.py has no
    opinion of its own on clip planes.

    up_vector is the pole every other pose in the bundle orbits, so a change to it here
    - only ever a Q/E roll, in this viewer - has to land on `config["up_vector"]`
    itself, not just on this camera's own `initial_camera` entry.
    """
    cam_cfg = config["initial_camera"]
    near_scale = cam_cfg["near"] / cam_cfg["radius"]
    far_scale = cam_cfg["far"] / cam_cfg["radius"]
    config["up_vector"] = camera.up.tolist()
    cam_cfg["target"] = camera.origin.tolist()
    cam_cfg["radius"] = float(camera.radius)
    cam_cfg["azimuth_deg"] = float(camera.azimuth)
    cam_cfg["elevation_deg"] = float(camera.elevation)
    cam_cfg["position"] = camera.position.tolist()
    cam_cfg["view_matrix"] = camera.view_matrix().tolist()
    cam_cfg["near"] = near_scale * camera.radius
    cam_cfg["far"] = far_scale * camera.radius

    config_path = Path(bundle_dir) / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"  wrote up_vector and initial_camera to {config_path}")


def run_viewer(bundle_dir: Path, width: int, height: int, modify_initial_cam: bool = False) -> None:
    config, uv_mesh, sh, relief = load_bundle(bundle_dir)
    degree, k_coeffs = config["sh"]["degree"], sh.shape[2]
    unorm8 = config["sh"].get("storage", "float16") == "unorm8"
    num_steps = config["height"]["num_steps"]
    cam_cfg = config["initial_camera"]

    print(
        f"{config['mesh']['vertices']} verts, {config['mesh']['faces']} faces  |  "
        f"SH degree {degree} ({k_coeffs} coeffs)  |  "
        f"{config['texture_resolution'][0]}x{config['texture_resolution'][1]}  |  "
        f"covered texels {(relief[..., 1] > 0.5).mean() * 100:.1f}%  |  num_steps {num_steps}  |  "
        f"SH storage {config['sh'].get('storage', 'float16')}"
    )

    camera = OrbitCamera(
        origin=np.asarray(cam_cfg["target"]),
        radius=cam_cfg["radius"],
        up=np.asarray(config["up_vector"]),
    )
    camera.azimuth, camera.elevation = cam_cfg["azimuth_deg"], cam_cfg["elevation_deg"]
    home = (camera.azimuth, camera.elevation, camera.radius, camera.origin.copy(), camera.up.copy())
    radius0 = camera.radius

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption(
        f"bundle: {Path(bundle_dir).name}  [drag orbit | wheel zoom | WASD pan | Q/E roll | R reset]"
    )

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    prog = ctx.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader=f"#version 330\n#define DEGREE {degree}\n"
        f"#define K_COEFFS {k_coeffs}\n#define NUM_STEPS {num_steps}\n"
        + ("#define SH_UNORM8\n" if unorm8 else "") + FRAGMENT_BODY,
    )

    tex_w, tex_h = config["texture_resolution"]
    height_tex = to_gl_texture(ctx, relief.astype(np.float32))
    height_tex.use(0)
    prog["u_height_tex"].value = 0
    prog["u_tex_size"].value = (float(tex_w), float(tex_h))
    prog["u_height_range"].value = float(config["height"]["range"])
    prog["u_layer_step"].value = float(2.0 * config["height"]["range"] / num_steps)

    # (H, W, K, 3) -> one texture-array layer per coefficient, v flipped like every
    # other texture here (GL's origin is bottom-left, the atlas's is top-left).
    if unorm8:
        # Uploaded as the bytes themselves (normalized unsigned, "f1"): half the memory of
        # RGBA16F, and the shader applies offset + scale after the filtered lookup.
        layers = np.ascontiguousarray(np.transpose(np.flipud(sh), (2, 0, 1, 3)), dtype=np.uint8)
        coeff_tex = ctx.texture_array((tex_w, tex_h, k_coeffs), 3, layers.tobytes(), dtype="f1")
        prog["u_sh_offset"].write(np.asarray(config["sh"]["offset"], dtype=np.float32).tobytes())
        prog["u_sh_scale"].write(np.asarray(config["sh"]["scale"], dtype=np.float32).tobytes())
    else:
        layers = np.ascontiguousarray(np.transpose(np.flipud(sh), (2, 0, 1, 3)), dtype=np.float32)
        coeff_tex = ctx.texture_array((tex_w, tex_h, k_coeffs), 3, layers.tobytes(), dtype="f4")
    coeff_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    coeff_tex.repeat_x = coeff_tex.repeat_y = False
    coeff_tex.use(3)
    prog["u_coeffs"].value = 3

    normals, tangent_raw, bitangent_raw = vertex_frame_attributes(uv_mesh)
    verts = np.concatenate(
        [
            uv_mesh.vertices.astype(np.float32),
            uv_mesh.uv.astype(np.float32),
            normals.astype(np.float32),
            tangent_raw.astype(np.float32),
            bitangent_raw.astype(np.float32),
        ],
        axis=1,
    )
    vao = ctx.vertex_array(
        prog,
        [(ctx.buffer(np.ascontiguousarray(verts).tobytes()), "3f 2f 3f 3f 3f",
          "in_pos", "in_uv", "in_normal", "in_tangent", "in_bitangent")],
        ctx.buffer(np.ascontiguousarray(uv_mesh.faces.astype(np.int32)).tobytes()),
    )

    clock = pygame.time.Clock()
    dragging = False
    last_mouse = (0, 0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                if modify_initial_cam:
                    write_back_camera(bundle_dir, config, camera)
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                camera.azimuth, camera.elevation, camera.radius, camera.origin, camera.up = (
                    home[0], home[1], home[2], home[3].copy(), home[4].copy()
                )
                camera._right, camera._forward = horizontal_basis(camera.up)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging, last_mouse = True, event.pos
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            if event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.pos[0] - last_mouse[0], event.pos[1] - last_mouse[1]
                last_mouse = event.pos
                camera.azimuth -= dx * 0.3
                camera.elevation += dy * 0.3
            if event.type == pygame.MOUSEWHEEL:
                camera.radius = float(np.clip(camera.radius * (0.9**event.y), radius0 * 0.05, radius0 * 8.0))

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
        # Rolling the up axis is how a crooked capture gets straightened by eye, rather
        # than by re-running whatever estimated it in the first place.
        if keys[pygame.K_q]:
            camera.roll(-ROLL_DEG_PER_FRAME)
        if keys[pygame.K_e]:
            camera.roll(ROLL_DEG_PER_FRAME)

        eye = camera.position.astype(np.float32)
        mvp = (
            perspective(cam_cfg["fov_deg"], width / height, camera.radius * 0.02, camera.radius * 20.0)
            @ camera.view_matrix()
        ).astype(np.float32)
        prog["u_mvp"].write(mvp.flatten(order="F").tobytes())
        prog["u_cam_pos"].value = tuple(eye.tolist())

        ctx.clear(*BACKGROUND)
        vao.render()
        pygame.display.flip()
        clock.tick(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help=f"a *{BUNDLE_SUFFIX} folder, or the export stage dir holding it",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--modify_initial_cam",
        action="store_true",
        help="on quit, write the camera's pose at that moment back into config.json as initial_camera",
    )
    args = parser.parse_args()

    bundle_dir = resolve_bundle_dir(args.bundle_dir)
    missing = [n for n in BUNDLE_FILES.values() if not (bundle_dir / n).exists()]
    if missing:
        sys.exit(f"Error: {bundle_dir} is not an export bundle - missing {', '.join(missing)}")

    run_viewer(bundle_dir, args.width, args.height, args.modify_initial_cam)


if __name__ == "__main__":
    main()
