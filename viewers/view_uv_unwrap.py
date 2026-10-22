#!/usr/bin/env python
"""Interactive viewer for uv_unwrap's result: the low-poly mesh under a procedural UV
checkerboard (stretched squares mean distortion). `--chart-colors` flat-colours each chart,
`--save-uv-layout` writes a 2D wireframe of the atlas, and `--layout-only` does that without a window.

Usage:
    pip install -e ".[viewer]"   # moderngl + pygame
    python viewers/view_uv_unwrap.py data/runs/apple_test
    python viewers/view_uv_unwrap.py data/runs/apple_test/06_uv_unwrap/mesh_uv.obj
    python viewers/view_uv_unwrap.py data/runs/apple_test --chart-colors
    python viewers/view_uv_unwrap.py data/runs/apple_test --save-uv-layout /tmp/uv.png

Controls: left-drag orbit, wheel zoom, WASD/arrows pan, ESC quit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling _common

from _common import DEFAULT_UP, OrbitCamera, capture_up_vector, perspective  # noqa: E402
from pipeline.schemas.manifest import read_manifest  # noqa: E402
from pipeline.schemas.uv_mesh import UVMesh  # noqa: E402

# ─── GLSL shaders ────────────────────────────────────────────────────────────

VERTEX_SHADER = """#version 330
in vec3 in_pos;
in vec2 in_uv;
in vec3 in_normal;
in vec3 in_chart_color;
uniform mat4 u_mvp;
out vec2 v_uv;
out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_chart_color;
void main(){
    v_uv = in_uv;
    v_world_pos = in_pos;
    v_normal = in_normal;
    v_chart_color = in_chart_color;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

FRAGMENT_SHADER = """#version 330
in vec2 v_uv;
in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_chart_color;
uniform vec3 u_cam_pos;
uniform float u_checker_scale;   // squares per [0,1] uv unit
uniform bool u_chart_colors;     // true: flat-color by chart instead of checker
out vec4 f_color;

void main(){
    vec3 base;
    if (u_chart_colors){
        base = v_chart_color;
    } else {
        vec2 c = floor(v_uv * u_checker_scale);
        float parity = mod(c.x + c.y, 2.0);
        base = mix(vec3(0.85, 0.15, 0.15), vec3(0.95, 0.95, 0.95), parity);
    }
    vec3 n = normalize(v_normal);
    vec3 view_dir = normalize(u_cam_pos - v_world_pos);
    float ndotv = max(dot(n, view_dir), 0.0);
    float shade = 0.15 + 0.85 * ndotv;  // same headlamp model as view_mesh.py
    f_color = vec4(base * shade, 1.0);
}
"""


# ─── data loading / analysis ───────────────────────────────────────────────────


def resolve_mesh_uv_path(run_dir_or_mesh: Path) -> Path:
    """Accepts a mesh_uv.obj path, a uv_unwrap stage directory, or a pipeline run directory
    (the stage dir is found by manifest.json `stage == "uv_unwrap"`)."""
    if run_dir_or_mesh.is_file():
        return run_dir_or_mesh
    direct = run_dir_or_mesh / "mesh_uv.obj"
    if direct.exists():
        return direct
    matches = [
        d for d in sorted(run_dir_or_mesh.iterdir())
        if d.is_dir() and (m := read_manifest(d)) and m.stage == "uv_unwrap"
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple 'uv_unwrap' stage dirs found under {run_dir_or_mesh}: {matches}")
    if not matches:
        raise FileNotFoundError(f"no uv_unwrap stage dir (or mesh_uv.obj) found under {run_dir_or_mesh}")
    return matches[0] / "mesh_uv.obj"


def _uv_texture_resolution(mesh_uv_path: Path) -> tuple[int, int]:
    manifest = read_manifest(mesh_uv_path.parent)
    if manifest is not None and "texture_resolution" in manifest.params:
        w, h = manifest.params["texture_resolution"]
        return (w, h)
    return (1024, 1024)


def uv_chart_ids(uv_mesh: UVMesh) -> np.ndarray:
    """Connected components of the uv_mesh's own (post seam-cut) graph, which are the UV charts."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    v = len(uv_mesh.vertices)
    f = uv_mesh.faces
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    graph = coo_matrix((data, (rows, cols)), shape=(v, v))
    n_components, labels = connected_components(graph, directed=False)
    return labels


def uv_packing_efficiency(uv_mesh: UVMesh) -> float:
    """Total UV-space triangle area (the atlas square is 1.0)."""
    uv = uv_mesh.uv
    f = uv_mesh.faces
    d1 = uv[f[:, 1]] - uv[f[:, 0]]
    d2 = uv[f[:, 2]] - uv[f[:, 0]]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    return float(area.sum())


_CHART_COLOR_GOLDEN_ANGLE = 0.6180339887498949


def chart_id_colors(chart_ids: np.ndarray) -> np.ndarray:
    """One RGB color per chart id via a golden-angle hue walk."""
    n_charts = int(chart_ids.max()) + 1
    hues = (np.arange(n_charts) * _CHART_COLOR_GOLDEN_ANGLE) % 1.0
    colors = np.stack([_hsv_to_rgb(h, 0.65, 0.95) for h in hues], axis=0)
    return colors[chart_ids]


def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return np.array([r, g, b], dtype=np.float32)


def save_uv_layout(uv_mesh: UVMesh, out_path: Path, size: int = 2048) -> None:
    """Flat top-down wireframe of the UV atlas. No GL context needed."""
    import cv2

    img = np.full((size, size, 3), 255, dtype=np.uint8)
    uv_px = np.stack(
        [uv_mesh.uv[:, 0] * (size - 1), (1.0 - uv_mesh.uv[:, 1]) * (size - 1)], axis=-1
    ).astype(np.int32)
    for i0, i1, i2 in uv_mesh.faces:
        pts = uv_px[[i0, i1, i2]].reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=(60, 60, 60), thickness=1, lineType=cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ─── entry point ─────────────────────────────────────────────────────────────


def run_viewer(
    uv_mesh: UVMesh,
    width: int,
    height: int,
    checker_scale: float,
    chart_colors_mode: bool,
    up: np.ndarray = DEFAULT_UP,
) -> None:
    import moderngl
    import pygame

    chart_ids = uv_chart_ids(uv_mesh)
    chart_colors = chart_id_colors(chart_ids).astype(np.float32) if chart_colors_mode else np.zeros(
        (len(uv_mesh.vertices), 3), dtype=np.float32
    )

    vertex_normals = uv_mesh.vertex_normals
    if vertex_normals is None:
        vertex_normals = uv_mesh.to_trimesh().vertex_normals

    centroid = uv_mesh.vertices.mean(axis=0)
    radius0 = float(np.linalg.norm(uv_mesh.vertices.max(axis=0) - uv_mesh.vertices.min(axis=0))) * 1.2
    camera = OrbitCamera(origin=centroid, radius=max(radius0, 1e-3), up=up)

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption(
        "UV unwrap viewer  [drag: orbit | wheel: zoom | WASD: pan | c: toggle chart colors | ESC: quit]"
    )

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

    verts = np.concatenate(
        [
            uv_mesh.vertices.astype(np.float32),
            uv_mesh.uv.astype(np.float32),
            vertex_normals.astype(np.float32),
            chart_colors,
        ],
        axis=1,
    )
    vbo = ctx.buffer(np.ascontiguousarray(verts).tobytes())
    ibo = ctx.buffer(np.ascontiguousarray(uv_mesh.faces.astype(np.int32)).tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 2f 3f 3f", "in_pos", "in_uv", "in_normal", "in_chart_color")], ibo)

    prog["u_checker_scale"].value = float(checker_scale)
    prog["u_chart_colors"].value = bool(chart_colors_mode)

    clock = pygame.time.Clock()
    dragging = False
    last_mouse = (0, 0)
    show_charts = chart_colors_mode

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                show_charts = not show_charts
                prog["u_chart_colors"].value = show_charts
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

        eye = camera.position.astype(np.float32)
        view = camera.view_matrix()
        proj = perspective(45.0, width / height, camera.radius * 0.02, camera.radius * 20.0)
        mvp = (proj @ view).astype(np.float32)

        prog["u_mvp"].write(mvp.flatten(order="F").tobytes())
        prog["u_cam_pos"].value = tuple(eye.tolist())

        ctx.clear(0.05, 0.1, 0.55)
        vao.render()
        pygame.display.flip()
        clock.tick(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="a run dir, a uv_unwrap stage dir, or a mesh_uv.obj file")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--checker-scale", type=float, default=40.0, help="checker squares per [0,1] uv unit")
    parser.add_argument(
        "--chart-colors", action="store_true", help="start flat-colored by UV chart instead of checkered ('c' toggles live)"
    )
    parser.add_argument("--save-uv-layout", type=Path, default=None, help="also save a flat 2D UV wireframe PNG")
    parser.add_argument(
        "--layout-only", action="store_true", help="save --save-uv-layout and exit, without opening the 3D viewer"
    )
    args = parser.parse_args()

    if not args.path.exists():
        parser.error(f"path not found: {args.path}")
    if args.layout_only and args.save_uv_layout is None:
        parser.error("--layout-only requires --save-uv-layout")

    mesh_uv_path = resolve_mesh_uv_path(args.path)
    uv_mesh = UVMesh.load(mesh_uv_path, texture_resolution=_uv_texture_resolution(mesh_uv_path))

    chart_ids = uv_chart_ids(uv_mesh)
    n_charts = int(chart_ids.max()) + 1
    utilization = uv_packing_efficiency(uv_mesh)
    print(
        f"{mesh_uv_path}: {len(uv_mesh.vertices)} verts, {len(uv_mesh.faces)} faces  |  "
        f"{n_charts} UV charts  |  atlas utilization {utilization * 100:.1f}%  |  "
        f"texture_resolution {uv_mesh.texture_resolution}"
    )

    if args.save_uv_layout is not None:
        save_uv_layout(uv_mesh, args.save_uv_layout)
        print(f"wrote UV layout wireframe -> {args.save_uv_layout}")
        if args.layout_only:
            return

    # capture_up_vector accepts a stage dir or a bare .obj (falls back to +Y).
    run_viewer(
        uv_mesh, args.width, args.height, args.checker_scale, args.chart_colors, up=capture_up_vector(args.path)
    )


if __name__ == "__main__":
    main()
