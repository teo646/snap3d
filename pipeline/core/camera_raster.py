"""GPU low-poly mesh rasterization + POM march in one moderngl pass: per camera pixel, the
resolved atlas coordinate that pixel observes. The march is slf_shaders.RESOLVE_GLSL.

Alongside each uv it writes the camera-space depth of the surface that uv stands for, and,
given a reference mesh, that mesh's own depth, so occluded pixels can be rejected.
"""

from __future__ import annotations

import numpy as np

from pipeline.core.slf_shaders import RESOLVE_GLSL, VERTEX_SHADER
from pipeline.core.uv_geometry import vertex_frame_attributes

__all__ = [
    "ParallaxRasterContext",
    "colmap_Rt_to_gl_view",
    "intrinsics_to_gl_projection",
    "to_gl_texture",
]

_FRAGMENT_BODY = (
    RESOLVE_GLSL
    + """
in vec2 v_uv;
in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
uniform sampler2D u_valid_tex;  // atlas.valid (uv_unwrap coverage, r channel)
uniform vec4 u_view_z;          // third row of the world->camera [R|t]
out vec3 f_uv_depth;

void main(){
    vec2 resolved_uv = resolve_parallax_uv(v_uv, v_world_pos, v_normal, v_tangent, v_bitangent);
    if (resolved_uv.x < 0.0 || resolved_uv.x > 1.0 || resolved_uv.y < 0.0 || resolved_uv.y > 1.0) discard;
    if (texture(u_valid_tex, resolved_uv).r < 0.5) discard;
    // The march's own geometry, inverted: a point at height h above the flat surface
    // lies h / vz along the view ray from the flat point (the same vz clamp the march
    // uses). h is read back at the resolved uv, which is where the ray met the surface.
    vec3 view_world = normalize(u_cam_pos - v_world_pos);
    float vz = max(dot(view_world, normalize(v_normal)), VZ_MIN);
    float h = sample_height(uv_to_xy(resolved_uv));
    vec3 hit = v_world_pos + view_world * (h / vz);
    f_uv_depth = vec3(clamp(resolved_uv, 0.0, 1.0), dot(u_view_z.xyz, hit) + u_view_z.w);
}
"""
)

_DEPTH_VERTEX_SHADER = """
#version 330
in vec3 in_pos;
uniform mat4 u_mvp;
uniform vec4 u_view_z;
out float v_depth;
void main(){
    v_depth = dot(u_view_z.xyz, in_pos) + u_view_z.w;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

_DEPTH_FRAGMENT_SHADER = """
#version 330
in float v_depth;
out float f_depth;
void main(){
    f_depth = v_depth;
}
"""


def _fs_header(num_steps: int) -> str:
    return f"#version 330\n#define NUM_STEPS {num_steps}\n"


def intrinsics_to_gl_projection(
    fx: float, fy: float, cx: float, cy: float, width: int, height: int, near: float, far: float
) -> np.ndarray:
    """COLMAP pinhole intrinsics -> OpenGL clip-space projection matrix."""
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2 * fx / width
    P[1, 1] = 2 * fy / height
    P[0, 2] = 1 - 2 * cx / width
    P[1, 2] = 2 * cy / height - 1
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = -1
    return P


def colmap_Rt_to_gl_view(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """COLMAP world->camera (R, t) -> OpenGL view matrix (flips y/z)."""
    V = np.eye(4, dtype=np.float64)
    V[:3, :3] = R
    V[:3, 3] = t
    flip = np.diag([1, -1, -1, 1]).astype(np.float64)
    return flip @ V


def to_gl_texture(ctx, raster: np.ndarray):
    """Uploads a (H, W, C) float32 raster (row 0 = top, v=1) as a GL texture, flipped so
    sampling with the mesh's own (u, v) lands on the correct texel."""
    import moderngl

    data = np.ascontiguousarray(np.flipud(raster), dtype=np.float32)
    h, w, c = data.shape
    tex = ctx.texture((w, h), c, data.tobytes(), dtype="f4")
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


class ParallaxRasterContext:
    """Holds the GL resources for one slf_fit run, built once and reused across photos (only the
    MVP changes). Call `release()` when done."""

    def __init__(
        self,
        uv_mesh,
        atlas,  # pipeline.core.uv_geometry.UVAtlasGeometry
        height_data: np.ndarray,
        height_range: float,
        num_steps: int,
        reference_mesh=None,  # pipeline.schemas.mesh.Mesh; enables reference depth in resolve()
    ) -> None:
        import moderngl

        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=_fs_header(num_steps) + _FRAGMENT_BODY)

        W, H = uv_mesh.texture_resolution
        textures = [
            to_gl_texture(self.ctx, height_data.astype(np.float32)[..., None]),
            to_gl_texture(self.ctx, atlas.valid.astype(np.float32)[..., None]),
        ]
        for unit, tex in enumerate(textures):
            tex.use(unit)
        self._textures = textures
        self.prog["u_height_tex"].value = 0
        self.prog["u_valid_tex"].value = 1
        self.prog["u_tex_size"].value = (float(W), float(H))
        self.prog["u_height_range"].value = float(height_range)
        self.prog["u_layer_step"].value = float(2.0 * height_range / num_steps)

        # The three frame attributes must be the same arrays uv_geometry interpolated into the atlas.
        self._verts = uv_mesh.vertices.astype(np.float64)
        normals, tangent_raw, bitangent_raw = vertex_frame_attributes(uv_mesh)
        vbo_data = np.concatenate(
            [
                uv_mesh.vertices.astype(np.float32),
                uv_mesh.uv.astype(np.float32),
                normals.astype(np.float32),
                tangent_raw.astype(np.float32),
                bitangent_raw.astype(np.float32),
            ],
            axis=1,
        )  # (V, 14): xyz + uv + normal + raw tangent + raw bitangent
        self._vbo = self.ctx.buffer(np.ascontiguousarray(vbo_data).tobytes())
        self._ibo = self.ctx.buffer(np.ascontiguousarray(uv_mesh.faces.astype("u4")).tobytes())
        self._vao = self.ctx.vertex_array(
            self.prog,
            [(self._vbo, "3f 2f 3f 3f 3f", "in_pos", "in_uv", "in_normal", "in_tangent", "in_bitangent")],
            self._ibo,
        )

        self._fbo_cache: dict[tuple[int, int, int], object] = {}
        # Pinned host buffers reused across views.
        self._host_cache: dict[tuple[int, int, int], object] = {}

        self._ref = None
        if reference_mesh is not None:
            ref_prog = self.ctx.program(vertex_shader=_DEPTH_VERTEX_SHADER, fragment_shader=_DEPTH_FRAGMENT_SHADER)
            ref_vbo = self.ctx.buffer(np.ascontiguousarray(reference_mesh.vertices.astype(np.float32)).tobytes())
            ref_ibo = self.ctx.buffer(np.ascontiguousarray(reference_mesh.faces.astype("u4")).tobytes())
            ref_vao = self.ctx.vertex_array(ref_prog, [(ref_vbo, "3f", "in_pos")], ref_ibo)
            self._ref = (ref_prog, ref_vbo, ref_ibo, ref_vao)

    def _get_fbo(self, width: int, height: int, components: int):
        key = (width, height, components)
        fbo = self._fbo_cache.get(key)
        if fbo is None:
            color = self.ctx.texture((width, height), components, dtype="f4")
            depth = self.ctx.depth_renderbuffer((width, height))
            fbo = self.ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
            self._fbo_cache[key] = fbo
        return fbo

    def _host_buffer(self, width: int, height: int, components: int) -> np.ndarray:
        key = (width, height, components)
        buf = self._host_cache.get(key)
        if buf is None:
            import torch

            pin = torch.cuda.is_available()
            buf = torch.empty((height, width, components), dtype=torch.float32, pin_memory=pin)
            self._host_cache[key] = buf
        return buf

    def _render(
        self, K: np.ndarray, R: np.ndarray, t: np.ndarray, cam_center: np.ndarray, width: int, height: int
    ):
        """Renders both passes into the pinned host buffers. Returns (uv_depth (H, W, 3),
        reference (H, W, 1) or None, any_front), in OpenGL's bottom-up row order."""
        cam_pts = (R @ self._verts.T).T + t
        depth = cam_pts[:, 2]
        front_depth = depth[depth > 1e-6]
        if front_depth.size == 0:
            return None, None, False
        near = max(float(front_depth.min()) * 0.5, 1e-5)
        far = float(front_depth.max()) * 2.0 + 1e-3

        # +0.5: GL samples fragments at pixel centers.
        proj = intrinsics_to_gl_projection(K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5, width, height, near, far)
        view = colmap_Rt_to_gl_view(R, t)
        mvp = (proj @ view).astype(np.float32)
        view_z = (float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(t[2]))

        fbo = self._get_fbo(width, height, 3)
        fbo.use()
        fbo.clear(-1.0, -1.0, 0.0, 0.0, depth=1.0)
        self.prog["u_mvp"].write(mvp.flatten(order="F").tobytes())
        self.prog["u_cam_pos"].value = tuple(float(x) for x in cam_center)
        self.prog["u_view_z"].value = view_z
        self._vao.render()
        uv_depth = self._host_buffer(width, height, 3)
        fbo.read_into(uv_depth.numpy(), components=3, dtype="f4")

        reference = None
        if self._ref is not None:
            ref_prog, _, _, ref_vao = self._ref
            ref_fbo = self._get_fbo(width, height, 1)
            ref_fbo.use()
            ref_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
            ref_prog["u_mvp"].write(mvp.flatten(order="F").tobytes())
            ref_prog["u_view_z"].value = view_z
            ref_vao.render()
            reference = self._host_buffer(width, height, 1)
            ref_fbo.read_into(reference.numpy(), components=1, dtype="f4")
        return uv_depth, reference, True

    def resolve_torch(
        self, K: np.ndarray, R: np.ndarray, t: np.ndarray, cam_center: np.ndarray, width: int, height: int, device
    ):
        """`resolve` as tensors on `device`: uv (H, W, 2), valid (H, W) bool, surface_depth (H, W),
        reference_depth (H, W) or None."""
        import torch

        uv_depth, reference, any_front = self._render(K, R, t, cam_center, width, height)
        if not any_front:
            z = torch.zeros((height, width), device=device)
            return torch.zeros((height, width, 2), device=device), z.bool(), z, None
        raw = uv_depth.to(device, non_blocking=True).flip(0)  # OpenGL row0=bottom -> row0=top
        valid = raw[:, :, 0] >= 0.0  # cleared to -1
        uv = torch.where(valid.unsqueeze(-1), raw[:, :, :2], 0.0)
        surface_depth = torch.where(valid, raw[:, :, 2], 0.0)
        reference_depth = None
        if reference is not None:
            reference_depth = reference.to(device, non_blocking=True).flip(0)[:, :, 0]
        return uv, valid, surface_depth, reference_depth

    def resolve(
        self, K: np.ndarray, R: np.ndarray, t: np.ndarray, cam_center: np.ndarray, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Returns (uv (H, W, 2) in [0,1], valid (H, W) bool, surface_depth (H, W),
        reference_depth (H, W) or None).

        valid is False where no triangle covers the pixel, the resolved uv left [0, 1], or
        atlas.valid is false. surface_depth is the camera-space z of the height-displaced point;
        reference_depth is the reference mesh's camera-space z (0 where it covers nothing).
        """
        uv_depth, reference, any_front = self._render(K, R, t, cam_center, width, height)
        if not any_front:
            z = np.zeros((height, width), dtype=np.float32)
            return np.zeros((height, width, 2), dtype=np.float64), z.astype(bool), z, None
        raw = np.flipud(uv_depth.numpy())  # OpenGL row0=bottom -> row0=top

        valid = raw[:, :, 0] >= 0.0  # cleared to -1
        uv = raw[:, :, :2].astype(np.float64)
        uv[~valid] = 0.0
        surface_depth = np.where(valid, raw[:, :, 2], 0.0).astype(np.float32)

        reference_depth = None
        if reference is not None:
            # copy: the host buffer is reused
            reference_depth = np.flipud(reference.numpy()[:, :, 0]).copy()
        return uv, valid, surface_depth, reference_depth

    def release(self) -> None:
        if self._ref is not None:
            for obj in reversed(self._ref):
                obj.release()
        self._vao.release()
        self._ibo.release()
        self._vbo.release()
        for tex in self._textures:
            tex.release()
        for fbo in self._fbo_cache.values():
            fbo.color_attachments[0].release()
            fbo.depth_attachment.release()
            fbo.release()
        self.ctx.release()
