"""GLSL shader source shared between SLF training's GPU parallax-resolve pass
(camera_raster.py, called from slf_fit.py's per-photo loop) and the interactive SLF
viewer (viewes/view_slf.py) - so the exact procedure that decided which uv a photo
pixel's observation was fit against, and the exact procedure the viewer resolves at
render time, are one literal shared GLSL source rather than two independently-
maintained ports of the same math.
"""

from __future__ import annotations

__all__ = ["VERTEX_SHADER", "RESOLVE_GLSL", "EVAL_SH_GLSL"]

VERTEX_SHADER = """
#version 330

in vec3 in_pos;
in vec2 in_uv;
in vec3 in_normal;
in vec3 in_tangent;    // RAW dP/du: length = world units per UV-u unit
in vec3 in_bitangent;  // RAW dP/dv
uniform mat4 u_mvp;
out vec2 v_uv;
out vec3 v_world_pos;
// All three interpolated and NOT normalized here: the fragment needs the tangents'
// lengths (see uv_geometry.vertex_frame_attributes), and normalizing per vertex would
// throw exactly that away.
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_bitangent;
void main(){
    v_uv = in_uv;
    v_world_pos = in_pos;
    v_normal = in_normal;
    v_tangent = in_tangent;
    v_bitangent = in_bitangent;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

RESOLVE_GLSL = """
uniform vec3      u_cam_pos;
uniform vec2      u_tex_size;      // (W, H) - the texture's real texel dimensions
uniform float     u_height_range;
uniform float     u_layer_step;    // 2*height_range / NUM_STEPS
uniform sampler2D u_height_tex;

const float VZ_MIN = 0.15;  // matches the reference repo's render.RESOLVE VZ_MIN

vec2 xy_per_height(vec3 t_raw, vec3 b_raw){
    return vec2(u_tex_size.x / max(length(t_raw), 1e-12), -u_tex_size.y / max(length(b_raw), 1e-12));
}

vec2 xy_to_uv(vec2 xy){
    return vec2((xy.x + 0.5) / u_tex_size.x, 1.0 - (xy.y + 0.5) / u_tex_size.y);
}

vec2 uv_to_xy(vec2 uv){
    return vec2(uv.x * u_tex_size.x - 0.5, (1.0 - uv.y) * u_tex_size.y - 0.5);
}

float sample_height(vec2 xy){
    return texture(u_height_tex, xy_to_uv(xy)).r;
}

vec2 resolve_parallax_uv(vec2 uv0, vec3 flat_pos0, vec3 n0, vec3 t_raw, vec3 b_raw){
    vec2 xy0 = uv_to_xy(uv0);
    vec3 N0 = normalize(n0);
    vec3 T0 = normalize(t_raw);
    vec3 B0 = normalize(b_raw);
    vec2 xy_per_h = xy_per_height(t_raw, b_raw);

    vec3 view_world = normalize(u_cam_pos - flat_pos0);
    vec3 view_local = vec3(dot(view_world, T0), dot(view_world, B0), dot(view_world, N0));
    float vz = max(view_local.z, VZ_MIN);
    vec2 dir_xy = (view_local.xy / vz) * xy_per_h;

    float probe_h = u_height_range;
    vec2 probe_xy = xy0 + probe_h * dir_xy;
    float surf_h = sample_height(probe_xy);

    float prev_h = probe_h;
    float prev_surf_h = surf_h;
    vec2 prev_xy = probe_xy;

    for (int i = 0; i < NUM_STEPS; ++i){
        if (surf_h >= probe_h){
            break;  // already past the surface - stop, don't fold this step into prev
        }
        prev_h = probe_h;
        prev_surf_h = surf_h;
        prev_xy = probe_xy;
        probe_h -= u_layer_step;
        probe_xy = xy0 + probe_h * dir_xy;
        surf_h = sample_height(probe_xy);
    }

    float prev_diff = prev_surf_h - prev_h;
    float cur_diff = surf_h - probe_h;
    float denom = prev_diff - cur_diff;
    float weight = 0.0;
    if (abs(denom) > 1e-8){
        weight = clamp(prev_diff / denom, 0.0, 1.0);
    }
    return xy_to_uv(prev_xy * (1.0 - weight) + probe_xy * weight);
}
"""

EVAL_SH_GLSL = """
uniform sampler2DArray u_coeffs;
#ifdef SH_UNORM8
// Bundle sh_storage "unorm8": the array holds RGBA8 UNORM samples, and each coefficient
// is offset + scale * sample. Affine, so applying it after the GPU's bilinear filter is
// the same as filtering the dequantized values.
uniform vec3 u_sh_offset[K_COEFFS];
uniform vec3 u_sh_scale[K_COEFFS];
#endif

vec3 eval_sh(vec2 uv, vec3 view_local){
    vec3 v = normalize(view_local);
    // hsh_remap (sh_math.py): z clamped to [0,1] (hemisphere boundary, not
    // extrapolated), then folded onto the full sphere - matches training exactly.

    // flat: backface (or through a hole into the mesh interior)
    if(v.z <= 0){
        return vec3(0.35, 0.33, 0.30);
    }
    float hz = clamp(v.z, 0.0, 1.0);
    vec3 d = vec3(2.0 * v.x * hz, 2.0 * v.y * hz, 2.0 * hz * hz - 1.0);
    float x = d.x, y = d.y, z = d.z;
    float Y[K_COEFFS];
    Y[0] = 0.28209479177387814;
    #if DEGREE >= 1
      Y[1] = -0.4886025119029199 * y;
      Y[2] =  0.4886025119029199 * z;
      Y[3] = -0.4886025119029199 * x;
    #endif
    #if DEGREE >= 2
      float xx = x*x, yy = y*y, zz = z*z, xy = x*y, yz = y*z, xz = x*z;
      Y[4] =  1.0925484305920792  * xy;
      Y[5] = -1.0925484305920792  * yz;
      Y[6] =  0.31539156525252005 * (2.0*zz - xx - yy);
      Y[7] = -1.0925484305920792  * xz;
      Y[8] =  0.5462742152960396  * (xx - yy);
    #endif
    #if DEGREE >= 3
      Y[9]  = -0.5900435899266435 * y * (3.0*xx - yy);
      Y[10] =  2.890611442640554  * xy * z;
      Y[11] = -0.4570457994644658 * y * (4.0*zz - xx - yy);
      Y[12] =  0.3731763325901154 * z * (2.0*zz - 3.0*xx - 3.0*yy);
      Y[13] = -0.4570457994644658 * x * (4.0*zz - xx - yy);
      Y[14] =  1.445305721320277  * z * (xx - yy);
      Y[15] = -0.5900435899266435 * x * (xx - 3.0*yy);
    #endif
    vec3 color = vec3(0.0);
    for (int k = 0; k < K_COEFFS; ++k){
        vec3 c = texture(u_coeffs, vec3(uv, float(k))).rgb;
        #ifdef SH_UNORM8
        c = u_sh_offset[k] + u_sh_scale[k] * c;
        #endif
        color += Y[k] * c;
    }
    return clamp(color, 0.0, 1.0);
}
"""
