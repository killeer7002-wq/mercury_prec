# visualizer_gpu.py
# GPU-only NV12 pipeline with nice look:
# - Fading thin trail (accumulated in texture with exponential decay)
# - Bright planet point + soft halo (NOT accumulated)
# - Proper NV12 CAI planes (UV as (H/2, W/2, 2)) => colors work
#
# EGL/OpenGL -> trail ping-pong -> compose final -> async PBO -> CUDA map
# -> RGBA->NV12 kernel (NVRTC) -> NVENC (PyNvVideoCodec device NV12) -> mp4 mux
#
# deps:
#   pip install moderngl cuda-python PyNvVideoCodec numpy
# system:
#   sudo apt install ffmpeg

from __future__ import annotations

import os
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import Sequence, Optional, Tuple

import numpy as np


# ----------------- helpers -----------------

def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def _ffmpeg_mux_h264_to_mp4(h264_path: str, mp4_path: str, fps: int) -> None:
    _run([
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-r", str(int(fps)),
        "-f", "h264",
        "-i", h264_path,
        "-c", "copy",
        mp4_path,
    ])

def _color_to_rgb01(color) -> tuple[float, float, float]:
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16) / 255.0
        g = int(color[3:5], 16) / 255.0
        b = int(color[5:7], 16) / 255.0
        return (r, g, b)
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        r, g, b = float(color[0]), float(color[1]), float(color[2])
        if max(r, g, b) > 1.0:
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        return (r, g, b)
    return (1.0, 1.0, 1.0)

def _palette_by_name(name: str) -> tuple[float, float, float]:
    n = name.lower()
    # приятная палитра "как в астрономии"
    if n == "sun":     return (1.0, 0.9, 0.2)
    if n == "mercury": return (0.85, 0.85, 0.85)
    if n == "venus":   return (1.0, 0.65, 0.25)
    if n == "earth":   return (0.35, 0.65, 1.0)
    if n == "mars":    return (1.0, 0.35, 0.25)
    if n == "jupiter": return (0.95, 0.8, 0.65)
    if n == "saturn":  return (0.95, 0.9, 0.7)
    if n == "uranus":  return (0.55, 0.9, 0.95)
    if n == "neptune": return (0.35, 0.55, 1.0)
    return (1.0, 1.0, 1.0)

def _fmt_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600); seconds -= 3600 * h
    m = int(seconds // 60);   seconds -= 60 * m
    s = int(seconds)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

def _enum(obj, *names: str, default: int = 0) -> int:
    for n in names:
        if hasattr(obj, n):
            return int(getattr(obj, n))
    return int(default)

def _trail_decay_from_half_life(fps: int, half_life_seconds: float) -> float:
    # after T seconds, intensity halves => decay = 0.5 ** (1/(fps*T))
    if half_life_seconds <= 0:
        return 1.0
    return float(0.5 ** (1.0 / (fps * half_life_seconds)))


# ----------------- CUDA Array Interface wrappers -----------------

class _AppCAI:
    def __init__(self, shape, strides, typestr: str, ptr: int):
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "strides": tuple(strides),
            "typestr": typestr,
            "data": (int(ptr), False),
            "version": 2,
        }

class _NV12Frame:
    """
    NV12 as two CAI planes.
    IMPORTANT for your PyNvVideoCodec build:
      - pitch/stride must equal width
      - UV plane should be represented as (H/2, W, 1) with stride (W,1,1)
        (interleaved UV bytes along width)
    """
    def __init__(self, W: int, H: int, base_ptr: int):
        self.W = int(W); self.H = int(H)
        self.pitch = int(W)     # MUST equal width
        self.base_ptr = int(base_ptr)

        y_ptr = self.base_ptr
        uv_ptr = self.base_ptr + self.pitch * self.H

        self._cai = [
            _AppCAI((self.H, self.W, 1), (self.pitch, 1, 1), "|u1", y_ptr),
            _AppCAI((self.H // 2, self.W // 2, 2), (self.pitch, 2, 1), "|u1", uv_ptr),
        ]

    def cuda(self):
        return self._cai


@dataclass
class _PlanetDraw:
    name: str
    color_rgb: tuple[float, float, float]
    seg_vbo: object
    seg_vao: object
    pt_vbo: object
    pt_vao: object
    tmp_seg: np.ndarray      # (seg_max,2) float32
    tmp_pt: np.ndarray       # (1,2) float32


# ----------------- NVRTC compile + kernel launch -----------------

def _compile_nvrtc_rgba_to_nv12(cu, nvrtc, device: int):
    """
    RGBA/BGRA -> NV12 (Y plane + interleaved UV plane).
    We include bgra flag to handle driver readback channel order.
    """
    cuda_src = r"""
    extern "C" __global__
    void rgba_to_nv12_full(
        const unsigned char* __restrict__ rgba,
        int pitch_rgba,
        unsigned char* __restrict__ y_plane,
        unsigned char* __restrict__ uv_plane,
        int width,
        int height,
        int flip_y,
        int bgra   // 0: rgba, 1: bgra
    ) {
        int ux = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int uy = (int)(blockIdx.y * blockDim.y + threadIdx.y);

        int x = ux * 2;
        int y = uy * 2;
        if (x + 1 >= width || y + 1 >= height) return;

        int sy0 = flip_y ? (height - 1 - y)     : y;
        int sy1 = flip_y ? (height - 1 - (y+1)) : (y+1);

        const unsigned char* row0 = rgba + sy0 * pitch_rgba;
        const unsigned char* row1 = rgba + sy1 * pitch_rgba;

        int i00 = (x    ) * 4;
        int i01 = (x + 1) * 4;

        // load 4 pixels, resolve channel order
        auto load_rgb = [&](const unsigned char* row, int idx, float &r, float &g, float &b) {
            unsigned char c0 = row[idx + 0];
            unsigned char c1 = row[idx + 1];
            unsigned char c2 = row[idx + 2];
            if (bgra) { b = (float)c0; g = (float)c1; r = (float)c2; }
            else      { r = (float)c0; g = (float)c1; b = (float)c2; }
        };

        float r00,g00,b00,r01,g01,b01,r10,g10,b10,r11,g11,b11;
        load_rgb(row0, i00, r00,g00,b00);
        load_rgb(row0, i01, r01,g01,b01);
        load_rgb(row1, i00, r10,g10,b10);
        load_rgb(row1, i01, r11,g11,b11);

        // ADD: alpha
        float a00 = row0[i00 + 3];
        float a01 = row0[i01 + 3];
        float a10 = row1[i00 + 3];
        float a11 = row1[i01 + 3];

        auto clamp_u8 = [](float v) -> unsigned char {
            v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
            return (unsigned char)(v + 0.5f);
        };

        // Full-range BT.709
        auto Yf = [](float r, float g, float b) -> float { return 0.2126f*r + 0.7152f*g + 0.0722f*b; };
        auto Uf = [](float r, float g, float b) -> float { return -0.1146f*r - 0.3854f*g + 0.5000f*b + 128.0f; };
        auto Vf = [](float r, float g, float b) -> float { return  0.5000f*r - 0.4542f*g - 0.0458f*b + 128.0f; };

        // Y (premultiplied OK — fading works naturally)
        unsigned char y00 = clamp_u8(Yf(r00,g00,b00));
        unsigned char y01 = clamp_u8(Yf(r01,g01,b01));
        unsigned char y10 = clamp_u8(Yf(r10,g10,b10));
        unsigned char y11 = clamp_u8(Yf(r11,g11,b11));

        unsigned char* yrow0 = y_plane + (y    ) * width;
        unsigned char* yrow1 = y_plane + (y + 1) * width;
        yrow0[x    ] = y00; yrow0[x + 1] = y01;
        yrow1[x    ] = y10; yrow1[x + 1] = y11;

        // --- REPLACED UV BLOCK: alpha-weighted chroma ---
        float sa = a00 + a01 + a10 + a11;

        float r, g, b;
        if (sa > 1e-3f) {
            float inv = 255.0f / sa;
            r = (r00 + r01 + r10 + r11) * inv;
            g = (g00 + g01 + g10 + g11) * inv;
            b = (b00 + b01 + b10 + b11) * inv;
        } else {
            r = g = b = 0.0f;
        }

        unsigned char U = 128, V = 128;
        if (sa > 1e-3f) {
            U = clamp_u8(Uf(r,g,b));
            V = clamp_u8(Vf(r,g,b));
        }

        unsigned char* uvrow = uv_plane + uy * width;
        int uvx = ux * 2;
        uvrow[uvx + 0] = U;
        uvrow[uvx + 1] = V;
    }
    """

    def nvrtc_ok(res):
        err, *rest = res
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            try:
                _, msg = nvrtc.nvrtcGetErrorString(err)
                raise RuntimeError(msg.decode())
            except Exception:
                raise RuntimeError(f"NVRTC error code: {err}")
        return rest[0] if len(rest) == 1 else rest

    def cu_ok(res):
        # supports both (err,...) and direct return
        if isinstance(res, tuple) and len(res) > 0 and isinstance(res[0], (int, np.integer)) and int(res[0]) < 10000:
            err, *rest = res
            if err != cu.CUresult.CUDA_SUCCESS:
                _, s = cu.cuGetErrorString(err)
                raise RuntimeError(f"CUDA error: {s.decode()}")
            if len(rest) == 0:
                return None
            if len(rest) == 1:
                return rest[0]
            return rest
        return res

    dev = cu_ok(cu.cuDeviceGet(int(device)))
    maj = int(cu_ok(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)))
    minu = int(cu_ok(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)))

    device_arch = f"compute_{maj}{minu}"
    env_arch = os.environ.get("NVRTC_ARCH")

    arch_candidates = []
    if env_arch:
        arch_candidates.append(env_arch)
    arch_candidates.append(device_arch)
    arch_candidates += ["compute_90", "compute_89", "compute_86", "compute_80", "compute_75", "compute_70", "compute_61", "compute_60", "compute_52"]

    seen = set()
    arch_candidates = [a for a in arch_candidates if not (a in seen or seen.add(a))]

    prog = nvrtc_ok(nvrtc.nvrtcCreateProgram(cuda_src.encode(), b"rgba_to_nv12.cu", 0, [], []))

    last_log = ""
    chosen_arch = None
    for arch in arch_candidates:
        opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++11"]
        try:
            nvrtc_ok(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
            chosen_arch = arch
            break
        except Exception:
            try:
                log_sz = int(nvrtc_ok(nvrtc.nvrtcGetProgramLogSize(prog)))
                log_buf = bytearray(log_sz)
                nvrtc.nvrtcGetProgramLog(prog, log_buf)
                last_log = log_buf.decode(errors="replace")
            except Exception:
                last_log = "NVRTC compile failed (no log)"
            low = last_log.lower()
            if "invalid value for --gpu-architecture" in low or "invalid option" in low:
                continue
            raise RuntimeError("NVRTC compile failed:\n" + last_log)

    if chosen_arch is None:
        raise RuntimeError(
            "NVRTC could not find supported --gpu-architecture.\n"
            "Try: NVRTC_ARCH=compute_90 (or compute_86)\n"
            f"Last log:\n{last_log}"
        )

    print(f"[NVRTC] compiled PTX with --gpu-architecture={chosen_arch} (device CC={device_arch})")

    ptx_sz = int(nvrtc_ok(nvrtc.nvrtcGetPTXSize(prog)))
    ptx = bytearray(ptx_sz)
    nvrtc_ok(nvrtc.nvrtcGetPTX(prog, ptx))

    module = cu_ok(cu.cuModuleLoadData(bytes(ptx)))
    func = cu_ok(cu.cuModuleGetFunction(module, b"rgba_to_nv12_full"))
    return module, func


def _launch_kernel(cu, func, grid, block, stream, args: list[np.ndarray]):
    arg_ptrs = np.array([a.ctypes.data for a in args], dtype=np.uint64)
    gx, gy, gz = grid
    bx, by, bz = block
    err, = cu.cuLaunchKernel(
        func,
        int(gx), int(gy), int(gz),
        int(bx), int(by), int(bz),
        0, int(stream),
        int(arg_ptrs.ctypes.data),
        0
    )
    if err != cu.CUresult.CUDA_SUCCESS:
        _, s = cu.cuGetErrorString(err)
        raise RuntimeError(f"cuLaunchKernel failed: {s.decode()}")


# ----------------- main renderer -----------------

def render_orbits_nvenc(
    planets: Sequence[object],
    filename: str = "./assets/orbit.mp4",
    *,
    fps: int = 60,
    days_per_frame: int = 3650,             # interval size in days per output video frame
    resolution: tuple[int, int] = (1920, 1080),
    backend: str = "egl",
    gpuid: int = 0,

    trail_half_life_seconds: float = 6.0,   # smaller => faster fade (e.g. 3.0)
    trail_blend: str = "max",               # "max" bounded (recommended) or "add" additive
    trail_gain: float = 25.0,               # boosts trail visibility when using max blend
    line_alpha: float = 0.05,               # thin trail intensity
    line_width: float = 1.0,
    point_size: float = 5.0,
    halo_size: float = 13.0,
    halo_alpha: float = 0.25,
    point_alpha: float = 1.0,
    exposure: float = 1.6,
    gamma: float = 2.2,

    # technical:
    flip_y: bool = True,
    rgba_order: str = "RGBA",               # if colors look wrong, try "BGRA"
    bounds_stride: int = 4000,
    progress_every_sec: float = 0.25,
    encoder_params: Optional[dict] = None,
) -> None:
    if not planets:
        raise ValueError("planets is empty")
    if days_per_frame < 1:
        raise ValueError("days_per_frame must be >= 1")

    W, H = map(int, resolution)
    if (W % 2) or (H % 2):
        raise ValueError("NV12 requires even resolution (e.g. 1920x1080).")

    import moderngl
    import PyNvVideoCodec as nvc
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    encoder_params = dict(encoder_params or {})

    # ---------- OpenGL ----------
    ctx = moderngl.create_context(standalone=True, backend=backend, require=330)
    print(f"[GL] vendor={ctx.info.get('GL_VENDOR','?')} renderer={ctx.info.get('GL_RENDERER','?')}")
    ctx.enable(moderngl.BLEND)

    # ---------- CUDA helpers ----------
    def cu_ok(res):
        if isinstance(res, tuple) and len(res) > 0 and isinstance(res[0], (int, np.integer)) and int(res[0]) < 10000:
            err, *rest = res
            if err != cu.CUresult.CUDA_SUCCESS:
                _, s = cu.cuGetErrorString(err)
                raise RuntimeError(f"CUDA error: {s.decode()}")
            if len(rest) == 0:
                return None
            if len(rest) == 1:
                return rest[0]
            return rest
        return res

    cu_ok(cu.cuInit(0))
    dev = cu_ok(cu.cuDeviceGet(int(gpuid)))
    cu_ctx = cu_ok(cu.cuDevicePrimaryCtxRetain(dev))
    cu_ok(cu.cuCtxSetCurrent(cu_ctx))
    stream = 0

    # ---------- NVRTC kernel ----------
    _, k_rgba_to_nv12 = _compile_nvrtc_rgba_to_nv12(cu, nvrtc, device=int(gpuid))

    # ---------- Bounds estimate ----------
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for p in planets:
        xs = np.asarray(p.path_x[::bounds_stride], dtype=np.float64)
        ys = np.asarray(p.path_y[::bounds_stride], dtype=np.float64)
        xmin = min(xmin, float(xs.min())); xmax = max(xmax, float(xs.max()))
        ymin = min(ymin, float(ys.min())); ymax = max(ymax, float(ys.max()))

    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    half = max(xmax - xmin, ymax - ymin) * 0.5
    half = max(half, 1e-12)

    # correct aspect compensation:
    s = 0.95 / half
    aspect = W / float(H)
    scale_x = s / aspect
    scale_y = s

    # ---------- Shaders ----------
    # line program (simple)
    line_prog = ctx.program(
        vertex_shader=r"""
            #version 330
            in vec2 in_pos;
            uniform vec2 u_center;
            uniform vec2 u_scale;
            void main() {
                vec2 p = (in_pos - u_center) * u_scale;
                gl_Position = vec4(p, 0.0, 1.0);
            }
        """,
        fragment_shader=r"""
            #version 330
            uniform vec3 u_color;
            uniform float u_alpha;
            out vec4 fragColor;
            void main() { fragColor = vec4(u_color * u_alpha, 1.0); }
        """,
    )
    line_prog["u_center"].value = (float(cx), float(cy))
    line_prog["u_scale"].value = (float(scale_x), float(scale_y))

    # point program (circular + smooth halo via gl_PointCoord)
    point_prog = ctx.program(
        vertex_shader=r"""
            #version 330
            in vec2 in_pos;
            uniform vec2 u_center;
            uniform vec2 u_scale;
            uniform float u_point_size;
            void main() {
                vec2 p = (in_pos - u_center) * u_scale;
                gl_Position = vec4(p, 0.0, 1.0);
                gl_PointSize = u_point_size;
            }
        """,
        fragment_shader=r"""
            #version 330
            uniform vec3 u_color;
            uniform float u_alpha;
            out vec4 fragColor;
            void main() {
                vec2 c = gl_PointCoord - vec2(0.5);
                float d = length(c);
                // soft circle: 1 at center -> 0 at edge
                float a = smoothstep(0.5, 0.0, d);
                float A = u_alpha * a;
                fragColor = vec4(u_color * A, A);
            }
        """,
    )
    point_prog["u_center"].value = (float(cx), float(cy))
    point_prog["u_scale"].value = (float(scale_x), float(scale_y))

    # decay blit (trail = trail * decay)
    blit_prog = ctx.program(
        vertex_shader=r"""
            #version 330
            in vec2 in_pos;
            out vec2 v_uv;
            void main() {
                v_uv = (in_pos + 1.0) * 0.5;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
        """,
        fragment_shader=r"""
            #version 330
            uniform sampler2D u_tex;
            uniform float u_decay;
            in vec2 v_uv;
            out vec4 fragColor;
            void main() {
                fragColor = texture(u_tex, v_uv) * u_decay;
            }
        """,
    )
    tonemap_prog = ctx.program(
        vertex_shader=r"""
            #version 330
            in vec2 in_pos;
            out vec2 v_uv;
            void main() {
                v_uv = (in_pos + 1.0) * 0.5;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
        """,
        fragment_shader=r"""
            #version 330
            uniform sampler2D u_tex;
            uniform float u_exposure;   // 1.0..3.0
            uniform float u_gamma;      // 2.2 usually
            uniform float u_gain;       // extra gain for trails (useful with MAX blend)
            in vec2 v_uv;
            out vec4 fragColor;

            void main() {
                vec4 t = texture(u_tex, v_uv); // HDR stored in trail (RGB = intensity)
                vec3 c = t.rgb * u_gain;

                // tonemap (keep black background)
                c = vec3(1.0) - exp(-u_exposure * c);

                // gamma for display
                c = pow(c, vec3(1.0 / u_gamma));

                // output alpha=1 for NV12 conversion stability
                fragColor = vec4(c, 1.0);
            }
        """,
    )

    quad_vbo = ctx.buffer(np.array([[-1,-1],[1,-1],[-1,1],[1,1]], dtype=np.float32).tobytes())
    quad_vao = ctx.vertex_array(blit_prog, [(quad_vbo, "2f", "in_pos")])
    tonemap_vao = ctx.vertex_array(tonemap_prog, [(quad_vbo, "2f", "in_pos")])

    # HDR trail (float16), final stays RGBA8
    trail_a = ctx.texture((W, H), components=4, dtype="f2")
    trail_b = ctx.texture((W, H), components=4, dtype="f2")
    final_tex = ctx.texture((W, H), components=4, dtype="f1")  # for PBO readback / NV12

    fbo_a = ctx.framebuffer(color_attachments=[trail_a])
    fbo_b = ctx.framebuffer(color_attachments=[trail_b])
    fbo_final = ctx.framebuffer(color_attachments=[final_tex])

    # init clear
    fbo_a.use(); fbo_a.clear(0.0, 0.0, 0.0, 0.0)
    fbo_b.use(); fbo_b.clear(0.0, 0.0, 0.0, 0.0)

    prev_tex, next_tex = trail_a, trail_b
    prev_fbo, next_fbo = fbo_a, fbo_b

    # ---------- Per-frame segment buffers ----------
    n_total = int(len(planets[0].path_x))
    step = int(days_per_frame)
    seg_max = step + 1

    planets_draw: list[_PlanetDraw] = []
    for p in planets:
        name = str(getattr(p, "name", "planet"))
        c = getattr(p, "color", None)
        color = _color_to_rgb01(c) if c is not None else _palette_by_name(name)
        # if user didn't set color (often white), enforce palette for visibility
        if color == (1.0, 1.0, 1.0):
            color = _palette_by_name(name)

        seg_vbo = ctx.buffer(reserve=seg_max * 2 * 4, dynamic=True)
        seg_vao = ctx.vertex_array(line_prog, [(seg_vbo, "2f", "in_pos")])

        pt_vbo = ctx.buffer(reserve=2 * 4, dynamic=True)
        pt_vao = ctx.vertex_array(point_prog, [(pt_vbo, "2f", "in_pos")])

        planets_draw.append(_PlanetDraw(
            name=name,
            color_rgb=color,
            seg_vbo=seg_vbo,
            seg_vao=seg_vao,
            pt_vbo=pt_vbo,
            pt_vao=pt_vao,
            tmp_seg=np.empty((seg_max, 2), dtype=np.float32),
            tmp_pt=np.empty((1, 2), dtype=np.float32),
        ))

    # ---------- PBO double buffer + fences ----------
    bytes_rgba = W * H * 4
    pbo = [ctx.buffer(reserve=bytes_rgba, dynamic=True), ctx.buffer(reserve=bytes_rgba, dynamic=True)]
    fences = [None, None]

    def insert_fence(i: int):
        try:
            fences[i] = ctx.fence()
        except Exception:
            fences[i] = None

    def wait_fence(i: int):
        f = fences[i]
        if f is None:
            ctx.finish()
        else:
            f.wait()

    # ---------- CUDA-GL interop: register PBOs ----------
    flags = _enum(
        cu.CUgraphicsMapResourceFlags,
        "CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY",
        "CU_GRAPHICS_MAP_RESOURCE_FLAGS_READONLY",
        default=0,
    )

    def register_gl_buffer(gl_buffer_id: int):
        return cu_ok(cu.cuGraphicsGLRegisterBuffer(int(gl_buffer_id), int(flags)))

    cu_res = [register_gl_buffer(pbo[0].glo), register_gl_buffer(pbo[1].glo)]

    def map_resource(res_handle):
        try:
            cu_ok(cu.cuGraphicsMapResources(1, res_handle, stream))
        except TypeError:
            cu_ok(cu.cuGraphicsMapResources(1, [res_handle], stream))

    def unmap_resource(res_handle):
        try:
            cu_ok(cu.cuGraphicsUnmapResources(1, res_handle, stream))
        except TypeError:
            cu_ok(cu.cuGraphicsUnmapResources(1, [res_handle], stream))

    def get_mapped_ptr(res_handle) -> Tuple[int, int]:
        out = cu_ok(cu.cuGraphicsResourceGetMappedPointer(res_handle))
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return int(out[0]), int(out[1])
        raise RuntimeError(f"Unexpected cuGraphicsResourceGetMappedPointer return: {out}")

    # ---------- NV12 device buffer (pitch == width) ----------
    nv12_size = W * H + W * (H // 2)
    nv12_base = int(cu_ok(cu.cuMemAlloc(int(nv12_size))))
    nv12_frame = _NV12Frame(W=W, H=H, base_ptr=nv12_base)

    # ---------- NVENC encoder (NV12 device) ----------
    encoder = None
    last_err = None
    for attempt in range(6):
        try:
            if attempt == 0:
                encoder = nvc.CreateEncoder(W, H, "NV12", False, gpuid=int(gpuid), **encoder_params)
            elif attempt == 1:
                encoder = nvc.CreateEncoder(W, H, "NV12", 0, gpuid=int(gpuid), **encoder_params)
            elif attempt == 2:
                encoder = nvc.CreateEncoder(W, H, "NV12", False, **encoder_params)
            elif attempt == 3:
                encoder = nvc.CreateEncoder(W, H, "NV12", 0, **encoder_params)
            elif attempt == 4:
                encoder = nvc.CreateEncoder(width=W, height=H, fmt="NV12", usecpuinputbuffer=False, gpuid=int(gpuid), **encoder_params)
            else:
                encoder = nvc.CreateEncoder(width=W, height=H, fmt="NV12", usecpuinputbuffer=False, **encoder_params)
            last_err = None
            break
        except Exception as e:
            last_err = e
            encoder = None
    if encoder is None:
        raise RuntimeError(f"CreateEncoder NV12(device) failed. Last error: {last_err}")

    # ---------- Output paths ----------
    abs_out = os.path.abspath(filename)
    out_dir = os.path.dirname(abs_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    base, _ext = os.path.splitext(abs_out)
    tmp_h264 = base + ".h264"
    print(f"[encode] -> {abs_out} (temp bitstream: {tmp_h264})")

    def encode_nv12(out_fh):
        bs = encoder.Encode(nv12_frame)  # your build wants object with .cuda()
        if bs:
            out_fh.write(bytearray(bs))

    # ---------- Rendering schedule ----------
    n_frames = (max(1, n_total - 1) + step - 1) // step
    print(f"[render] frames={n_frames} (days_per_frame={step})")

    # ---------- Visual settings ----------
    ctx.line_width = float(line_width)
    decay = _trail_decay_from_half_life(fps, float(trail_half_life_seconds))

    # Line blending: additive to make trails crisp on black
    # Points: halo additive, core normal (so it stays vivid)
    bgra = 1 if rgba_order.upper() == "BGRA" else 0

    # ---------- Kernel launch config ----------
    block = (16, 16, 1)
    grid = ((W // 2 + block[0] - 1) // block[0], (H // 2 + block[1] - 1) // block[1], 1)
    pitch_rgba = W * 4
    flip = 1 if flip_y else 0

    # ---------- Progress ----------
    t0 = time.perf_counter()
    last_print = t0
    bar_w = 30

    def print_progress(done_frames: int):
        nonlocal last_print
        now = time.perf_counter()
        if now - last_print < progress_every_sec and done_frames < n_frames:
            return
        last_print = now
        elapsed = now - t0
        fps_eff = done_frames / elapsed if elapsed > 0 else 0.0
        rem = n_frames - done_frames
        eta = rem / fps_eff if fps_eff > 0 else 0.0
        frac = done_frames / n_frames
        filled = int(frac * bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)
        sys.stdout.write(
            f"\r[{bar}] {done_frames:>6}/{n_frames} ({frac*100:5.1f}%)  "
            f"{fps_eff:7.1f} fps  ETA {_fmt_eta(eta)}"
        )
        sys.stdout.flush()

    # ---------- Main loop ----------
    try:
        with open(tmp_h264, "wb") as fh:
            for fi in range(n_frames):
                t_start = fi * step
                t_end = min((fi + 1) * step, n_total - 1)
                seg_len = t_end - t_start + 1

                # 1) next_trail = prev_trail * decay
                next_fbo.use()
                # overwrite copy (avoid leftover blending state)
                try:
                    ctx.blend_equation = moderngl.FUNC_ADD
                except Exception:
                    try:
                        ctx.blend_equation = 0x8006  # GL_FUNC_ADD
                    except Exception:
                        pass
                ctx.blend_func = (moderngl.ONE, moderngl.ZERO)
                blit_prog["u_decay"].value = float(decay)
                prev_tex.use(location=0)
                blit_prog["u_tex"].value = 0
                quad_vao.render(mode=moderngl.TRIANGLE_STRIP)

                # 2) draw thin segment INTO trail (accumulated)
                if seg_len >= 2:
                    # bounded trail update prevents brightness blow-up when orbit re-hits same pixels many times
                    tb = (trail_blend or "max").lower().strip()
                    if tb == "max":
                        # MAX blend equation: dst = max(dst, src) per component
                        try:
                            ctx.blend_equation = moderngl.MAX
                        except Exception:
                            try:
                                ctx.blend_equation = 0x8008  # GL_MAX
                            except Exception:
                                # fallback to additive if MAX unsupported
                                ctx.blend_equation = moderngl.FUNC_ADD
                        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
                    else:
                        try:
                            ctx.blend_equation = moderngl.FUNC_ADD
                        except Exception:
                            try:
                                ctx.blend_equation = 0x8006  # GL_FUNC_ADD
                            except Exception:
                                pass
                        ctx.blend_func = (moderngl.ONE, moderngl.ONE)  # additive

                    line_prog["u_alpha"].value = float(line_alpha)

                    for pd, p in zip(planets_draw, planets):
                        tmp = pd.tmp_seg
                        # fast, no temporary allocations:
                        np.copyto(tmp[:seg_len, 0], p.path_x[t_start:t_end + 1], casting="unsafe")
                        np.copyto(tmp[:seg_len, 1], p.path_y[t_start:t_end + 1], casting="unsafe")
                        pd.seg_vbo.write(tmp[:seg_len].tobytes())

                        line_prog["u_color"].value = pd.color_rgb
                        pd.seg_vao.render(mode=moderngl.LINE_STRIP, first=0, vertices=seg_len)

                # 3) compose FINAL frame:
                # reset blend equation to normal add for composition
                try:
                    ctx.blend_equation = moderngl.FUNC_ADD
                except Exception:
                    try:
                        ctx.blend_equation = 0x8006  # GL_FUNC_ADD
                    except Exception:
                        pass
                # final = trail (no decay) + bright points on top (not accumulated)
                fbo_final.use()
                fbo_final.clear(0.0, 0.0, 0.0, 0.0)

                # tone-map HDR trail -> final RGBA8
                ctx.blend_func = (moderngl.ONE, moderngl.ZERO)  # overwrite
                tonemap_prog["u_exposure"].value = float(exposure)
                tonemap_prog["u_gamma"].value = float(gamma)

                # boost trail visibility when using MAX blend (since it doesn't accumulate)
                _tb = (trail_blend or "max").lower().strip()
                _gain = float(trail_gain) if _tb == "max" else 1.0
                tonemap_prog["u_gain"].value = _gain
                next_tex.use(location=0)
                tonemap_prog["u_tex"].value = 0
                tonemap_vao.render(mode=moderngl.TRIANGLE_STRIP)

                # draw points (halo then core)
                if seg_len >= 1:
                    for pd, p in zip(planets_draw, planets):
                        # current position = last sample in segment
                        x = float(p.path_x[t_end])
                        y = float(p.path_y[t_end])
                        pd.tmp_pt[0, 0] = x
                        pd.tmp_pt[0, 1] = y
                        pd.pt_vbo.write(pd.tmp_pt.tobytes())

                        # halo (additive)
                        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
                        point_prog["u_color"].value = pd.color_rgb
                        point_prog["u_point_size"].value = float(halo_size if pd.name.lower() != "sun" else halo_size * 1.2)
                        point_prog["u_alpha"].value = float(halo_alpha)
                        pd.pt_vao.render(mode=moderngl.POINTS, first=0, vertices=1)

                        # core (normal alpha blend)
                        ctx.blend_func = (moderngl.ONE, moderngl.ONE)  # additive; point shader encodes coverage in RGB
                        point_prog["u_point_size"].value = float(point_size if pd.name.lower() != "sun" else point_size * 1.25)
                        point_prog["u_alpha"].value = float(point_alpha)
                        pd.pt_vao.render(mode=moderngl.POINTS, first=0, vertices=1)

                # 4) async read final -> PBO[cur]
                cur = fi & 1
                prev = cur ^ 1
                fbo_final.read_into(pbo[cur], components=4, alignment=1)
                try:
                    fences[cur] = ctx.fence()
                except Exception:
                    fences[cur] = None

                # 5) encode previous PBO while GPU continues
                if fi > 0:
                    if fences[prev] is None:
                        ctx.finish()
                    else:
                        fences[prev].wait()

                    res = cu_res[prev]
                    map_resource(res)
                    rgba_ptr, rgba_size = get_mapped_ptr(res)
                    if rgba_size < bytes_rgba:
                        unmap_resource(res)
                        raise RuntimeError(f"Mapped PBO too small: {rgba_size} < {bytes_rgba}")

                    args = [
                        np.array([np.uint64(rgba_ptr)], dtype=np.uint64),
                        np.array([np.int32(pitch_rgba)], dtype=np.int32),
                        np.array([np.uint64(nv12_base)], dtype=np.uint64),
                        np.array([np.uint64(nv12_base + W * H)], dtype=np.uint64),
                        np.array([np.int32(W)], dtype=np.int32),
                        np.array([np.int32(H)], dtype=np.int32),
                        np.array([np.int32(flip)], dtype=np.int32),
                        np.array([np.int32(bgra)], dtype=np.int32),
                    ]
                    _launch_kernel(cu, k_rgba_to_nv12, grid, block, stream, args)
                    encode_nv12(fh)
                    unmap_resource(res)

                # swap trail ping-pong
                prev_tex, next_tex = next_tex, prev_tex
                prev_fbo, next_fbo = next_fbo, prev_fbo

                print_progress(fi + 1)

            # encode last pending PBO
            if n_frames > 0:
                last = (n_frames - 1) & 1
                if fences[last] is None:
                    ctx.finish()
                else:
                    fences[last].wait()

                res = cu_res[last]
                map_resource(res)
                rgba_ptr, rgba_size = get_mapped_ptr(res)
                if rgba_size < bytes_rgba:
                    unmap_resource(res)
                    raise RuntimeError(f"Mapped PBO too small: {rgba_size} < {bytes_rgba}")

                args = [
                    np.array([np.uint64(rgba_ptr)], dtype=np.uint64),
                    np.array([np.int32(pitch_rgba)], dtype=np.int32),
                    np.array([np.uint64(nv12_base)], dtype=np.uint64),
                    np.array([np.uint64(nv12_base + W * H)], dtype=np.uint64),
                    np.array([np.int32(W)], dtype=np.int32),
                    np.array([np.int32(H)], dtype=np.int32),
                    np.array([np.int32(flip)], dtype=np.int32),
                    np.array([np.int32(bgra)], dtype=np.int32),
                ]
                _launch_kernel(cu, k_rgba_to_nv12, grid, block, stream, args)
                encode_nv12(fh)
                unmap_resource(res)

            # flush encoder
            try:
                tail = encoder.EndEncode()
                if tail:
                    fh.write(bytearray(tail))
            except Exception:
                pass

        sys.stdout.write("\n")
        if abs_out.lower().endswith(".mp4"):
            _ffmpeg_mux_h264_to_mp4(tmp_h264, abs_out, fps=fps)

        t1 = time.perf_counter()
        elapsed = t1 - t0
        print(f"[done] {abs_out}")
        print(f"[speed] {n_frames/elapsed:.1f} output-frames/s, elapsed {elapsed:.2f}s")
        print(f"[look] trail half-life={trail_half_life_seconds}s (decay={decay:.6f}), line_alpha={line_alpha}")
        if rgba_order.upper() == "RGBA":
            print("[tip] If colors still look off, try rgba_order='BGRA' in render_orbits_nvenc()")

    finally:
        for r in cu_res:
            try:
                cu_ok(cu.cuGraphicsUnregisterResource(r))
            except Exception:
                pass
        try:
            cu_ok(cu.cuMemFree(int(nv12_base)))
        except Exception:
            pass
