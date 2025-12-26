"""
CPU matplotlib visualizer reading huge mmap binary (utils.load_data_binary),
but rendering ONLY the first simulation window you request:

You set:
  - --step (simulation time between frames, e.g. 1h)
  - --fps
  - --duration (seconds of the output GIF/MP4)

Then it renders exactly:
  frames = fps * duration
  simulated_time = frames * step
starting at t=0 by default (or --start).

It does NOT try to render the whole 60000 years.
It also does NOT require you to shrink the binary or guess "how many points to take".

Example:
  python visualizer_cpu_from_binary_limited.py --data ./assets/data_bin_60000 --out out.gif --step 1h --fps 120 --duration 20
  -> 2400 frames, 1 hour per frame => ~100 days of simulation rendered.

Notes:
  - Source data is typically daily dt=86400s. If you choose step < dt, we linearly interpolate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import time
import subprocess
import matplotlib.style as mplstyle

from utils import load_data_binary


_UNIT_SECONDS = {
    # seconds
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    # minutes
    "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    # hours
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    # days
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
    # years (Julian-ish)
    "y": 365.25 * 86400.0, "yr": 365.25 * 86400.0, "yrs": 365.25 * 86400.0,
    "year": 365.25 * 86400.0, "years": 365.25 * 86400.0,
}

def parse_timestr_to_seconds(s: str) -> float:
    """
    Examples: "1h", "0.5h", "20s", "2min", "10d", "600" (seconds)
    """
    s = str(s).strip().lower()
    if not s:
        raise ValueError("Empty time string")
    # plain number => seconds
    if all(c.isdigit() or c in ".+-eE" for c in s):
        return float(s)

    import re
    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([a-z]+)\s*", s)
    if not m:
        raise ValueError(f"Bad time format: {s!r}")
    val = float(m.group(1))
    unit = m.group(2)
    if unit not in _UNIT_SECONDS:
        raise ValueError(f"Unknown unit in {s!r}. Use one of: {sorted(set(_UNIT_SECONDS.keys()))}")
    return val * _UNIT_SECONDS[unit]

def format_sim_time(t_seconds: float) -> str:
    years = t_seconds / (365.25 * 86400.0)
    if abs(years) >= 1.0:
        if abs(years) >= 1000:
            return f"{years:,.0f} years".replace(",", " ")
        return f"{years:.2f} years"
    days = t_seconds / 86400.0
    if abs(days) >= 1.0:
        return f"{days:.2f} days"
    hours = t_seconds / 3600.0
    if abs(hours) >= 1.0:
        return f"{hours:.2f} hours"
    return f"{t_seconds:.2f} s"


@dataclass
class SampledPlanet:
    planet: object
    x: np.ndarray  # (frames,)
    y: np.ndarray  # (frames,)


def _sample_planet_xy(
    p: object,
    t_seconds: np.ndarray,
    input_dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample fixed-dt arrays at arbitrary times using linear interpolation.
    Works with memmap-backed arrays efficiently for a few thousand frames.
    """
    xsrc = p.path_x
    ysrc = p.path_y
    n = len(xsrc)

    idx = t_seconds / float(input_dt_seconds)
    idx = np.clip(idx, 0.0, float(n - 1))
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    f = (idx - i0.astype(np.float64)).astype(np.float32)

    x0 = xsrc[i0]
    y0 = ysrc[i0]
    x1 = xsrc[i1]
    y1 = ysrc[i1]

    x = (1.0 - f) * x0 + f * x1
    y = (1.0 - f) * y0 + f * y1
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def animate_orbits_from_binary_limited(
    data_folder: str,
    *,
    filename: str = "orbit.gif",
    fps: int = 30,
    duration_seconds: float = 10.0,
    step: str = "1d",
    start: str = "0s",
    input_dt_seconds: float = 86400.0,

    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,

    first_n: Optional[int] = None,

    trace_length: int = 200,

    figsize: tuple[float, float] = (10, 10),
    grid: bool = True,
    grid_alpha: float = 0.25,
    grid_style: str = "--",
    dpi: int = 150,
    title: Optional[str] = None,
    time_text: bool = True,

    # Saving options:
    gif_direct: bool = False,
    gif_fps: Optional[int] = None,
    gif_scale: Optional[float] = None,
    mp4_codec: str = "libx264",
    mp4_extra_args: Optional[list[str]] = None,
) -> None:
    planets = load_data_binary(data_folder, include=include, exclude=exclude)
    if not planets:
        raise ValueError("No planets loaded (check include/exclude)")

    if first_n is not None:
        first_n = int(first_n)
        if first_n <= 0:
            raise ValueError("--first-n must be a positive integer")
        planets = planets[:first_n]
        if not planets:
            raise ValueError("--first-n resulted in empty planet list")

    fps = int(fps)
    frames_req = int(round(float(duration_seconds) * fps))
    frames_req = max(2, frames_req)

    step_seconds = float(parse_timestr_to_seconds(step))
    start_seconds = float(parse_timestr_to_seconds(start))
    if step_seconds <= 0:
        raise ValueError("--step must be > 0")

    # Dataset length in seconds (assume all planets have same length)
    n_total = int(len(planets[0].path_x))
    total_sim_seconds = float(max(0, n_total - 1)) * float(input_dt_seconds)

    # Effective frames limited by dataset end (no "freezing" at the end).
    if start_seconds >= total_sim_seconds:
        raise ValueError(f"--start is beyond dataset length ({format_sim_time(total_sim_seconds)})")

    max_frames_possible = int(np.floor((total_sim_seconds - start_seconds) / step_seconds)) + 1
    frames = min(frames_req, max_frames_possible)
    if frames < 2:
        raise ValueError("Not enough data points for even 2 frames in the requested window.")

    # Time grid: ONLY the requested window.
    t_seconds = start_seconds + np.arange(frames, dtype=np.float64) * step_seconds

    # Pre-sample for all frames
    sampled: list[SampledPlanet] = []
    for p in planets:
        x, y = _sample_planet_xy(p, t_seconds, input_dt_seconds=float(input_dt_seconds))
        sampled.append(SampledPlanet(p, x, y))

    # Bounds from the sampled window only (fast and matches your request)
    max_range = 0.0
    for sp in sampled:
        max_range = max(max_range, float(np.max(np.abs(sp.x))), float(np.max(np.abs(sp.y))))
    limit = max_range * 1.1 if max_range > 0 else 1.0

    # Plot style (keep close to your original visualizer)
    mplstyle.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")

    if grid:
        ax.grid(True, alpha=grid_alpha, linestyle=grid_style)

    if title is None:
        title = "Orbit visualization"
    ax.set_title(title)

    # Time overlay (top-left)
    t_text_artist = None
    if time_text:
        t_text_artist = ax.text(
            0.02, 0.95, "",
            transform=ax.transAxes,
            fontsize=12,
            ha="left", va="top",
            color="white",
        )

    lines = []
    dots = []
    labels = []

    for sp in sampled:
        p = sp.planet
        color = getattr(p, "color", None) or "white"

        ln, = ax.plot([], [], color=color, lw=1, alpha=0.7)
        lines.append(ln)

        marker_size = 8 if str(getattr(p, "name", "")).lower() == "sun" else 5
        dot, = ax.plot([], [], "o", color=color, markersize=marker_size)
        dots.append(dot)

        lab = ax.text(0, 0, str(getattr(p, "name", "")), fontsize=9, color=color)
        labels.append(lab)

    def init():
        for ln, dot, lab in zip(lines, dots, labels):
            ln.set_data([], [])
            dot.set_data([], [])
            lab.set_text("")
        if t_text_artist is not None:
            t_text_artist.set_text("")
        artists = lines + dots + labels
        if t_text_artist is not None:
            artists.append(t_text_artist)
        return artists

    def update(fi: int):
        if t_text_artist is not None:
            # show relative time from simulation start by default (like "Time: ... years")
            t_rel = float(t_seconds[fi] - start_seconds)
            t_text_artist.set_text(f"Time: {format_sim_time(t_rel)}")

        for i, sp in enumerate(sampled):
            x = sp.x[fi]
            y = sp.y[fi]

            dots[i].set_data([x], [y])
            labels[i].set_position((x + limit * 0.02, y + limit * 0.02))
            labels[i].set_text(str(getattr(sp.planet, "name", "")))

            if trace_length == -1:
                start_i = 0
            elif trace_length <= 0:
                start_i = fi
            else:
                start_i = max(0, fi - int(trace_length))

            lines[i].set_data(sp.x[start_i:fi + 1], sp.y[start_i:fi + 1])

        artists = lines + dots + labels
        if t_text_artist is not None:
            artists.append(t_text_artist)
        return artists

    sim_span = (frames - 1) * step_seconds
    print(
        f"Rendering ONLY the first window: frames={frames}/{frames_req} @ {fps} fps, "
        f"duration={frames/fps:.3f}s (requested {duration_seconds}s), "
        f"sim_step={step_seconds:.3f}s, sim_span={format_sim_time(sim_span)}"
    )

    anim = FuncAnimation(
        fig,
        update,
        frames=range(frames),
        init_func=init,
        blit=True,
        interval=1000 / float(fps),
    )

    # --- Save with progress bar (manual frame loop) ---
    def _print_progress(i: int, total: int, t0: float) -> None:
        now = time.perf_counter()
        done = i + 1
        elapsed = now - t0
        fps_eff = done / elapsed if elapsed > 0 else 0.0
        rem = total - done
        eta = rem / fps_eff if fps_eff > 0 else 0.0
        bar_w = 30
        filled = int(bar_w * done / max(1, total))
        bar = "█" * filled + " " * (bar_w - filled)
        eta_m = int(eta // 60)
        eta_s = int(eta % 60)
        pct = 100.0 * done / max(1, total)
        print(f"\r[{bar}] {done:5d}/{total} ({pct:5.1f}%)  render {fps_eff:7.1f} fps  ETA {eta_m}:{eta_s:02d}", end="", flush=True)

    def _run(cmd: list[str]) -> None:
        subprocess.run(cmd, check=True)

    # IMPORTANT: --fps controls OUTPUT playback rate (timestamps), not how fast your CPU renders frames.
    expected_out_sec = frames / float(fps)
    print(f"[info] output playback: {frames} frames / {fps} fps = {expected_out_sec:.3f} seconds")

    if mp4_extra_args is None:
        mp4_extra_args = []

    try:
        t0_save = time.perf_counter()
        out_is_gif = filename.lower().endswith(".gif")
        out_is_mp4 = filename.lower().endswith(".mp4")

        if out_is_gif and (not gif_direct):
            # Render frames into a temp MP4 at the requested fps (timestamps accurate),
            # then convert MP4 -> GIF via ffmpeg with an explicit fps filter + palette.
            tmp_mp4 = filename[:-4] + ".__tmp__.mp4"
            writer = FFMpegWriter(fps=fps, codec=mp4_codec, extra_args=mp4_extra_args)
            with writer.saving(fig, tmp_mp4, dpi=int(dpi)):
                init()
                for i in range(frames):
                    update(i)
                    writer.grab_frame()
                    _print_progress(i, frames, t0_save)
            print()

            gif_fps_eff = int(gif_fps) if gif_fps is not None else int(min(fps, 60))
            if gif_scale is None:
                scale_expr = "scale=iw:ih:flags=lanczos"
            else:
                s = float(gif_scale)
                scale_expr = f"scale=trunc(iw*{s}/2)*2:trunc(ih*{s}/2)*2:flags=lanczos"

            palette = filename[:-4] + ".__palette__.png"
            _run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", tmp_mp4,
                "-vf", f"fps={gif_fps_eff},{scale_expr},palettegen",
                palette,
            ])
            _run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", tmp_mp4,
                "-i", palette,
                "-lavfi", f"fps={gif_fps_eff},{scale_expr}[x];[x][1:v]paletteuse",
                "-loop", "0",
                filename,
            ])

            try:
                os.remove(tmp_mp4)
            except OSError:
                pass
            try:
                os.remove(palette)
            except OSError:
                pass

            print(f"Saved to {filename} (GIF via ffmpeg @ {gif_fps_eff} fps)")

        elif out_is_mp4:
            writer = FFMpegWriter(fps=fps, codec=mp4_codec, extra_args=mp4_extra_args)
            with writer.saving(fig, filename, dpi=int(dpi)):
                init()
                for i in range(frames):
                    update(i)
                    writer.grab_frame()
                    _print_progress(i, frames, t0_save)
            print()
            print(f"Saved to {filename} (MP4 codec={mp4_codec})")

        else:
            # Direct GIF (PillowWriter) or other writer fallback
            writer = PillowWriter(fps=fps)
            with writer.saving(fig, filename, dpi=int(dpi)):
                init()
                for i in range(frames):
                    update(i)
                    writer.grab_frame()
                    _print_progress(i, frames, t0_save)
            print()
            print(f"Saved to {filename}")

    except Exception as e:
        print()
        print(f"Error saving animation: {e}. Showing plot instead.")
        plt.show()


def _split_csv(s: Optional[str]) -> Optional[list[str]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Matplotlib orbit animator (CPU) reading mmap binary, rendering ONLY a requested time window."
    )
    ap.add_argument("--data", required=True, help="Path to binary folder (e.g. ./assets/data_bin_60000)")
    ap.add_argument("--out", default="orbit.gif", help="Output file (.gif or .mp4)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=10.0, help="Output duration in seconds (e.g. 20)")
    ap.add_argument("--step", required=True, help='Simulation step between frames (e.g. "1h"). REQUIRED.')
    ap.add_argument("--start", default="0s", help='Simulation start offset (e.g. "0s", "10d"). Default: 0s.')
    ap.add_argument("--input-dt", type=float, default=86400.0, help="Source dt in seconds (default: 86400)")
    ap.add_argument("--include", default=None, help='Comma-separated names to include (e.g. "Sun,Mercury,Earth")')
    ap.add_argument("--exclude", default=None, help='Comma-separated names to exclude')
    ap.add_argument("--first-n", type=int, default=None, help="Render only the first N planets after include/exclude")
    ap.add_argument("--trace-length", type=int, default=200, help="Tail length in frames (-1 = full)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-grid", action="store_true", help="Disable grid")
    ap.add_argument("--gif-direct", action="store_true", help="Save GIF directly via PillowWriter (some viewers clamp high fps). Default: render MP4 then convert to GIF via ffmpeg palette.")
    ap.add_argument("--gif-fps", type=int, default=None, help="FPS for final GIF when using ffmpeg conversion (default: min(--fps, 60)).")
    ap.add_argument("--gif-scale", type=float, default=None, help="Scale factor for GIF via ffmpeg (e.g. 0.5). Default: no scaling.")
    ap.add_argument("--mp4-codec", default="libx264", help="FFmpeg codec for MP4 saving (e.g. libx264 or h264_nvenc if available).")
    ap.add_argument("--mp4-extra", default="", help='Extra ffmpeg args for MP4 writer, comma-separated (e.g. "-preset,fast,-crf,18" or "-rc,vbr,-cq,19").')
    args = ap.parse_args()

    mp4_extra_args = []
    if args.mp4_extra:
        mp4_extra_args = [t for t in args.mp4_extra.split(",") if t]

    animate_orbits_from_binary_limited(
        args.data,
        filename=args.out,
        fps=args.fps,
        duration_seconds=float(args.duration),
        step=args.step,
        start=args.start,
        input_dt_seconds=float(args.input_dt),
        include=_split_csv(args.include),
        exclude=_split_csv(args.exclude),
        first_n=args.first_n,
        trace_length=int(args.trace_length),
        grid=(not args.no_grid),
        dpi=int(args.dpi),
        gif_direct=bool(args.gif_direct),
        gif_fps=args.gif_fps,
        gif_scale=args.gif_scale,
        mp4_codec=str(args.mp4_codec),
        mp4_extra_args=mp4_extra_args,
    )


if __name__ == "__main__":
    main()
