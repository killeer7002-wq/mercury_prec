from utils import load_data_binary
from visualizer_gpu import render_orbits_nvenc

planets = load_data_binary("./assets/data_bin_60000")
planets = [p for p in planets if p.name in ("Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")]
planets = planets[:5]

render_orbits_nvenc(
    planets,
    "./assets/mercury.mp4",
    fps=120,
    days_per_frame=3650 * 2,            # 10 лет/кадр, но внутри рисуются ВСЕ дневные точки интервала
    trail_half_life_seconds=1.0,    # исчезает заметно (поиграй 2..8)
    trail_blend="max",              # ВАЖНО: bounded trail (без "взрыва" яркости)
    line_alpha=0.03,                # тоньше/слабее линия
    halo_alpha=0.35,
    halo_size=10.0,
    point_size=1.5,
    rgba_order="BGRA",              # если цвета не вернулись/странные -> "BGRA"
    line_width=0.1,
    point_alpha=1.0,
)
