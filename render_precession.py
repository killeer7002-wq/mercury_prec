import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from numba import njit

from consts import AU, GM
from utils import load_data_binary

# Для сохранения в MP4 вместо GIF раскомментируйте и укажите путь к FFmpeg
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'


@njit(fastmath=True)
def calculate_orbital_elements(r: np.ndarray, v: np.ndarray, gm: float) -> tuple[float, float, float]:
    """
    Вычисляет мгновенные оскулирующие орбитальные элементы.

    На основе векторов положения (r) и скорости (v) тела относительно
    центрального массивного объекта (с гравитационным параметром gm)
    рассчитывает ключевые параметры эллиптической орбиты.

    Args:
        r (np.ndarray): Вектор положения (3D).
        v (np.ndarray): Вектор скорости (3D).
        gm (float): Гравитационный параметр центрального тела (G * M).

    Returns:
        tuple[float, float, float]: Кортеж, содержащий:
        - a (float): большая полуось эллипса.
        - e (float): эксцентриситет (скаляр).
        - angle (float): угол (аргумент) перигелия в радианах.
    """
    r_mag = np.sqrt(np.sum(r**2))
    v_mag_sq = np.sum(v**2)

    # Вектор удельного углового момента: h = r x v
    h_vec = np.cross(r, v)

    # Вектор эксцентриситета: e = (v x h) / GM - r / |r|
    e_vec = np.cross(v, h_vec) / gm - r / r_mag
    e_mag = np.sqrt(np.sum(e_vec**2))

    # Большая полуось: a = -GM / 2E, где E - удельная орбитальная энергия
    specific_energy = v_mag_sq / 2.0 - gm / r_mag
    a = -gm / (2.0 * specific_energy)

    # Угол перигелия - направление вектора эксцентриситета в плоскости XY
    angle = np.arctan2(e_vec[1], e_vec[0])

    return a, e_mag, angle


def create_animation(folder_name: str, dt_save: float):
    """
    Создает анимацию прецессии орбиты Меркурия.

    Загружает данные траектории, вычисляет изменение орбитальных элементов
    с течением времени и визуализирует его в виде GIF-анимации, где
    поворачивающийся эллипс и вектор показывают смещение перигелия.

    Args:
        folder_name (str): Имя папки в 'assets/', содержащей данные симуляции.
        dt_save (float): Шаг по времени (в секундах), с которым сохранялись
                         данные в симуляции.
    """
    data_path = f"assets/{folder_name}"
    print(f"Загрузка данных из {data_path}...")
    try:
        planets = load_data_binary(data_path)
    except FileNotFoundError:
        print(f"Ошибка: Директория '{data_path}' не найдена.")
        return

    mercury = next(p for p in planets if p.name == "Mercury")

    # Использование numpy-массивов для эффективного доступа к данным
    rx = np.array(mercury.path_x)
    ry = np.array(mercury.path_y)
    rz = np.array(mercury.path_z)
    vx = np.array(mercury.path_vx)
    vy = np.array(mercury.path_vy)
    vz = np.array(mercury.path_vz)

    total_points = len(rx)
    print(f"Всего точек в траектории: {total_points:,}")
    print(f"Шаг симуляции (dt): {dt_save} с")

    # Настройка параметров рендеринга для получения видео ~15 секунд при 30 FPS
    target_fps = 30
    video_duration_sec = 15
    total_frames = target_fps * video_duration_sec

    # Вычисляем шаг (stride) для прореживания данных, чтобы уложиться в нужное число кадров
    stride = max(1, total_points // total_frames)
    years_per_frame = (stride * dt_save) / (24 * 3600 * 365.25)

    print("Параметры рендеринга:")
    print(f"  - Итоговое количество кадров: {total_frames}")
    print(f"  - Шаг по данным (stride): {stride} точек на кадр")
    print(f"  - Скорость симуляции: {years_per_frame:.2f} лет/кадр")

    # Настройка графики
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))

    limit = 0.45 * AU * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_title("Прецессия орбиты Меркурия")

    # Графические элементы
    ax.plot(0, 0, '*', color='yellow', markersize=15, zorder=10, label="Солнце")
    line_perihelion, = ax.plot([], [], color='red', lw=2, label='Вектор перигелия')
    orbit_patch = Ellipse((0, 0), width=0, height=0, angle=0,
                          edgecolor='cyan', facecolor='none', lw=1.5, alpha=0.8,
                          label='Оскулирующая орбита')
    ax.add_patch(orbit_patch)
    ax.legend()

    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)
    angle_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, color='cyan', fontsize=12)

    def init():
        """Инициализация анимации."""
        line_perihelion.set_data([], [])
        return line_perihelion, orbit_patch, time_text, angle_text

    def update(frame_idx: int):
        """Обновление кадра анимации."""
        idx = frame_idx * stride
        if idx >= total_points:
            idx = total_points - 1

        r = np.array([rx[idx], ry[idx], rz[idx]])
        v = np.array([vx[idx], vy[idx], vz[idx]])

        # Вычисляем мгновенные параметры орбиты
        a, e, angle_rad = calculate_orbital_elements(r, v, GM)

        # Обновление эллипса орбиты
        # Центр эллипса смещен от фокуса (Солнца) на расстояние c = a * e
        cx = -a * e * np.cos(angle_rad)
        cy = -a * e * np.sin(angle_rad)

        orbit_patch.center = (cx, cy)
        orbit_patch.width = 2 * a
        orbit_patch.height = 2 * a * np.sqrt(1 - e**2)
        orbit_patch.angle = np.degrees(angle_rad)

        # Обновление линии, указывающей на перигелий
        # Расстояние до перигелия: rp = a * (1 - e)
        rp = a * (1 - e)
        px = rp * np.cos(angle_rad)
        py = rp * np.sin(angle_rad)
        line_perihelion.set_data([0, px], [0, py])

        # Обновление текста с информацией
        years = idx * dt_save / (365.25 * 24 * 3600)
        degrees = np.degrees(angle_rad)
        time_text.set_text(f'Время: {years:.0f} лет')
        angle_text.set_text(f'Угол перигелия: {degrees:.2f}°')

        return line_perihelion, orbit_patch, time_text, angle_text

    print("Создание анимации...")
    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=50)

    out_file = f"assets/precession_{folder_name}.gif"
    print(f"Сохранение в файл {out_file}...")
    try:
        anim.save(out_file, writer='pillow', fps=target_fps)
        print("Готово!")
    except Exception as e:
        print(f"Ошибка при сохранении: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # При запуске без аргументов используется папка и dt по умолчанию
        default_folder = "data_bin"
        if os.path.exists(f"assets/{default_folder}"):
            print(f"Папка с данными не указана. Используется папка по умолчанию: '{default_folder}', dt=3600s")
            create_animation(default_folder, 3600.0)
        else:
            print("Использование: python render_precession.py <ПАПКА_С_ДАННЫМИ> [ШАГ_В_СЕКУНДАХ]")
            sys.exit(1)
    else:
        folder = sys.argv[1]
        dt_val = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
        create_animation(folder, dt_val)
