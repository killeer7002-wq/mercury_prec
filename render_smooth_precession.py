import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from numba import njit

from consts import AU, GM
from utils import load_data_binary


@njit(fastmath=True)
def calculate_averaged_orbital_vector(rx_slice: np.ndarray, ry_slice: np.ndarray, rz_slice: np.ndarray,
                                      vx_slice: np.ndarray, vy_slice: np.ndarray, vz_slice: np.ndarray,
                                      gm: float) -> tuple[float, float]:
    """
    Вычисляет средний вектор эксцентриситета по набору точек траектории.

    Это ключевая функция для сглаживания: вместо вычисления мгновенного
    угла перигелия, она усредняет компоненты вектора эксцентриситета
    по заданному "окну" данных. Это позволяет получить более стабильное
    направление вектора, убирая осцилляции.

    Args:
        rx_slice, ry_slice, rz_slice (np.ndarray): Срезы координат x, y, z.
        vx_slice, vy_slice, vz_slice (np.ndarray): Срезы компонент скорости.
        gm (float): Гравитационный параметр центрального тела (G * M).

    Returns:
        tuple[float, float]: Усредненные компоненты вектора эксцентриситета (e_x, e_y).
    """
    n_points = len(rx_slice)
    sum_ex = 0.0
    sum_ey = 0.0

    for i in range(n_points):
        r = np.array([rx_slice[i], ry_slice[i], rz_slice[i]])
        v = np.array([vx_slice[i], vy_slice[i], vz_slice[i]])

        r_mag = np.sqrt(np.sum(r**2))
        if r_mag == 0: continue

        # Вектор углового момента h = r x v
        h_vec = np.cross(r, v)

        # Компоненты вектора эксцентриситета e = (v x h) / GM - r / |r|
        e_vec = np.cross(v, h_vec) / gm - r / r_mag

        sum_ex += e_vec[0]
        sum_ey += e_vec[1]

    # Возвращаем средние компоненты вектора
    return sum_ex / n_points, sum_ey / n_points


@njit(fastmath=True)
def calculate_initial_shape_params(r: np.ndarray, v: np.ndarray, gm: float) -> tuple[float, float]:
    """
    Вычисляет большую полуось и эксцентриситет для одной точки.

    Вспомогательная функция для определения осредненной формы орбиты
    на начальном этапе симуляции.

     Args:
        r (np.ndarray): Вектор положения (3D).
        v (np.ndarray): Вектор скорости (3D).
        gm (float): Гравитационный параметр центрального тела (G * M).

    Returns:
        tuple[float, float]: Кортеж (большая полуось, эксцентриситет).
    """
    r_mag = np.sqrt(np.sum(r**2))
    v_mag_sq = np.sum(v**2)

    h_vec = np.cross(r, v)
    e_vec = np.cross(v, h_vec) / gm - r / r_mag
    e_mag = np.sqrt(np.sum(e_vec**2))

    specific_energy = v_mag_sq / 2.0 - gm / r_mag
    a = -gm / (2.0 * specific_energy) if specific_energy != 0 else 0

    return a, e_mag


def create_animation(folder_name: str, dt_save: float):
    """
    Создает сглаженную анимацию прецессии орбиты Меркурия.

    Эта версия использует две техники для получения плавной визуализации:
    1.  Форма эллипса (большая полуось и эксцентриситет) вычисляется один
        раз как среднее значение за первый год симуляции и остается постоянной.
        Это предотвращает "пульсацию" эллипса.
    2.  Угол перигелия вычисляется на основе усредненного по временному
        окну вектора эксцентриситета, что устраняет дрожание.

    Args:
        folder_name (str): Имя папки в 'assets/', содержащей данные симуляции.
        dt_save (float): Шаг по времени (в секундах) из симуляции.
    """
    path = f"assets/{folder_name}"
    print(f"Загрузка данных из {path}...")
    try:
        planets = load_data_binary(path)
    except FileNotFoundError:
        print(f"Ошибка: Директория {path} не найдена.")
        return

    mercury = next(p for p in planets if p.name == "Mercury")

    rx, ry, rz = np.array(mercury.path_x), np.array(mercury.path_y), np.array(mercury.path_z)
    vx, vy, vz = np.array(mercury.path_vx), np.array(mercury.path_vy), np.array(mercury.path_vz)
    total_points = len(rx)

    # 1. Фиксация формы эллипса на основе средних параметров за первый год
    print("Расчет осредненной формы орбиты...")
    points_per_year = int((365.25 * 24 * 3600) / dt_save)
    init_window = min(total_points, points_per_year)
    num_samples = 100
    
    a_sum, e_sum = 0.0, 0.0
    for i in range(0, init_window, max(1, init_window // num_samples)):
        r_vec = np.array([rx[i], ry[i], rz[i]])
        v_vec = np.array([vx[i], vy[i], vz[i]])
        a_i, e_i = calculate_initial_shape_params(r_vec, v_vec, GM)
        a_sum += a_i
        e_sum += e_i

    A_AVG = a_sum / num_samples
    E_AVG = e_sum / num_samples

    ellipse_width = 2 * A_AVG
    ellipse_height = 2 * A_AVG * np.sqrt(1 - E_AVG**2)
    focus_dist = A_AVG * E_AVG
    rp_avg = A_AVG * (1 - E_AVG)
    print(f"Фиксированные параметры: a={A_AVG/AU:.3f} AU, e={E_AVG:.4f}")

    # 2. Настройка параметров анимации и окна сглаживания
    target_fps = 30
    video_duration = 15
    total_frames = target_fps * video_duration
    stride = max(1, total_points // total_frames)

    # Окно сглаживания для угла (например, 2 года)
    smooth_window_years = 2.0
    points_to_smooth = int((smooth_window_years * 365.25 * 24 * 3600) / dt_save)
    smooth_window = min(points_to_smooth, stride)
    print(f"Окно сглаживания: {smooth_window} точек ({smooth_window * dt_save / (24*3600*365.25):.1f} лет)")

    # 3. Подготовка графика
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    limit = A_AVG * (1 + E_AVG) * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_title("Сглаженная прецессия орбиты Меркурия")

    ax.plot(0, 0, '*', color='yellow', markersize=15, zorder=10)
    line_perihelion, = ax.plot([], [], color='red', lw=2)
    orbit_patch = Ellipse((0, 0), width=ellipse_width, height=ellipse_height, angle=0,
                          edgecolor='cyan', facecolor='none', lw=1.5, alpha=0.8)
    ax.add_patch(orbit_patch)

    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)
    angle_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, color='cyan', fontsize=12)

    def init():
        line_perihelion.set_data([], [])
        return line_perihelion, orbit_patch, time_text, angle_text

    def update(frame_idx: int):
        center_idx = frame_idx * stride
        if center_idx >= total_points: center_idx = total_points - 1

        # Определение границ окна для сглаживания
        start_idx = max(0, center_idx - smooth_window // 2)
        end_idx = min(total_points, start_idx + smooth_window)
        if start_idx >= end_idx: end_idx = start_idx + 1
        
        # Вычисление сглаженного угла на основе среднего вектора эксцентриситета
        avg_ex, avg_ey = calculate_averaged_orbital_vector(
            rx[start_idx:end_idx], ry[start_idx:end_idx], rz[start_idx:end_idx],
            vx[start_idx:end_idx], vy[start_idx:end_idx], vz[start_idx:end_idx], GM
        )
        angle_rad_smooth = np.arctan2(avg_ey, avg_ex)

        # Обновление графики с использованием осредненных/фиксированных значений
        cx = -focus_dist * np.cos(angle_rad_smooth)
        cy = -focus_dist * np.sin(angle_rad_smooth)
        orbit_patch.center = (cx, cy)
        orbit_patch.angle = np.degrees(angle_rad_smooth)

        px = rp_avg * np.cos(angle_rad_smooth)
        py = rp_avg * np.sin(angle_rad_smooth)
        line_perihelion.set_data([0, px], [0, py])

        years = center_idx * dt_save / (365.25 * 24 * 3600)
        degrees = np.degrees(angle_rad_smooth)
        time_text.set_text(f'Время: {years:.0f} лет')
        angle_text.set_text(f'Угол перигелия: {degrees:.2f}°')

        return line_perihelion, orbit_patch, time_text, angle_text

    print("Создание сглаженной анимации...")
    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=40)

    out_file = f"assets/precession_smooth_{folder_name}.gif"
    print(f"Сохранение в файл {out_file}...")
    try:
        anim.save(out_file, writer='pillow', fps=target_fps)
        print("Готово!")
    except Exception as e:
        print(f"Ошибка при сохранении: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        default_folder = "data_bin_60000"
        if os.path.exists(f"assets/{default_folder}"):
            print(f"Папка с данными не указана. Используется папка по умолчанию: '{default_folder}', dt=3600s")
            create_animation(default_folder, 3600.0)
        else:
            print("Использование: python render_smooth_precession.py <ПАПКА_С_ДАННЫМИ> [ШАГ_В_СЕКУНДАХ]")
            sys.exit(1)
    else:
        folder = sys.argv[1]
        dt_val = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
        create_animation(folder, dt_val)
