import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import sys
import os
from numba import njit

# Импорты из вашего проекта
from utils import load_data_binary
from consts import GM, M_SUN, AU

# --- НОВАЯ ФУНКЦИЯ ДЛЯ СГЛАЖИВАНИЯ ---
@njit(fastmath=True)
def calculate_averaged_orbital_vector(rx_slice, ry_slice, rz_slice, vx_slice, vy_slice, vz_slice, GM):
    """
    Вычисляет СРЕДНИЙ вектор эксцентриситета (ex, ey) по набору точек.
    Это нужно для сглаживания дёрганий угла.
    """
    n_points = len(rx_slice)
    sum_ex = 0.0
    sum_ey = 0.0
    
    for i in range(n_points):
        r = np.array([rx_slice[i], ry_slice[i], rz_slice[i]])
        v = np.array([vx_slice[i], vy_slice[i], vz_slice[i]])
        
        r_mag = np.sqrt(np.sum(r**2))
        
        # Вектор углового момента h = r x v
        hx = r[1]*v[2] - r[2]*v[1]
        hy = r[2]*v[0] - r[0]*v[2]
        hz = r[0]*v[1] - r[1]*v[0]
        
        # Компоненты вектора эксцентриситета e = (v x h) / GM - r / |r|
        vxh_x = v[1]*hz - v[2]*hy
        vxh_y = v[2]*hx - v[0]*hz
        
        ex = vxh_x / GM - r[0] / r_mag
        ey = vxh_y / GM - r[1] / r_mag
        
        sum_ex += ex
        sum_ey += ey
        
    # Возвращаем средние компоненты вектора
    return sum_ex / n_points, sum_ey / n_points

@njit(fastmath=True)
def calculate_initial_shape_params(r, v, GM):
    """Считает a и e для одной точки (для инициализации средних)"""
    r_mag = np.sqrt(np.sum(r**2))
    v_mag_sq = np.sum(v**2)
    
    hx = r[1]*v[2] - r[2]*v[1]
    hy = r[2]*v[0] - r[0]*v[2]
    hz = r[0]*v[1] - r[1]*v[0]
    
    vxh_x = v[1]*hz - v[2]*hy
    vxh_y = v[2]*hx - v[0]*hz
    
    ex = vxh_x / GM - r[0] / r_mag
    ey = vxh_y / GM - r[1] / r_mag
    e_mag = np.sqrt(ex**2 + ey**2)
    
    specific_energy = v_mag_sq / 2.0 - GM / r_mag
    a = -GM / (2.0 * specific_energy)
    
    return a, e_mag

def create_animation(folder_name, dt_save):
    # 1. Загрузка данных
    path = f"assets/{folder_name}"
    print(f"Loading data from {path}...")
    try:
        planets = load_data_binary(path)
    except FileNotFoundError:
        print(f"Error: Folder {path} not found.")
        return

    mercury = next(p for p in planets if p.name == "Mercury")
    
    rx = np.array(mercury.path_x)
    ry = np.array(mercury.path_y)
    rz = np.array(mercury.path_z)
    vx = np.array(mercury.path_vx)
    vy = np.array(mercury.path_vy)
    vz = np.array(mercury.path_vz)
    
    total_points = len(rx)
    
    # --- ФИКС 1: Фиксация формы эллипса ---
    print("Calculating average orbital shape...")
    # Берем точки за первый "год" (примерно) для оценки средних a и e
    points_per_year = int((365.25 * 24 * 3600) / dt_save)
    init_window = min(total_points, points_per_year)
    
    a_sum = 0
    e_sum = 0
    for i in range(0, init_window, max(1, init_window//100)): # Пробегаем 100 точек за год
        r_vec = np.array([rx[i], ry[i], rz[i]])
        v_vec = np.array([vx[i], vy[i], vz[i]])
        a_i, e_i = calculate_initial_shape_params(r_vec, v_vec, GM)
        a_sum += a_i
        e_sum += e_i
        
    A_AVG = a_sum / 100
    E_AVG = e_sum / 100
    
    # Параметры для рисования фиксированного эллипса
    ellipse_width = 2 * A_AVG
    ellipse_height = 2 * A_AVG * np.sqrt(1 - E_AVG**2)
    # Расстояние от центра до фокуса
    focus_dist = A_AVG * E_AVG
    # Длина линии перигелия
    rp_avg = A_AVG * (1 - E_AVG)

    print(f"Fixed parameters: a={A_AVG/AU:.3f} AU, e={E_AVG:.4f}")

    # 2. Настройка анимации
    target_fps = 30
    video_duration = 15 # секунд
    total_frames = target_fps * video_duration
    stride = max(1, total_points // total_frames)
    
    # --- ФИКС 2: Окно сглаживания для угла ---
    # Для каждого кадра будем усреднять вектор по окну в 2 года (например)
    smooth_window_years = 2.0
    points_to_smooth = int((smooth_window_years * 365.25 * 24 * 3600) / dt_save)
    # Окно не должно быть больше шага между кадрами, иначе нет смысла
    smooth_window = min(points_to_smooth, stride)
    print(f"Smoothing window: {smooth_window} points ({smooth_window*dt_save/(24*3600*365.25):.1f} years)")

    # 3. Подготовка графика
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    limit = A_AVG * (1 + E_AVG) * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    ax.plot(0, 0, '*', color='yellow', markersize=15, zorder=10)
    line_perihelion, = ax.plot([], [], color='red', lw=2, label='Perihelion Vector')
    
    # Используем ФИКСИРОВАННЫЕ размеры
    orbit_patch = Ellipse((0, 0), width=ellipse_width, height=ellipse_height, angle=0, 
                          edgecolor='cyan', facecolor='none', lw=1.5, alpha=0.8)
    ax.add_patch(orbit_patch)
    
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)
    angle_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, color='cyan', fontsize=12)

    def init():
        line_perihelion.set_data([], [])
        return line_perihelion, orbit_patch, time_text, angle_text

    def update(frame_idx):
        # Центральный индекс кадра
        center_idx = frame_idx * stride
        if center_idx >= total_points: center_idx = total_points - 1
        
        # --- СГЛАЖИВАНИЕ УГЛА ---
        # Определяем границы окна вокруг центральной точки
        start_idx = max(0, center_idx - smooth_window // 2)
        end_idx = min(total_points, center_idx + smooth_window // 2)
        
        # Берем срезы данных (Numpy делает это эффективно через memory map)
        # Если окно слишком маленькое (<2 точек), берем хотя бы 1 точку
        if end_idx - start_idx < 2:
             start_idx = center_idx
             end_idx = min(total_points, center_idx + 1)

        rx_slice = rx[start_idx:end_idx]
        ry_slice = ry[start_idx:end_idx]
        rz_slice = rz[start_idx:end_idx]
        vx_slice = vx[start_idx:end_idx]
        vy_slice = vy[start_idx:end_idx]
        vz_slice = vz[start_idx:end_idx]
        
        # Считаем СРЕДНИЙ вектор эксцентриситета через Numba
        avg_ex, avg_ey = calculate_averaged_orbital_vector(
            rx_slice, ry_slice, rz_slice, 
            vx_slice, vy_slice, vz_slice, GM
        )
        
        # Вычисляем сглаженный угол
        angle_rad_smooth = np.arctan2(avg_ey, avg_ex)
        
        # --- ОБНОВЛЕНИЕ ГРАФИКИ (используя средний угол и фикс. размеры) ---
        
        # 1. Поворот и смещение эллипса
        cx = -focus_dist * np.cos(angle_rad_smooth)
        cy = -focus_dist * np.sin(angle_rad_smooth)
        
        orbit_patch.center = (cx, cy)
        # width и height уже заданы и не меняются!
        orbit_patch.angle = np.degrees(angle_rad_smooth)
        
        # 2. Поворот линии перигелия
        px = rp_avg * np.cos(angle_rad_smooth)
        py = rp_avg * np.sin(angle_rad_smooth)
        line_perihelion.set_data([0, px], [0, py])
        
        # 3. Текст
        years = center_idx * dt_save / (365.25 * 24 * 3600) 
        degrees = np.degrees(angle_rad_smooth)
        time_text.set_text(f'Time: {years:.0f} years')
        angle_text.set_text(f'Perihelion Angle: {degrees:.2f}°')
        
        return line_perihelion, orbit_patch, time_text, angle_text

    print("Generating animation with smoothing...")
    # Уменьшим interval для более плавной гифки
    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=40)
    
    out_file = f"assets/precession_smooth_{folder_name}.gif"
    print(f"Saving to {out_file}...")
    try:
        anim.save(out_file, writer='pillow', fps=target_fps)
        print("Done!")
    except Exception as e:
        print(f"Error saving: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        default_folder = "data_bin"
        if os.path.exists(f"assets/{default_folder}"):
             print(f"Using default folder: {default_folder}, dt=3600s")
             create_animation(default_folder, 3600.0)
        else:
            print("Usage: python render_precession.py <DATA_FOLDER_NAME> [DT_SECONDS]")
    else:
        folder = sys.argv[1]
        dt_val = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
        create_animation(folder, dt_val)