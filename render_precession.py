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

# Настройка для FFmpeg (если есть), иначе будет GIF через Pillow
# plt.rcParams['animation.ffmpeg_path'] = 'путь_к_ffmpeg' 

@njit(fastmath=True)
def calculate_orbital_elements(r, v, GM):
    """
    Вычисляет параметры орбиты для визуализации:
    - a (большая полуось)
    - e (эксцентриситет)
    - angle (угол перигелия)
    """
    # Расстояние и скорость
    r_mag = np.sqrt(np.sum(r**2))
    v_mag_sq = np.sum(v**2)
    
    # Вектор удельного углового момента h = r x v
    # (упрощенно для 2D плоскости x-y, так как орбита почти плоская)
    hx = r[1]*v[2] - r[2]*v[1]
    hy = r[2]*v[0] - r[0]*v[2]
    hz = r[0]*v[1] - r[1]*v[0]
    # h_mag_sq = hx**2 + hy**2 + hz**2
    
    # Вектор эксцентриситета e = (v x h) / GM - r / |r|
    # Считаем компоненты (v x h)
    vxh_x = v[1]*hz - v[2]*hy
    vxh_y = v[2]*hx - v[0]*hz
    # vxh_z = v[0]*hy - v[1]*hx
    
    ex = vxh_x / GM - r[0] / r_mag
    ey = vxh_y / GM - r[1] / r_mag
    
    e_mag = np.sqrt(ex**2 + ey**2)
    
    # Большая полуось a = -GM / 2E
    specific_energy = v_mag_sq / 2.0 - GM / r_mag
    a = -GM / (2.0 * specific_energy)
    
    # Угол перигелия (аргумент перицентра + долгота узла в 2D)
    angle = np.arctan2(ey, ex)
    
    return a, e_mag, angle

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
    
    # Используем memory mapping
    rx = np.array(mercury.path_x)
    ry = np.array(mercury.path_y)
    rz = np.array(mercury.path_z)
    vx = np.array(mercury.path_vx)
    vy = np.array(mercury.path_vy)
    vz = np.array(mercury.path_vz)
    
    total_points = len(rx)
    print(f"Total history points: {total_points:,}")
    print(f"Time step per frame (dt): {dt_save} s")

    # 2. Настройка анимации
    # Мы хотим видео длительностью ~10-15 секунд @ 30 FPS
    target_fps = 30
    video_duration = 15 # секунд
    total_frames = target_fps * video_duration
    
    stride = max(1, total_points // total_frames)
    
    # Рассчитываем, сколько лет проходит за 1 кадр анимации
    years_per_frame = (stride * dt_save) / (24*3600*365.25)
    
    print(f"Rendering setup:")
    print(f"  - Output frames: {total_frames}")
    print(f"  - Stride: {stride} (skipping points)")
    print(f"  - Sim. speed: {years_per_frame:.2f} years/frame")

    # 3. Подготовка графика
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Лимиты (чуть больше орбиты Меркурия)
    limit = 0.45 * AU * 1.1 
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Элементы графика
    # Солнце
    ax.plot(0, 0, '*', color='yellow', markersize=15, zorder=10)
    
    # Линия к перигелию (будет обновляться)
    line_perihelion, = ax.plot([], [], color='red', lw=2, label='Perihelion Vector')
    
    # Эллипс орбиты (будет обновляться)
    orbit_patch = Ellipse((0, 0), width=0, height=0, angle=0, 
                          edgecolor='cyan', facecolor='none', lw=1.5, alpha=0.8)
    ax.add_patch(orbit_patch)
    
    # Текст
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)
    angle_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, color='cyan', fontsize=12)

    def init():
        line_perihelion.set_data([], [])
        return line_perihelion, orbit_patch, time_text, angle_text

    def update(frame_idx):
        # Индекс в реальном массиве
        idx = frame_idx * stride
        if idx >= total_points: idx = total_points - 1
        
        # Берем векторы
        r = np.array([rx[idx], ry[idx], rz[idx]])
        v = np.array([vx[idx], vy[idx], vz[idx]])
        
        # Считаем параметры орбиты (Мгновенно через Numba)
        a, e, angle_rad = calculate_orbital_elements(r, v, GM)
        
        # --- Рисуем Эллипс ---
        # Центр эллипса смещен от фокуса (Солнца) на величину c = a * e
        # Смещение идет вдоль большой оси (по углу angle_rad)
        cx = -a * e * np.cos(angle_rad)
        cy = -a * e * np.sin(angle_rad)
        
        orbit_patch.center = (cx, cy)
        orbit_patch.width = 2 * a
        orbit_patch.height = 2 * a * np.sqrt(1 - e**2)
        orbit_patch.angle = np.degrees(angle_rad)
        
        # --- Рисуем Линию к Перигелию ---
        # Расстояние в перигелии: rp = a(1-e)
        rp = a * (1 - e)
        px = rp * np.cos(angle_rad)
        py = rp * np.sin(angle_rad)
        
        line_perihelion.set_data([0, px], [0, py])
        
        # --- Текст ---
        # Используем переданный dt_save для расчета времени
        years = idx * dt_save / (365.25 * 24 * 3600) 
        degrees = np.degrees(angle_rad)
        
        time_text.set_text(f'Time: {years:.0f} years')
        angle_text.set_text(f'Perihelion Angle: {degrees:.2f}°')
        
        return line_perihelion, orbit_patch, time_text, angle_text

    print("Generating animation...")
    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=50)
    
    out_file = f"assets/precession_{folder_name}.gif"
    print(f"Saving to {out_file}...")
    try:
        anim.save(out_file, writer='pillow', fps=target_fps)
        print("Done!")
    except Exception as e:
        print(f"Error saving: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Если аргументов нет, пробуем дефолтные значения
        default_folder = "data_bin"
        if os.path.exists(f"assets/{default_folder}"):
             print(f"Using default folder: {default_folder}, dt=3600s")
             create_animation(default_folder, 3600.0)
        else:
            print("Usage: python render_precession.py <DATA_FOLDER_NAME> [DT_SECONDS]")
    else:
        folder = sys.argv[1]
        # Если второй аргумент есть - берем его, иначе 3600
        dt_val = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
        create_animation(folder, dt_val)