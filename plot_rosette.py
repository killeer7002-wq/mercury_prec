import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from utils import load_data_binary
from consts import * # G, M_SUN и другие константы

# --- ФИКС 1: Лечим OverflowError для миллионов точек ---
plt.rcParams['agg.path.chunksize'] = 10000 

# Доопределяем константы, которых нет в consts.py
# Большая полуось Меркурия (для расчета средней угловой скорости)
A_MERCURY = 0.387098 * AU 

def plot_rosette(folder_name, dt_input):
    # 1. Загрузка данных
    path = f"assets/{folder_name}"
    print(f"Loading data from {path}...")
    try:
        planets = load_data_binary(path)
    except FileNotFoundError:
        print(f"Error: Folder {path} not found.")
        return

    mercury = next(p for p in planets if p.name == "Mercury")
    
    # 2. Подготовка данных с прореживанием
    # Загружаем полные массивы (memory mapping работает быстро)
    x_full = np.array(mercury.path_x)
    y_full = np.array(mercury.path_y)
    
    total_points = len(x_full)
    print(f"Total points available: {total_points:,}")

    # --- ФИКС 2: Прореживание (Downsampling) ---
    # Ограничиваем график 1 миллионом точек, чтобы не убить matplotlib
    target_points = 1_000_000 
    step = max(1, total_points // target_points)
    
    if step > 1:
        print(f"Downsampling active: using every {step}-th point (plotting ~{total_points//step:,} pts)...")
    
    x = x_full[::step]
    y = y_full[::step]
    
    # 3. Расчет времени и вращения
    # Средняя угловая скорость (Mean Motion)
    n = np.sqrt(GM / A_MERCURY**3)
    print(f"Mercury Mean Motion: {n:.2e} rad/s")
    
    # Массив времени с учетом прореживания и введенного dt
    # t = индексы * шаг_прореживания * шаг_времени_симуляции
    t = np.arange(len(x)) * step * dt_input
    
    print("Calculating rotating frame coordinates...")
    
    # Угол поворота системы
    theta = n * t
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Поворот координат (x', y')
    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t
    
    # 4. Рисование
    print("Plotting...")
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 12), dpi=150)
    
    # Рисуем линию (cyan)
    plt.plot(x_rot, y_rot, color='cyan', lw=0.3, alpha=0.4)
    
    # Рисуем Солнце
    plt.plot(0, 0, '*', color='yellow', markersize=10, label='Sun (Fixed)')
    
    # Считаем годы для заголовка
    total_sim_years = total_points * dt_input / (365.25 * 24 * 3600)
    
    plt.title(f'Mercury Perihelion Precession (Rosette Pattern)\nDuration: {total_sim_years:.0f} years (dt={dt_input}s)', fontsize=14, color='white')
    plt.xlabel('Rotating X (m)')
    plt.ylabel('Rotating Y (m)')
    plt.axis('equal')
    plt.grid(True, alpha=0.1)
    
    # Убираем рамки
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    output_file = f"assets/rosette_{folder_name}.png"
    plt.savefig(output_file)
    print(f"✅ Saved rosette visualization to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Проверка аргументов (как в твоем коде)
    if len(sys.argv) < 3:
        print("Usage: python plot_rosette.py <DATA_FOLDER_NAME> <DT_SAVE_SECONDS>")
        print("Example: python plot_rosette.py sim_60000y 3600")
    else:
        folder = sys.argv[1]
        dt_val = float(sys.argv[2])
        plot_rosette(folder, dt_val)