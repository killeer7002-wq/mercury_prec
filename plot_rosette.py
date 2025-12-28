"""
Этот скрипт предназначен для визуализации прецессии перигелия Меркурия
в виде "розетки". Он строит траекторию движения Меркурия во вращающейся
системе координат, синхронизированной со средним орбитальным движением
планеты, что наглядно демонстрирует смещение орбиты с течением времени.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from utils import load_data_binary
from consts import * 

# Увеличиваем `chunksize` для agg бэкенда matplotlib, чтобы избежать
# OverflowError при обработке большого количества точек на графике.
plt.rcParams['agg.path.chunksize'] = 10000 

# Большая полуось Меркурия (для расчета средней угловой скорости)
# Определяется здесь, так как может отсутствовать в основном файле констант.
A_MERCURY = 0.387098 * AU 

def plot_rosette(folder_name, dt_input):
    """
    Загружает данные симуляции, строит и сохраняет график розетки для Меркурия.

    Процесс работы функции:
    1. Загрузка бинарных данных траектории из указанной папки.
    2. Прореживание (downsampling) данных, если их объем превышает 1 млн точек,
       для предотвращения проблем с производительностью и памятью в matplotlib.
    3. Пересчет координат траектории Меркурия во вращающуюся систему отсчета.
       Система вращается со средней угловой скоростью Меркурия, что позволяет
       визуально выделить прецессию перигелия.
    4. Создание и стилизация графика с помощью matplotlib.
    5. Сохранение итогового изображения в формате PNG.

    Args:
        folder_name (str): Имя папки в `assets/`, содержащей данные симуляции.
        dt_input (float): Шаг по времени (в секундах), который использовался
                          при сохранении данных симуляции.
    """
    # Загрузка данных
    path = f"assets/{folder_name}"
    print(f"Loading data from {path}...")
    try:
        planets = load_data_binary(path)
    except FileNotFoundError:
        print(f"Error: Folder {path} not found.")
        return

    mercury = next(p for p in planets if p.name == "Mercury")
    
    # Подготовка данных с возможным прореживанием
    x_full = np.array(mercury.path_x)
    y_full = np.array(mercury.path_y)
    
    total_points = len(x_full)
    print(f"Total points available: {total_points:,}")

    # Прореживание (Downsampling) для оптимизации отрисовки.
    # Ограничиваем количество точек на графике для стабильной работы matplotlib.
    target_points = 1_000_000 
    step = max(1, total_points // target_points)
    
    if step > 1:
        print(f"Downsampling active: using every {step}-th point (plotting ~{total_points//step:,} pts)...")
    
    x = x_full[::step]
    y = y_full[::step]
    
    # Расчет времени и координат во вращающейся системе
    # Средняя угловая скорость (Mean Motion) Меркурия
    n = np.sqrt(GM / A_MERCURY**3)
    print(f"Mercury Mean Motion: {n:.2e} rad/s")
    
    # Расчет массива времени для каждой точки с учетом прореживания:
    # t = индекс_точки * шаг_прореживания * шаг_времени_симуляции
    t = np.arange(len(x)) * step * dt_input
    
    print("Calculating rotating frame coordinates...")
    
    # Угол поворота для каждой точки времени
    theta = n * t
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Поворот координат в новую систему (x', y')
    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t
    
    # Визуализация
    print("Plotting...")
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 12), dpi=150)
    
    plt.plot(x_rot, y_rot, color='cyan', lw=0.3, alpha=0.4)
    plt.plot(0, 0, '*', color='yellow', markersize=10, label='Sun (Fixed)')
    
    total_sim_years = total_points * dt_input / (365.25 * 24 * 3600)
    
    plt.title(f'Mercury Perihelion Precession (Rosette Pattern)\nDuration: {total_sim_years:.0f} years (dt={dt_input}s)', fontsize=14, color='white')
    plt.xlabel('Rotating X (m)')
    plt.ylabel('Rotating Y (m)')
    plt.axis('equal')
    plt.grid(True, alpha=0.1)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    output_file = f"assets/rosette_{folder_name}.png"
    plt.savefig(output_file)
    print(f"✅ Saved rosette visualization to {output_file}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_rosette.py <DATA_FOLDER_NAME> <DT_SAVE_SECONDS>")
        print("Example: python plot_rosette.py sim_60000y 3600")
    else:
        folder = sys.argv[1]
        dt_val = float(sys.argv[2])
        plot_rosette(folder, dt_val)
