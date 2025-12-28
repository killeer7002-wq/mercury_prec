import numpy as np
from numba import njit
import time
import sys

from models import Planet
from ephemeris import get_j2000_state
from utils import save_data_binary
from consts import *

# --- НАСТРОЙКИ ---
# Считываются из аргументов командной строки
TARGET_DT = float(sys.argv[1])         # Шаг интегрирования в секундах
TARGET_YEARS = float(sys.argv[2])      # Длительность симуляции в годах
SAVE_STRIDE_SEC = float(sys.argv[3]) * 3600.0 # Интервал сохранения данных в часах
FILENAME = sys.argv[4]                 # Имя папки для сохранения результатов

# Расчетные параметры
STEPS_PER_YEAR = int(365.25 * 24 * 3600 / TARGET_DT)
TOTAL_STEPS = int(TARGET_YEARS * STEPS_PER_YEAR)
SAVE_STRIDE = int(SAVE_STRIDE_SEC / TARGET_DT)
TOTAL_SAVES = TOTAL_STEPS // SAVE_STRIDE

@njit(fastmath=True, cache=True)
def compute_acc_and_pot(pos, vel, masses, G, C):
    """
    Вычисляет ускорения для всех тел в системе с использованием Numba.

    Эта функция, скомпилированная с помощью Numba, эффективно рассчитывает
    гравитационные взаимодействия между всеми парами тел. Она включает
    пост-ньютоновскую поправку для учета эффектов Общей теории относительности.

    Args:
        pos (np.ndarray): Массив позиций всех тел, форма (N, 3).
        vel (np.ndarray): Массив скоростей всех тел, форма (N, 3).
        masses (np.ndarray): Массив масс всех тел, форма (N,).
        G (float): Гравитационная постоянная.
        C (float): Скорость света.

    Returns:
        np.ndarray: Массив ускорений для всех тел, форма (N, 3).
    """
    n = len(masses)
    acc = np.zeros((n, 3), dtype=np.float64)
    c2 = C * C

    # Оптимизированный цикл по парам тел (i, j) где j > i
    for i in range(n):
        for j in range(i + 1, n): 
            # Вектор r_ij (от i к j)
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            
            dist_sq = dx*dx + dy*dy + dz*dz
            dist = np.sqrt(dist_sq) 
            
            # Пост-ньютоновская поправка (ОТО)
            dvx = vel[i, 0] - vel[j, 0]
            dvy = vel[i, 1] - vel[j, 1]
            dvz = vel[i, 2] - vel[j, 2]
            
            # Векторное произведение L = r x v
            lx = dy*dvz - dz*dvy
            ly = dz*dvx - dx*dvz
            lz = dx*dvy - dy*dvx
            l_sq = lx*lx + ly*ly + lz*lz
            
            # Расчет полной силы с релятивистским множителем
            base_force = G / (dist_sq * dist)
            einstein_factor = 1.0 + (3.0 * l_sq) / (c2 * dist_sq)
            
            force_mag = base_force * einstein_factor
            
            fx = force_mag * dx
            fy = force_mag * dy
            fz = force_mag * dz
            
            # Применение силы к телу i (притягивается к j)
            mj = masses[j]
            acc[i, 0] += mj * fx
            acc[i, 1] += mj * fy
            acc[i, 2] += mj * fz
            
            # Применение силы к телу j (3-й закон Ньютона)
            mi = masses[i]
            acc[j, 0] -= mi * fx
            acc[j, 1] -= mi * fy
            acc[j, 2] -= mi * fz
            
    return acc

@njit(fastmath=True, nogil=True)
def run_chunk(pos, vel, acc, masses, dt, steps, G, C):
    """
    Выполняет заданное количество шагов симуляции (один "чанк").

    Использует интегратор Velocity Verlet ("kick-drift-kick").
    Функция скомпилирована с Numba для максимальной производительности.

    Args:
        pos (np.ndarray): Начальные позиции.
        vel (np.ndarray): Начальные скорости.
        acc (np.ndarray): Начальные ускорения.
        masses (np.ndarray): Массы тел.
        dt (float): Шаг по времени.
        steps (int): Количество шагов для выполнения.
        G (float): Гравитационная постоянная.
        C (float): Скорость света.

    Returns:
        tuple: Кортеж с конечными состояниями (pos, vel, acc).
    """
    dt_05 = 0.5 * dt
    
    for _ in range(steps):
        # Kick 1: Обновление скорости на полшага
        vel += acc * dt_05
        
        # Drift: Обновление позиции на полный шаг
        pos += vel * dt
        
        # Перерасчет сил (ускорений) в новой позиции
        acc = compute_acc_and_pot(pos, vel, masses, G, C)
        
        # Kick 2: Обновление скорости на вторую половину шага
        vel += acc * dt_05
        
    return pos, vel, acc

def main():
    """
    Основная функция для запуска высокопроизводительной симуляции на CPU.

    Скрипт выполняет следующие действия:
    1.  Считывает параметры симуляции (шаг, длительность, частота сохранения)
        из аргументов командной строки.
    2.  Инициализирует объекты планет и Солнца, используя эфемериды J2000.
    3.  Выполняет коррекцию скорости Солнца для стабилизации центра масс системы.
    4.  Подготавливает numpy-массивы для позиций, скоростей и масс.
    5.  Выделяет память для хранения истории симуляции.
    6.  "Прогревает" Numba-функции для их компиляции перед основным циклом.
    7.  Запускает основной цикл, который выполняет симуляцию по "чанкам"
        (отрезкам времени), сохраняя результаты и отображая прогресс.
    8.  Обеспечивает возможность прерывания (Ctrl+C) с последующим сохранением
        уже рассчитанных данных.
    9.  Сохраняет итоговые траектории в бинарном формате .npy.
    """
    print(f"\n💻 STARTING CPU OPTIMIZED SIMULATION")
    print(f"=====================================")
    print(f"Time Step:   {TARGET_DT} s")
    print(f"Duration:    {TARGET_YEARS} years")
    print(f"Total Steps: {TOTAL_STEPS:,}")
    print(f"=====================================\n")

    # 1. Инициализация планет
    planets_names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    colors = ["gray", "yellow", "blue", "red", "orange", "gold", "lightblue", "darkblue"]
    masses_list = [3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.898e27, 5.68e26, 8.68e25, 1.02e26]
    
    all_planets = []
    sun = Planet(M_SUN, np.array([0,0,0], dtype=float), name="Sun", color="white")
    all_planets.append(sun)
    for name, c, m in zip(planets_names, colors, masses_list):
        r, v = get_j2000_state(name)
        all_planets.append(Planet(m, r, v, name=name, color=c))

    # Коррекция скорости Солнца для обнуления импульса системы
    p_tot = np.zeros(3)
    for p in all_planets[1:]: p_tot += p.mass * p.u
    sun.u = -p_tot / sun.mass

    # Создание numpy-массивов для Numba
    n_planets = len(all_planets)
    pos = np.array([p.r for p in all_planets], dtype=np.float64)
    vel = np.array([p.u for p in all_planets], dtype=np.float64)
    masses = np.array([p.mass for p in all_planets], dtype=np.float64)
    
    # Выделение памяти под историю траекторий
    print(f"Allocating RAM for history ({TOTAL_SAVES} frames)...")
    hist_pos = np.zeros((TOTAL_SAVES + 5, n_planets, 3), dtype=np.float64)
    hist_vel = np.zeros((TOTAL_SAVES + 5, n_planets, 3), dtype=np.float64)
    
    # "Прогрев" и компиляция Numba-функций
    print("Compiling JIT (please wait)...", end=" ", flush=True)
    acc = compute_acc_and_pot(pos, vel, masses, G, C)
    run_chunk(pos, vel, acc, masses, TARGET_DT, 1, G, C)
    print("Done.")

    # 2. Основной цикл симуляции
    chunk_steps = int(SAVE_STRIDE_SEC / TARGET_DT) # Количество шагов в одном чанке
    total_chunks = TOTAL_SAVES
    
    start_time = time.time()
    
    # Сохранение начального состояния
    hist_pos[0] = pos
    hist_vel[0] = vel
    
    print(f"Running {total_chunks} chunks...")
    print("Ctrl+C to stop and save.")
    
    frames_saved = 0

    try:
        for i in range(total_chunks):
            # Выполнение одного чанка симуляции
            pos, vel, acc = run_chunk(pos, vel, acc, masses, TARGET_DT, chunk_steps, G, C)
            
            # Сохранение результата в массив истории
            hist_pos[i+1] = pos
            hist_vel[i+1] = vel
            frames_saved = i + 1
            
            # Обновление индикатора прогресса
            if i % 100 == 0:
                elapsed = time.time() - start_time
                progress = (i + 1) / total_chunks
                
                if elapsed > 0:
                    total_steps_done = (i + 1) * chunk_steps
                    speed = total_steps_done / elapsed / 1e6 # M шагов/с
                    eta = (elapsed / progress - elapsed) if progress > 0 else 0
                    
                    bar_len = 30
                    filled = int(bar_len * progress)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    sys.stdout.write(f"\r|{bar}| {progress*100:5.1f}% Spd: {speed:5.2f} M/s ETA: {int(eta//60)}m {int(eta%60):02d}s")
                    sys.stdout.flush()
                    
    except KeyboardInterrupt:
        print("\n\nСимуляция остановлена пользователем.")

    total_time = time.time() - start_time
    print(f"\nSimulation finished in {total_time/60:.1f} min.")
    
    # 3. Сохранение результатов
    print("Saving to disk...")
    planets_meta = []
    for i, p in enumerate(all_planets):
        meta = {
            "name": p.name, "color": p.color, "mass": p.mass,
            "last_r": pos[i].tolist(),
            "last_u": vel[i].tolist()
        }
        planets_meta.append(meta)

    # Сохраняем только фактически рассчитанные кадры
    save_data_binary(planets_meta, hist_pos[:frames_saved+1], hist_vel[:frames_saved+1], "assets/" + FILENAME)
    print("Done.")

if __name__ == "__main__":
    main()
