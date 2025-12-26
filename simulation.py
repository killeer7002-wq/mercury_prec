import numpy as np
from numba import njit
import time

# Импортируем структуру данных и константы
from models import Planet
from utils import save_data_binary
from ephemeris import get_j2000_state
from consts import *

@njit(fastmath=True)
def compute_accelerations(pos, vel, masses, n_planets, G, C, sun_idx, mercury_idx):
    """
    Рассчитывает ускорения для всех тел (Ньютон + Эйнштейн).
    """
    acc = np.zeros((n_planets, 3), dtype=np.float64)

    for i in range(n_planets):
        for j in range(n_planets):
            if i == j:
                continue

            # Вектор r_vec от i к j (Target -> Source)
            # В models.py NewtonF: r = pln2.r - pln1.r (Source -> Target). 
            # Force ~ -r. Значит притяжение.
            # Здесь считаем ускорение i-го тела от j-го.
            # Вектор r_ij = pos[j] - pos[i] направлен К источнику j.
            
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            
            dist_sq = dx*dx + dy*dy + dz*dz
            dist = np.sqrt(dist_sq)

            # --- 1. Ньютоновская гравитация ---
            # a = G * M_j / r^2 * (r_vec / r)
            a_mag = G * masses[j] / (dist_sq * dist)
            
            acc[i, 0] += a_mag * dx
            acc[i, 1] += a_mag * dy
            acc[i, 2] += a_mag * dz

            # --- 2. Эйнштейновская поправка
            
            # r_vec (Source->Target) для формулы из models.py
            # r_vec = pos[i] - pos[j]
            rx = -dx
            ry = -dy
            rz = -dz
            r_len = dist
            
            # v_vec (Rel velocity) = vel[i] - vel[j]
            vx = vel[i, 0] - vel[j, 0]
            vy = vel[i, 1] - vel[j, 1]
            vz = vel[i, 2] - vel[j, 2]
            
            # Cross product L = r x v
            Lx = ry*vz - rz*vy
            Ly = rz*vx - rx*vz
            Lz = rx*vy - ry*vx
            
            L_sq = Lx*Lx + Ly*Ly + Lz*Lz
            
            # Formula: - (3 * G * M_sun * L^2) / (c^2 * r^5) * r_vec
            c2 = C * C
            r5 = r_len * r_len * r_len * r_len * r_len
            
            prefactor = (3.0 * G * masses[j] * L_sq) / (c2 * r5)
            
            # Добавляем к ускорению. 
            # Force vector is -prefactor * r_vec.
            # r_vec направлен от Солнца к Меркурию. 
            # Сила направлена к Солнцу (притяжение).
            # Мы работаем с ускорением.
            
            acc[i, 0] += -prefactor * rx
            acc[i, 1] += -prefactor * ry
            acc[i, 2] += -prefactor * rz

    return acc

@njit
def run_simulation_loop(init_pos, init_vel, masses, dt, total_steps, G, C, sun_idx, mercury_idx):
    """
    Основной цикл Velocity Verlet.
    """
    n_planets = len(masses)
    
    # Аллокация истории.
    # ВНИМАНИЕ: Если total_steps огромный (миллионы), это съест всю память.
    # Но для анализа verify_gr.py нужны все шаги, если мы хотим точно интегрировать.
    history_pos = np.zeros((total_steps + 1, n_planets, 3), dtype=np.float64)
    history_vel = np.zeros((total_steps + 1, n_planets, 3), dtype=np.float64)

    # Текущее состояние
    pos = init_pos.copy()
    vel = init_vel.copy()
    
    # Запись начального состояния
    history_pos[0] = pos
    history_vel[0] = vel

    # Предварительный расчет ускорения
    acc = compute_accelerations(pos, vel, masses, n_planets, G, C, sun_idx, mercury_idx)

    for step in range(1, total_steps + 1):
        # 1. Первый полушаг скорости
        vel += acc * (dt * 0.5)
        
        # 2. Обновление позиции
        pos += vel * dt
        
        # 3. Пересчет ускорений
        acc = compute_accelerations(pos, vel, masses, n_planets, G, C, sun_idx, mercury_idx)
        
        # 4. Второй полушаг скорости
        vel += acc * (dt * 0.5)
        
        # 5. Сохранение
        history_pos[step] = pos
        history_vel[step] = vel
        
    return history_pos, history_vel

def main():
    print(f"Initializing Fast Simulation (Numba accelerated)...")
    print(f"Time Step (DT): {DT} s")
    
    # --- 1. Инициализация планет (как в simulation.py) ---
    planets_names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    colors = ["gray", "yellow", "blue", "red", "orange", "gold", "lightblue", "darkblue"]
    masses_list = [3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.898e27, 5.68e26, 8.68e25, 1.02e26]
    
    # Создаем объекты (нужны для метаданных и сохранения)
    sun = Planet(M_SUN, np.array([0, 0, 0], dtype=np.float64), name="Sun", color="white")
    all_planets = [sun]
    
    for name, color, mass in zip(planets_names, colors, masses_list):
        r_vec, v_vec = get_j2000_state(name)
        p = Planet(mass=mass, r=r_vec, u=v_vec, name=name, color=color)
        all_planets.append(p)

    # --- 2. Коррекция импульса Солнца ---
    total_momentum = np.zeros(3, dtype=np.float64)
    for p in all_planets[1:]:
        total_momentum += p.mass * p.u
    sun.u = - total_momentum / sun.mass
    print(f"Sun correction velocity: {sun.u} m/s")

    # --- 3. Подготовка данных для Numba ---
    n_planets = len(all_planets)
    pos_arr = np.array([p.r for p in all_planets], dtype=np.float64)
    vel_arr = np.array([p.u for p in all_planets], dtype=np.float64)
    mass_arr = np.array([p.mass for p in all_planets], dtype=np.float64)
    
    # Индексы для спец. сил
    sun_idx = -1
    mercury_idx = -1
    for i, p in enumerate(all_planets):
        if p.name == "Sun": sun_idx = i
        if p.name == "Mercury": mercury_idx = i

    # --- 4. Настройки времени ---
    years = YEARS_SIM
    total_time = years * 365 * 24 * 3600
    steps = int(total_time / DT)
    
    print(f"Simulating {years} years in {steps} steps...")
    
    # Проверка памяти
    mem_size_gb = (steps * n_planets * 3 * 8 * 2) / (1024**3)
    print(f"Estimated RAM usage for history arrays: {mem_size_gb:.2f} GB")
    if mem_size_gb > 8.0:
        print("WARNING: High memory usage! Consider increasing DT or reducing time.")

    # --- 5. Запуск симуляции ---
    start_time = time.time()
    
    # Первый запуск скомпилирует функцию (займет 1-2 сек), последующие пойдут быстро
    hist_pos, hist_vel = run_simulation_loop(
        pos_arr, vel_arr, mass_arr, DT, steps, G, C, sun_idx, mercury_idx
    )
    
    elapsed = time.time() - start_time
    print(f"Simulation finished in {elapsed:.2f} seconds.")
    print(f"Speed: {steps / elapsed:.0f} steps/sec")

    # --- 6. Сохранение без конвертации в объекты ---
    print("Saving data directly from arrays...")
    
    # Подготовим метаданные для манифеста
    planets_meta = []
    
    # Нам нужно сохранить последнее состояние в метаданные
    # hist_pos имеет форму (Steps, Planets, 3)
    final_pos = hist_pos[-1]
    final_vel = hist_vel[-1]

    for i, p in enumerate(all_planets):
        meta = {
            "name": p.name,
            "color": p.color,
            "mass": p.mass,
            "last_r": final_pos[i].tolist(),
            "last_u": final_vel[i].tolist()
        }
        planets_meta.append(meta)

    # Импортируем новую функцию
    from utils import save_data_from_arrays
    
    # Передаем сырые массивы.
    # save_data_from_arrays сама транспонирует их если нужно.
    save_data_binary(planets_meta, hist_pos, hist_vel)

if __name__ == "__main__":
    main()