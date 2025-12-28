import time

import numpy as np
from numba import njit

from consts import C, DT, G, M_SUN, YEARS_SIM
from ephemeris import get_j2000_state
from models import Planet
from utils import save_data_binary


@njit(fastmath=True)
def compute_accelerations(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray,
                          n_planets: int, G_const: float, C_const: float) -> np.ndarray:
    """
    Рассчитывает ускорения для всех тел в системе.

    Для каждой пары тел вычисляется сила гравитационного взаимодействия.
    Ускорение включает в себя как классический Ньютоновский компонент,
    так и поправку из Общей теории относительности (ОТО), которая важна
    для точного моделирования прецессии орбиты Меркурия.

    Args:
        pos (np.ndarray): Массив положений всех тел (форма: [N, 3]).
        vel (np.ndarray): Массив скоростей всех тел (форма: [N, 3]).
        masses (np.ndarray): Массив масс всех тел (форма: [N]).
        n_planets (int): Количество тел в симуляции.
        G_const (float): Гравитационная постоянная.
        C_const (float): Скорость света.

    Returns:
        np.ndarray: Массив ускорений для каждого тела (форма: [N, 3]).
    """
    acc = np.zeros((n_planets, 3), dtype=np.float64)

    for i in range(n_planets):
        for j in range(n_planets):
            if i == j:
                continue

            # Вектор r_ij от тела i к телу j
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]

            dist_sq = dx*dx + dy*dy + dz*dz
            dist = np.sqrt(dist_sq)
            dist_inv = 1.0 / dist

            # --- 1. Ньютоновская гравитация ---
            # a = G * M_j / r^2 * (r_vec / r)
            a_mag_newton = G_const * masses[j] / dist_sq
            acc[i, 0] += a_mag_newton * dx * dist_inv
            acc[i, 1] += a_mag_newton * dy * dist_inv
            acc[i, 2] += a_mag_newton * dz * dist_inv

            # --- 2. Поправка ОТО (для всех пар, но значима для Меркурий-Солнце) ---
            # Формула дополнительной силы: F_gr = (3 * G * M * L^2) / (c^2 * r^4) * u_r
            # где L - угловой момент, u_r - единичный вектор.
            
            # Относительные векторы для формулы (r = pos_i - pos_j)
            r_vec = pos[i] - pos[j]
            v_vec = vel[i] - vel[j]

            # Угловой момент на единицу массы L = r x v
            L_vec = np.cross(r_vec, v_vec)
            L_sq = np.sum(L_vec**2)

            # Коэффициент в формуле релятивистской поправки
            # Знак минус в r_vec компенсируется направлением силы к источнику
            prefactor = (3.0 * G_const * masses[j] * L_sq) / (C_const**2 * dist**5)

            acc[i, 0] -= prefactor * r_vec[0]
            acc[i, 1] -= prefactor * r_vec[1]
            acc[i, 2] -= prefactor * r_vec[2]

    return acc


@njit
def run_simulation_loop(init_pos: np.ndarray, init_vel: np.ndarray, masses: np.ndarray,
                        dt: float, total_steps: int, G_const: float, C_const: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Выполняет основной цикл симуляции методом Верле (Velocity Verlet).

    Этот метод является численно стабильным и сохраняет энергию в долгосрочной
    перспективе. Цикл выполняется в коде, скомпилированном Numba, для
    максимальной производительности.

    Внимание: функция аллоцирует в памяти массивы для хранения полной
    истории положений и скоростей, что может требовать значительного
    объема ОЗУ при большом количестве шагов.

    Args:
        init_pos (np.ndarray): Начальные положения тел.
        init_vel (np.ndarray): Начальные скорости тел.
        masses (np.ndarray): Массы тел.
        dt (float): Шаг интегрирования по времени (в секундах).
        total_steps (int): Общее количество шагов симуляции.
        G_const (float): Гравитационная постоянная.
        C_const (float): Скорость света.

    Returns:
        tuple[np.ndarray, np.ndarray]: Два массива:
        - history_pos: полная история положений (форма: [steps, N, 3]).
        - history_vel: полная история скоростей (форма: [steps, N, 3]).
    """
    n_planets = len(masses)
    history_pos = np.zeros((total_steps + 1, n_planets, 3), dtype=np.float64)
    history_vel = np.zeros((total_steps + 1, n_planets, 3), dtype=np.float64)

    pos = init_pos.copy()
    vel = init_vel.copy()

    history_pos[0] = pos
    history_vel[0] = vel

    # Начальное ускорение
    acc = compute_accelerations(pos, vel, masses, n_planets, G_const, C_const)

    for step in range(1, total_steps + 1):
        # 1. Обновление скорости на половину шага
        vel += acc * (dt * 0.5)
        # 2. Обновление позиции на полный шаг
        pos += vel * dt
        # 3. Вычисление нового ускорения в новой позиции
        acc = compute_accelerations(pos, vel, masses, n_planets, G_const, C_const)
        # 4. Обновление скорости на вторую половину шага
        vel += acc * (dt * 0.5)

        history_pos[step] = pos
        history_vel[step] = vel

    return history_pos, history_vel


def main():
    """
    Главная функция для запуска N-body симуляции Солнечной системы.

    Процесс выполнения:
    1. Инициализация планет: создание объектов, загрузка их начальных
       положений и скоростей на эпоху J2000.
    2. Коррекция барицентра: установка скорости Солнца таким образом,
       чтобы суммарный импульс системы был равен нулю.
    3. Подготовка данных: преобразование списков объектов в NumPy массивы.
    4. Запуск симуляции: вызов Numba-ускоренного цикла `run_simulation_loop`.
    5. Сохранение результатов: запись метаданных и полной истории
       траекторий в бинарные файлы.
    """
    print(f"Запуск быстрой симуляции (ускорение Numba)...")
    print(f"Шаг по времени (DT): {DT} с")

    # Инициализация планет на основе реальных эфемерид
    planets_names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    colors = ["gray", "yellow", "blue", "red", "orange", "gold", "lightblue", "darkblue"]
    masses_list = [3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.898e27, 5.68e26, 8.68e25, 1.02e26]

    sun = Planet(M_SUN, np.array([0, 0, 0], dtype=np.float64), name="Sun", color="white")
    all_planets = [sun]
    for name, color, mass in zip(planets_names, colors, masses_list):
        r_vec, v_vec = get_j2000_state(name)
        all_planets.append(Planet(mass=mass, r=r_vec, u=v_vec, name=name, color=color))

    # Коррекция скорости Солнца для обнуления полного импульса системы
    total_momentum = np.zeros(3, dtype=np.float64)
    for p in all_planets[1:]:
        total_momentum += p.mass * p.u
    sun.u = -total_momentum / sun.mass
    print(f"Корректирующая скорость Солнца: {sun.u} м/с")

    # Подготовка массивов для Numba
    n_planets = len(all_planets)
    pos_arr = np.array([p.r for p in all_planets], dtype=np.float64)
    vel_arr = np.array([p.u for p in all_planets], dtype=np.float64)
    mass_arr = np.array([p.mass for p in all_planets], dtype=np.float64)

    # Настройки времени симуляции
    years = YEARS_SIM
    total_time = years * 365.25 * 24 * 3600  # Учитываем високосные годы
    steps = int(total_time / DT)
    print(f"Симуляция на {years} лет, шагов: {steps}...")

    mem_size_gb = (steps * n_planets * 6 * 8) / (1024**3)
    print(f"Ожидаемое использование ОЗУ для истории: {mem_size_gb:.2f} GB")
    if mem_size_gb > 8.0:
        print("ВНИМАНИЕ: Высокое потребление памяти! Рассмотрите увеличение DT или уменьшение времени симуляции.")

    # Запуск симуляции с замером времени
    start_time = time.time()
    # Первый вызов JIT-функции может занять время на компиляцию
    hist_pos, hist_vel = run_simulation_loop(pos_arr, vel_arr, mass_arr, DT, steps, G, C)
    elapsed = time.time() - start_time
    print(f"Симуляция завершена за {elapsed:.2f} секунд.")
    print(f"Скорость: {steps / elapsed:.0f} шагов/сек")

    # Сохранение результатов
    print("Сохранение данных из массивов...")
    planets_meta = []
    final_pos = hist_pos[-1]
    final_vel = hist_vel[-1]

    for i, p in enumerate(all_planets):
        meta = {
            "name": p.name, "color": p.color, "mass": p.mass,
            "last_r": final_pos[i].tolist(), "last_u": final_vel[i].tolist()
        }
        planets_meta.append(meta)

    # Локальный импорт неиспользуемой функции, оставлен без изменений
    # согласно заданию. Фактически для сохранения используется `save_data_binary`.
    from utils import save_data_from_arrays
    save_data_binary(planets_meta, hist_pos, hist_vel)
    print("Сохранение завершено.")


if __name__ == "__main__":
    main()
