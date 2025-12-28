import numpy as np
import numpy.typing as npt

from models import Planet, Forces
from ephemeris import get_j2000_state
from utils import save_data

from consts import *

def mod(arr: npt.NDArray[np.float64]) -> np.float64:
  """
  Вычисляет евклидову норму (длину) вектора.

  Args:
      arr (npt.NDArray[np.float64]): Входной numpy-вектор.

  Returns:
      np.float64: Скалярное значение длины вектора.
  """
  return np.float64(np.linalg.norm(arr))


def NewtonF(pln1: Planet, pln2: Planet) -> npt.NDArray[np.float64]:
  """
  Рассчитывает силу гравитационного притяжения по закону Ньютона.

  Вычисляет вектор силы, действующей на планету `pln1` со стороны `pln2`.
  Знак "минус" инвертирует направление силы от `pln2` к `pln1`.

  Args:
      pln1 (Planet): Планета, на которую действует сила.
      pln2 (Planet): Планета, которая создает гравитационное поле.

  Returns:
      npt.NDArray[np.float64]: Вектор силы в 3D.
  """
  r = pln2.r - pln1.r
  return - G * pln1.mass / (mod(r) ** 3) * r

def EinsteinF(pln1: Planet, pln2: Planet) -> npt.NDArray[np.float64]:
    """
    Рассчитывает релятивистскую поправку к силе (Общая теория относительности).

    Эта функция вычисляет дополнительную силу, ответственную за прецессию
    орбиты. В данной реализации расчет производится только для пары
    Солнце-Меркурий, так как для других планет этот эффект пренебрежимо мал.

    Args:
        pln1 (Planet): Планета, на которую действует сила (Солнце).
        pln2 (Planet): Планета, для которой рассчитывается поправка (Меркурий).

    Returns:
        npt.NDArray[np.float64]: Вектор релятивистской поправки к силе.
                                 Возвращает нулевой вектор для всех других пар.
    """
    # Считаем только для пары Солнце-Меркурий
    if pln1.name != "Sun" or pln2.name != "Mercury":
        return np.zeros_like(DIM, dtype=np.float64)
    
    r_vec = pln2.r - pln1.r
    r_len = mod(r_vec)

    v_vec = pln2.u - pln1.u

    L_vec = np.cross(r_vec, v_vec).astype(np.float64)
    L_sq = mod(L_vec)**2

    prefactor = (3 * G * pln1.mass * L_sq) / (C**2 * r_len**5)
    
    return - prefactor * r_vec


def acc_calculate(planet: Planet, planets: list[Planet], forces: Forces) -> None:
  """
  Рассчитывает и обновляет полное ускорение для одной планеты.

  Суммирует все гравитационные силы, действующие на `planet` со стороны
  всех остальных планет `planets` в системе, и обновляет поле `planet.a`.

  Args:
      planet (Planet): Планета, для которой вычисляется ускорение.
      planets (list[Planet]): Полный список планет в симуляции.
      forces (Forces): Объект, управляющий зарегистрированными типами сил.
  """
  planet.a = np.zeros_like(DIM, dtype=np.float64)
  for pln in planets:
    if (pln == planet):
      continue
    planet.a += forces.calculate(pln, planet)

def move_calculate(dt, planets: list[Planet], forces: Forces) -> None:
  """
  Выполняет один шаг симуляции по методу "kick-drift-kick" (разновидность Липфрога).

  Этапы алгоритма:
  1. "Kick": Обновление скоростей на половине шага (u_i+1/2).
  2. "Drift": Обновление позиций с использованием новых скоростей (r_i+1).
  3. Перерасчет сил (ускорений) в новой позиции.
  4. Второй "Kick": Обновление скоростей на второй половине шага (u_i+1).
  5. Запись нового состояния (r_i+1, u_i+1) в историю.

  Args:
      dt: Временной шаг симуляции.
      planets (list[Planet]): Список всех планет в симуляции.
      forces (Forces): Объект, управляющий расчетом сил.
  """
  # 1. Первый "Kick" (обновление скорости на dt/2)
  for planet in planets:
    planet.u += planet.a * dt / 2
  
  # 2. "Drift" (обновление позиции на полный шаг dt)
  for planet in planets:
    planet.r += planet.u * dt

  # 3. Перерасчет сил (ускорений) в новых позициях
  for planet in planets:
    acc_calculate(planet, planets, forces)

  # 4. Второй "Kick" (обновление скорости на оставшиеся dt/2)
  for planet in planets:
    planet.u += planet.a * dt / 2
    
  # 5. Синхронная запись состояния r(t+1) и u(t+1)
  for planet in planets:
    planet.record_state()

def main() -> None:
  """
  Основная функция для запуска симуляции гравитационной системы.

  Процесс выполнения:
  1. Инициализация планет Солнечной системы с использованием реальных
     масс и эфемерид (позиций и скоростей) на эпоху J2000.
  2. Коррекция скорости Солнца для обеспечения нулевого суммарного
     импульса системы, чтобы избежать ее дрейфа.
  3. Настройка симуляции: регистрация действующих сил (Ньютон + ОТО).
  4. Запуск основного цикла симуляции методом Липфрога.
  5. Сохранение результатов симуляции в CSV файлы.
  """
  planets_names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
  colors = ["gray", "yellow", "blue", "red", "orange", "gold", "lightblue", "darkblue"]
  masses = [3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.898e27, 5.68e26, 8.68e25, 1.02e26] # кг
  
  sun = Planet(M_SUN, np.array([0, 0, 0], dtype=np.float64), name="Sun", color="white")
  all_planets = [sun]
  
  for name, color, mass in zip(planets_names, colors, masses):
    # Получаем точные векторы на 2000 год
    r_vec, v_vec = get_j2000_state(name)
    
    p = Planet(
      mass=mass,
      r=r_vec,
      u=v_vec,
      name=name,
      color=color
    )
    all_planets.append(p)

  # Коррекция дрейфа центра масс системы
  # Считаем суммарный импульс всех планет (P = m * v)
  total_momentum = np.zeros_like(DIM, dtype=np.float64)
  for p in all_planets[1:]:
      total_momentum += p.mass * p.u
  
  # Придаем Солнцу скорость в обратную сторону, чтобы сумма импульсов была равна нулю
  sun.u = - total_momentum / sun.mass
  
  print(f"Sun correction velocity: {sun.u} m/s")

  # Настройка симуляции
  forces = Forces(DIM)
  forces.registrate(NewtonF)
  forces.registrate(EinsteinF)

  t = 0
  total_time = 1.0 * 365 * 24 * 3600 * YEARS_SIM

  print(f"Start simulation: {total_time/3600/24/365:.2f} Earth years...")

  # Предварительный расчет сил в начальный момент времени
  for planet in all_planets:
      acc_calculate(planet, all_planets, forces)
      
  # Основной цикл
  steps = int(total_time / DT)
  check_interval = steps // 10 # Вывод прогресса 10 раз за симуляцию
  
  step_count = 0
  while t < total_time:
      move_calculate(DT, all_planets, forces)
      t += DT
      step_count += 1
      if step_count % check_interval == 0:
          print(f"Progress: {int(t/total_time*100)}%")
  
  print("Simulation finished.")

  save_data(all_planets)

if __name__ == "__main__":
  main()