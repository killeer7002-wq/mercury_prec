import pickle
import numpy as np
import numpy.typing as npt

from models import Planet, Forces
from ephemeris import get_j2000_state

G = 6.674e-11
C = 299.8e6
M_SUN = 1.989e30
AU = 1.496e11
DIM = np.array([0, 0, 0])

def mod(arr: npt.NDArray[np.float64]) -> np.float64:
  return np.float64(np.linalg.norm(arr))

########## Forces #########

def NewtonF(pln1: Planet, pln2: Planet) -> npt.NDArray[np.float64]:
  r = pln2.r - pln1.r
  return - G * pln1.mass / (mod(r) ** 3) * r

def EinsteinF(pln1: Planet, pln2: Planet) -> npt.NDArray[np.float64]:
    # Считаем только для пары Солнце-Меркурий
    if pln1.name != "Sun" or pln2.name != "Mercury":
        return np.zeros_like(DIM, dtype=np.float64)
    
    r_vec = pln2.r - pln1.r
    r_len = mod(r_vec)

    v_vec = pln2.u - pln1.u

    L_vec = np.cross(r_vec, v_vec)
    L_sq = mod(L_vec)**2

    prefactor = (3 * G * pln1.mass * L_sq) / (C**2 * r_len**5)
    
    return - prefactor * r_vec

###########################

def acc_calculate(planet: Planet, planets: list[Planet], forces: Forces) -> None:
  planet.a = np.zeros_like(DIM, dtype=np.float64)
  for pln in planets:
    if (pln == planet):
      continue
    planet.a += forces.calculate(pln, planet)

def move_calculate(dt, planets: list[Planet], forces: Forces) -> None:
  for planet in planets:
    planet.u += planet.a * dt / 2
  
  # 2. Drift (position)
  for planet in planets:
    planet.r += planet.u * dt
    # НЕ записываем путь здесь! Ждем обновления скорости.

  # 3. Recalculate forces
  for planet in planets:
    acc_calculate(planet, planets, forces)

  # 4. Second half-kick (velocity)
  for planet in planets:
    planet.u += planet.a * dt / 2
    
  # 5. ТЕПЕРЬ, когда r(t+1) и u(t+1) синхронны, сохраняем состояние
  for planet in planets:
    planet.record_state()

def save_data(planets: list[Planet], filename: str = "assets/simulation_data.pkl"):
    """Сохраняет список объектов планет в бинарный файл"""
    with open(filename, 'wb') as f:
        pickle.dump(planets, f)
    print(f"Data successfully saved to {filename}")

def main() -> None:
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

  # --- ФИКС ДРЕЙФА СОЛНЦА ---
  # Считаем суммарный импульс всех планет (P = m * v)
  total_momentum = np.zeros_like(DIM, dtype=np.float64)
  for p in all_planets[1:]:
      total_momentum += p.mass * p.u
  
  # Придаем Солнцу скорость в обратную сторону, чтобы сумма была 0
  sun.u = - total_momentum / sun.mass
  
  print(f"Sun correction velocity: {sun.u} m/s")

  # --- Настройка симуляции ---
  forces = Forces(DIM)
  forces.registrate(NewtonF)
  forces.registrate(EinsteinF)

  # dt = 1 час. Для Меркурия это ок, для Нептуна это очень детально.
  dt = 60 
  t = 0
  total_time = 1.0 * 365 * 24 * 3600 * 2

  print(f"Start simulation: {total_time/3600/24/365:.2f} Earth years...")

  # Предварительный расчет сил
  for planet in all_planets:
      acc_calculate(planet, all_planets, forces)
      
  # Основной цикл
  steps = int(total_time / dt)
  check_interval = steps // 10 # Вывод прогресса 10 раз за симуляцию
  
  step_count = 0
  while t < total_time:
      move_calculate(dt, all_planets, forces)
      t += dt
      step_count += 1
      if step_count % check_interval == 0:
          print(f"Progress: {int(t/total_time*100)}%")
  
  print("Simulation finished.")

  save_data(all_planets)

if __name__ == "__main__":
  main()
