import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from models import Planet, Forces
from visualizer import animate_orbits

G = 6.674e-11
C = 299.8e6
M_SUN = 1.989e30
AU = 1.496e11
DIM = np.array([0, 0])

def mod(arr: npt.NDArray[np.float64]) -> np.float64:
  return np.float64(np.linalg.norm(arr))

########## Forces #########

def NewtonF(pln1: Planet, pln2: Planet) -> npt.NDArray[np.float64]:
  r = pln2.r - pln1.r
  return - G * pln1.mass / (mod(r) ** 3) * r

def EinsteinF(pln1: Planet, pln2: Planet) -> npt.NDArray[np.float64]:
  if pln1.name != "Sun" or pln2.name != "Mercury":
        return np.zeros_like(DIM, dtype=np.float64)
  
  r = pln2.r - pln1.r
  return (2 * G * pln1.mass / (C * mod(r) ** 2)) ** 2 * r

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
  
  for planet in planets:
    planet.r += planet.u * dt
    planet.record_path()

  for planet in planets:
    acc_calculate(planet, planets, forces)

  for planet in planets:
    planet.u += planet.a * dt / 2 

def plot_static(planets: list[Planet]) -> None:
  plt.figure(figsize=(8, 8))
  for planet in planets:
      plt.plot(planet.path_x, planet.path_y, color=planet.color, label=planet.name)
      plt.plot(planet.r[0], planet.r[1], 'o', color=planet.color)

  plt.title("Орбита Меркурия (Ньютоновская)")
  plt.xlabel("X (метры)")
  plt.ylabel("Y (метры)")
  plt.axis('equal')
  plt.legend()
  plt.grid(True)
  plt.savefig('./assets/sim.png')

def main() -> None:
    # --- Инициализация данных планет ---
    # Данные приближенные (средние), стартуем всех с оси X (Парад планет)
    # Формат: "Name": (Mass [kg], Distance [m], Velocity [m/s], Color)
    
    planets_data = {
        "Mercury": (3.30e23,  57.9e9,  47400, "gray"),
        "Venus":   (4.87e24, 108.2e9,  35000, "yellow"),
        "Earth":   (5.97e24, 149.6e9,  29800, "blue"),
        "Mars":    (6.42e23, 227.9e9,  24100, "red"),
        "Jupiter": (1.898e27, 778.6e9, 13100, "orange"),
        "Saturn":  (5.68e26, 1433.5e9, 9700,  "gold"),
        "Uranus":  (8.68e25, 2872.5e9, 6800,  "lightblue"),
        "Neptune": (1.02e26, 4495.1e9, 5400,  "darkblue")
    }

    # Создаем Солнце
    sun = Planet(M_SUN, np.array([0, 0], dtype=np.float64), name="Sun", color="white")
    
    all_planets = [sun]
    
    # Создаем планеты
    for name, (mass, dist, vel, color) in planets_data.items():
        p = Planet(
            mass=mass,
            r=np.array([dist, 0], dtype=np.float64),      # Позиция по X
            u=np.array([0, vel], dtype=np.float64),       # Скорость по Y (перпендикулярно)
            name=name,
            color=color
        )
        all_planets.append(p)

    # --- Настройка симуляции ---
    forces = Forces(DIM)
    forces.registrate(NewtonF)
    forces.registrate(EinsteinF)

    # dt = 1 час. Для Меркурия это ок, для Нептуна это очень детально.
    dt = 3600 
    t = 0
    # Симулируем 1 Земной год (Меркурий успеет сделать ~4 оборота)
    # Если поставить больше, расчет будет идти дольше
    total_time = 1.0 * 365 * 24 * 3600 

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
    
    # --- Визуализация ---
    plot_static(all_planets)
    
    # ВАЖНО: Для анимации берем только внутренние планеты + Юпитер.
    # Если рисовать Нептун, Меркурий станет невидимым из-за зума.
    # Индексы: 0-Sun, 1-Mercury, 2-Venus, 3-Earth, 4-Mars, 5-Jupiter
    inner_system = all_planets[:2] 
    
    print("Rendering animation for Inner Solar System + Jupiter...")
    animate_orbits(
        inner_system, 
        filename="./assets/orbit.gif", 
        fps=30, 
        stride=24,   # Берем кадр раз в сутки (так как dt=1 час)
        trace_length=-1
    )

if __name__ == "__main__":
  main()
