import pickle
import numpy as np
import matplotlib.pyplot as plt

from models import Planet
from visualizer import animate_orbits

__all__ = ["plot"]

def load_data(filename: str = "assets/simulation_data.pkl") -> list[Planet]:
    with open(filename, 'rb') as f:
        planets = pickle.load(f)
    print(f"Loaded {len(planets)} planets from {filename}")
    return planets

def plot_static(planets: list[Planet], filename: str = "./assets/sim.png") -> None:
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
  plt.savefig(filename)

def plot(all_planets: list[Planet],
         filename_png: str = "./assets/sim.png",
         filename_gif: str = "./assets/orbit.gif") -> None:
  # --- Визуализация ---
  plot_static(all_planets, filename_png)
  
  # ВАЖНО: Для анимации берем только внутренние планеты + Юпитер.
  # Если рисовать Нептун, Меркурий станет невидимым из-за зума.
  # Индексы: 0-Sun, 1-Mercury, 2-Venus, 3-Earth, 4-Mars, 5-Jupiter
  inner_system = all_planets[:6] 
  
  print(f"Rendering animation for Inner Solar System: {", ".join([pln.name for pln in inner_system])}")
  animate_orbits(
      inner_system, 
      filename=filename_gif, 
      fps=30, 
      stride=24 * 40,   # Берем кадр раз в сутки (так как dt=1 час)
      trace_length=400
  )

if __name__ == "__main__":
  plot(load_data())
