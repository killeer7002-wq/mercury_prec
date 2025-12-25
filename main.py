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

def plot(planets: list[Planet]) -> None:
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
  mercury_dist_p = 46.0e9
  mercury_vel_p = 58980.0
  mercury_mass = 3.30e23

  sun = Planet(M_SUN, np.array([0, 0], dtype=np.float64), name="Sun", color="orange")
  mercury = Planet(mercury_mass,
                   np.array([mercury_dist_p, 0], dtype=np.float64),
                   np.array([0, mercury_vel_p], dtype=np.float64),
                   name="Mercury",
                   color="gray")

  planets = [sun, mercury]

  forces = Forces(DIM)

  forces.registrate(NewtonF)
  forces.registrate(EinsteinF)

  dt = 3600
  t = 0
  total_time = 0.25 * 365 * 24 * 3600 

  print(f"Start simulation: {total_time=}...")

  for planet in planets:
    acc_calculate(planet, planets, forces)
  while t < total_time:
    move_calculate(dt, planets, forces)
    t += dt
  
  animate_orbits(planets, "./assets/orbit.gif", 60, 10, 200)
  plot(planets)

if __name__ == "__main__":
  main()
