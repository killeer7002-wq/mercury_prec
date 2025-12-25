from collections.abc import Callable
from typing import Union
import numpy as np
import numpy.typing as npt

__all__ = ["Planet", "Forces"]

class _Missing:
    pass

MISSING = _Missing()

class Planet:
  def __init__(self,
               mass: float,
               r: npt.NDArray[np.float64],
               u: Union[npt.NDArray[np.float64], _Missing] = MISSING,
               a: Union[npt.NDArray[np.float64], _Missing] = MISSING,
               name: str = "unknown",
               color: str = "red",):
    self.mass = mass
    self.r = r
    self.u = u
    if u is MISSING:
      self.u = np.zeros_like(r, dtype=np.float64)
    self.a = a
    if a is MISSING:
      self.a = np.zeros_like(r, dtype=np.float64)
    self.name = name
    self.color = color

    self.path_x = [r[0]]
    self.path_y = [r[1]]

  def record_path(self):
      self.path_x.append(self.r[0])
      self.path_y.append(self.r[1])

class Forces:
  def __init__(self, dim) -> None:
    self.forces = dict()
    self.dim = dim
  
  def registrate(self, func: Callable[[Planet, Planet], npt.NDArray[np.float64]]) -> None:
    self.forces[func.__name__] = func
  
  def calculate(self, pln1: Planet, pln2: Planet) -> npt.NDArray[np.float64]:
    a = np.zeros_like(self.dim, dtype=np.float64)
    for force in self.forces.values():
      a += force(pln1, pln2)
    return a