"""
Этот модуль определяет основные классы для симуляции: `Planet` и `Forces`.
`Planet` представляет собой небесное тело с его физическими свойствами и историей движения.
`Forces` служит для регистрации и расчета совокупного действия различных сил.
"""

from collections.abc import Callable
from typing import Optional
import numpy as np
import numpy.typing as npt

__all__ = ["Planet", "Forces"]

class Planet:
  """Представляет небесное тело в симуляции.

  Хранит физические параметры, такие как масса, положение, скорость,
  ускорение, а также историю перемещений и скоростей для последующего
  анализа и визуализации.

  Attributes:
      mass (float): Масса тела в кг.
      r (npt.NDArray[np.float64]): Вектор положения [x, y, z] в метрах.
      u (npt.NDArray[np.float64]): Вектор скорости [vx, vy, vz] в м/с.
      a (npt.NDArray[np.float64]): Вектор ускорения [ax, ay, az] в м/с^2.
      name (str): Имя тела.
      color (str): Цвет для визуализации.
      path_x (list[float]): История координат по оси X.
      path_y (list[float]): История координат по оси Y.
      path_z (list[float]): История координат по оси Z.
      path_vx (list[float]): История проекций скорости на ось X.
      path_vy (list[float]): История проекций скорости на ось Y.
      path_vz (list[float]): История проекций скорости на ось Z.
  """
  r: npt.NDArray[np.float64]
  u: npt.NDArray[np.float64]
  a: npt.NDArray[np.float64]

  def __init__(self,
               mass: float,
               r: npt.NDArray[np.float64],
               u: Optional[npt.NDArray[np.float64]] = None,
               a: Optional[npt.NDArray[np.float64]] = None,
               name: str = "unknown",
               color: str = "red",):
    """Инициализирует объект Planet.

    Args:
        mass (float): Масса тела в кг.
        r (npt.NDArray[np.float64]): Начальный вектор положения [x, y, z].
        u (Optional[npt.NDArray[np.float64]], optional): Начальный вектор скорости [vx, vy, vz].
            Если None, инициализируется нулями. Defaults to None.
        a (Optional[npt.NDArray[np.float64]], optional): Начальный вектор ускорения [ax, ay, az].
            Если None, инициализируется нулями. Defaults to None.
        name (str, optional): Имя тела. Defaults to "unknown".
        color (str, optional): Цвет для визуализации. Defaults to "red".
    """
    self.mass = mass
    self.r = r
    if u is None:
      self.u = np.zeros_like(r, dtype=np.float64)
    else:
      self.u = u
    if a is None:
      self.a = np.zeros_like(r, dtype=np.float64)
    else:
      self.a = a
    self.name = name
    self.color = color

    # Инициализация списков для хранения истории координат
    self.path_x = [r[0]]
    self.path_y = [r[1]]
    self.path_z = [r[2]] if r.shape[0] > 2 else [0.0]
    
    # Инициализация списков для хранения истории скоростей
    self.path_vx = [self.u[0]]
    self.path_vy = [self.u[1]]
    self.path_vz = [self.u[2]] if self.u.shape[0] > 2 else [0.0]

  def record_state(self):
    """Записывает текущее состояние (положение и скорость) в историю."""
    self.path_x.append(self.r[0])
    self.path_y.append(self.r[1])
    if self.r.shape[0] > 2: self.path_z.append(self.r[2])
    
    self.path_vx.append(self.u[0])
    self.path_vy.append(self.u[1])
    if self.u.shape[0] > 2: self.path_vz.append(self.u[2])

class Forces:
  """Менеджер для регистрации и вычисления сил, действующих на объекты.

  Этот класс позволяет регистрировать различные функции, вычисляющие силы
  (в виде ускорений), и затем суммировать их для получения полного ускорения,
  действующего на тело.

  Attributes:
      forces (dict): Словарь, где ключи - имена функций, а значения - сами
                     функции для расчета сил.
      dim (any): Размерность пространства, используется для создания
                 векторов ускорения нужной формы.
  """
  def __init__(self, dim) -> None:
    """Инициализирует менеджер сил.

    Args:
        dim (any): Объект, определяющий размерность пространства
                   (например, numpy массив нужной длины).
    """
    self.forces = dict()
    self.dim = dim
  
  def registrate(self, func: Callable[[Planet, Planet], npt.NDArray[np.float64]]) -> None:
    """Регистрирует новую функцию для расчета силы.

    Args:
        func (Callable): Функция, принимающая два объекта Planet и
                         возвращающая вектор ускорения, вызванного этой силой.
                         Имя функции используется как ключ для регистрации.
    """
    self.forces[func.__name__] = func
  
  def calculate(self, pln1: Planet, pln2: Planet) -> npt.NDArray[np.float64]:
    """Вычисляет суммарное ускорение, действующее на pln1 со стороны pln2.

    Метод итерируется по всем зарегистрированным функциям сил, вызывает их
    и суммирует результирующие векторы ускорений.

    Args:
        pln1 (Planet): Тело, на которое действует сила.
        pln2 (Planet): Тело, которое является источником силы.

    Returns:
        npt.NDArray[np.float64]: Вектор полного ускорения, действующего на pln1.
    """
    a = np.zeros_like(self.dim, dtype=np.float64)
    for force in self.forces.values():
      a += force(pln1, pln2)
    return a
