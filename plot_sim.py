import matplotlib.pyplot as plt

from models import Planet
from utils import load_data_binary
from visualizer import animate_orbits

__all__ = ["plot"]


def plot_static(planets: list[Planet], filename: str = "./assets/sim.png") -> None:
    """
    Создает и сохраняет статическое изображение орбит планет.

    Функция отрисовывает полный путь каждой планеты и ее текущее положение
    на графике Matplotlib, после чего сохраняет результат в файл.

    Args:
        planets (list[Planet]): Список объектов планет для отрисовки.
        filename (str): Путь для сохранения итогового изображения (PNG).
    """
    plt.figure(figsize=(8, 8))
    for planet in planets:
        plt.plot(planet.path_x, planet.path_y, color=planet.color, label=planet.name)
        # Отмечаем последнее положение планеты
        plt.plot(planet.r[0], planet.r[1], 'o', color=planet.color)

    plt.title("Орбиты планет Солнечной системы")
    plt.xlabel("X (метры)")
    plt.ylabel("Y (метры)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)


def plot(all_planets: list[Planet],
         filename_png: str = "./assets/sim.png",
         filename_gif: str = "./assets.orbit.gif") -> None:
    """
    Генерирует статическую и анимированную визуализацию орбит.

    Сначала создает статическое изображение всех орбит с помощью `plot_static`.
    Затем создает анимированный GIF для планет внутренней Солнечной системы
    (до Юпитера включительно), чтобы сохранить маcштаб и видимость орбит
    внутренних планет, таких как Меркурий.

    Args:
        all_planets (list[Planet]): Полный список объектов планет.
        filename_png (str): Путь для сохранения статического изображения.
        filename_gif (str): Путь для сохранения анимированного GIF-файла.
    """
    plot_static(all_planets, filename_png)

    # Для анимации используется срез планет до Юпитера.
    # Это необходимо, так как орбита Нептуна слишком велика и при общем
    # масштабировании орбита Меркурия становится практически неразличимой.
    # Индексы: 0-Солнце, 1-Меркурий, 2-Венера, 3-Земля, 4-Марс, 5-Юпитер.
    inner_system = all_planets[:6]

    print(f"Создание анимации для внутренней Солнечной системы: {", ".join([p.name for p in inner_system])}")
    animate_orbits(
        inner_system,
        filename=filename_gif,
        fps=30,
        stride=24 * 40,  # Один кадр на каждые 40 дней симуляции (при dt=1 час)
        trace_length=400
    )


if __name__ == "__main__":
    # Загрузка данных симуляции и запуск отрисовки
    data_path = "assets/data_bin"
    print(f"Загрузка данных из '{data_path}'...")
    planets_data = load_data_binary(data_path)
    print("Запуск визуализации...")
    plot(planets_data)
    print("Визуализация завершена.")