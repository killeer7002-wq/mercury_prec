import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.style as mplstyle

from models import *

def animate_orbits(
    planets: list[Planet], 
    filename: str = "orbit.gif", 
    fps: int = 30, 
    stride: int = 10,
    trace_length: int = 200
) -> None:
    """
    Args:
        planets: Список объектов Planet после симуляции.
        filename: Имя файла для сохранения (например, .gif или .mp4).
        fps: Кадры в секунду.
        stride: Шаг прореживания данных (рисовать каждую N-ную точку). 
                Это критично, если точек миллионы.
        trace_length: Длина "хвоста" орбиты (в кадрах). Если -1, рисует весь путь.
    """
    
    # 1. Настройка стиля "Космос"
    mplstyle.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Вычисляем границы графика на основе максимального удаления всех планет
    # Чтобы масштаб был фиксирован и не прыгал
    max_range = 0.0
    for p in planets:
        max_x = np.max(np.abs(p.path_x))
        max_y = np.max(np.abs(p.path_y))
        max_range = max(max_range, max_x, max_y)
    
    limit = max_range * 1.1 # +10% отступа
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_title(f"Simulation: {', '.join([p.name for p in planets])}")

    # 2. Подготовка графических элементов (линии и точки)
    lines = [] # Хвосты орбит
    dots = []  # Сами планеты
    names = [] # Подписи

    for p in planets:
        # Линия орбиты (сначала пустая)
        ln, = ax.plot([], [], color=p.color, lw=1, alpha=0.7)
        lines.append(ln)
        
        # Точка планеты (размер зависит от массы, логарифмически для наглядности)
        # Солнце делаем побольше визуально
        marker_size = 8 if p.name == "Sun" else 5
        dot, = ax.plot([], [], 'o', color=p.color, markersize=marker_size)
        dots.append(dot)
        
        # Подпись
        annotation = ax.text(0, 0, p.name, fontsize=9, color=p.color)
        names.append(annotation)

    # Данные симуляции уже в planet.path_x/y, но их много.
    # Берем каждый stride-элемент, чтобы анимация не тормозила.
    # Предполагаем, что длина массивов path у всех планет одинаковая.
    total_steps = len(planets[0].path_x)
    frame_indices = range(0, total_steps, stride)

    def init():
        for ln, dot, txt in zip(lines, dots, names):
            ln.set_data([], [])
            dot.set_data([], [])
            txt.set_position((0,0))
            txt.set_text("")
        return lines + dots + names

    def update(frame_idx):
        for i, p in enumerate(planets):
            # Текущие координаты
            x = p.path_x[frame_idx]
            y = p.path_y[frame_idx]
            
            # Обновляем точку
            dots[i].set_data([x], [y]) # set_data ожидает sequence
            names[i].set_position((x + limit*0.02, y + limit*0.02))
            names[i].set_text(p.name)

            # Обновляем хвост
            # Если trace_length != -1, обрезаем хвост
            start_trace = 0
            if trace_length > 0:
                start_trace = max(0, frame_idx - trace_length * stride) # учитываем stride
                # Но для индексов массива нужно брать правильно срезы
                # Так как frame_idx - это реальный индекс в массиве path
            
            # Чтобы хвост рисовался плавно, берем срез из истории
            # Важно: path_x содержит все точки, нам нужно взять отрезок
            history_x = p.path_x[start_trace : frame_idx+1]
            history_y = p.path_y[start_trace : frame_idx+1]
            
            lines[i].set_data(history_x, history_y)
            
        return lines + dots + names

    # 3. Запуск анимации
    print(f"Generating animation ({len(frame_indices)} frames)... please wait.")
    anim = FuncAnimation(
        fig, 
        update, 
        frames=frame_indices, 
        init_func=init, 
        blit=True, 
        interval=1000/fps
    )

    # Сохранение
    # Для gif нужен pillow, для mp4 нужен ffmpeg
    try:
        if filename.endswith('.mp4'):
            anim.save(filename, writer='ffmpeg', fps=fps, dpi=150)
        else:
            anim.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved to {filename}")
    except Exception as e:
        print(f"Error saving animation: {e}. Showing plot instead.")
        plt.show()