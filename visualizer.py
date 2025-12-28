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
    Создает и сохраняет анимацию орбит планет.

    Args:
        planets (list[Planet]): Список объектов Planet с данными траекторий
                                после симуляции.
        filename (str): Имя файла для сохранения анимации (например, 'orbit.gif'
                        или 'orbit.mp4').
        fps (int): Количество кадров в секунду для итоговой анимации.
        stride (int): Шаг прореживания данных. Рисуется каждая `stride`-я точка
                      из симуляции, что критически важно для производительности
                      при большом количестве шагов.
        trace_length (int): Длина "хвоста" (траектории) планеты в кадрах
                            анимации. Если значение -1, рисуется вся
                            пройденная траектория.
    """
    
    # 1. Настройка стиля и области графика
    mplstyle.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Вычисление границ для фиксированного масштаба на основе максимального удаления планет
    max_range = 0.0
    for p in planets:
        max_x = np.max(np.abs(p.path_x))
        max_y = np.max(np.abs(p.path_y))
        max_range = max(max_range, max_x, max_y)
    
    limit = max_range * 1.1 # Добавляем 10% отступ
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_title(f"Simulation: {', '.join([p.name for p in planets])}")

    # 2. Инициализация графических элементов
    lines = []   # Линии для хвостов орбит
    dots = []    # Точки, представляющие планеты
    names = []   # Текстовые подписи для планет

    for p in planets:
        # Линия орбиты (изначально пустая)
        ln, = ax.plot([], [], color=p.color, lw=1, alpha=0.7)
        lines.append(ln)
        
        # Точка планеты (размер зависит от массы, для Солнца увеличен)
        marker_size = 8 if p.name == "Sun" else 5
        dot, = ax.plot([], [], 'o', color=p.color, markersize=marker_size)
        dots.append(dot)
        
        # Подпись к планете
        annotation = ax.text(0, 0, p.name, fontsize=9, color=p.color)
        names.append(annotation)

    # Прореживание данных для ускорения анимации
    # Предполагается, что у всех планет одинаковое количество точек в траектории.
    total_steps = len(planets[0].path_x)
    frame_indices = range(0, total_steps, stride)

    def init():
        """Инициализирующая функция для анимации, очищает все элементы."""
        for ln, dot, txt in zip(lines, dots, names):
            ln.set_data([], [])
            dot.set_data([], [])
            txt.set_position((0,0))
            txt.set_text("")
        return lines + dots + names

    def update(frame_idx):
        """Функция обновления для каждого кадра анимации."""
        for i, p in enumerate(planets):
            # Текущие координаты из полного набора данных
            x = p.path_x[frame_idx]
            y = p.path_y[frame_idx]
            
            # Обновление положения точки и подписи
            dots[i].set_data([x], [y])
            names[i].set_position((x + limit*0.02, y + limit*0.02))
            names[i].set_text(p.name)

            # Обновление хвоста траектории
            start_trace = 0
            if trace_length > 0:
                # Определяем начальный индекс для среза истории, чтобы ограничить длину хвоста
                start_trace = max(0, frame_idx - trace_length * stride)
            
            # Извлекаем срез из истории для отрисовки хвоста
            history_x = p.path_x[start_trace : frame_idx+1]
            history_y = p.path_y[start_trace : frame_idx+1]
            
            lines[i].set_data(history_x, history_y)
            
        return lines + dots + names

    # 3. Создание и сохранение анимации
    print(f"Generating animation ({len(frame_indices)} frames)... please wait.")
    anim = FuncAnimation(
        fig, 
        update, 
        frames=frame_indices, 
        init_func=init, 
        blit=True, # Используем blitting для оптимизации производительности
        interval=1000/fps
    )

    # Сохранение в файл
    try:
        if filename.endswith('.mp4'):
            anim.save(filename, writer='ffmpeg', fps=fps, dpi=150)
        else:
            anim.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved to {filename}")
    except Exception as e:
        print(f"Error saving animation: {e}. Showing plot instead.")
        plt.show()
