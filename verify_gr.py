import pickle
import numpy as np
import matplotlib.pyplot as plt
from models import Planet

# Константы для расчетов
G = 6.674e-11
M_SUN = 1.989e30
ARCSEC_PER_RAD = 206265 # Сколько угловых секунд в радиане

def load_data(filename: str = "assets/simulation_data.pkl") -> list[Planet]:
    with open(filename, 'rb') as f:
        planets = pickle.load(f)
    return planets

def calculate_eccentricity_vector(r, v, mu):
    """
    Считает вектор эксцентриситета (Рунге-Ленца) для каждого момента времени.
    e = (1/mu) * ((v x h) - mu * (r/|r|)), где h = r x v
    Показывает направление на перигелий.
    """
    # r и v - массивы формы (N, 3)
    dist = np.linalg.norm(r, axis=1)[:, np.newaxis]
    
    # Удельный угловой момент h = r x v
    h = np.cross(r, v)
    
    # v x h
    v_cross_h = np.cross(v, h)
    
    # Формула вектора Лапласа-Рунге-Ленца
    e_vec = (v_cross_h / mu) - (r / dist)
    return e_vec

def analyze_precession():
    planets = load_data()
    mercury = next(p for p in planets if p.name == "Mercury")
    
    print("Analyzing Mercury data...")
    
    # Преобразуем списки в numpy массивы (N, 3)
    # simulation.py сохраняет path_x и path_y. Если вы перешли на 3D, нужно проверить path_z
    # Предположим пока 2D данные в path_x, path_y, заполним z нулями
    
    rx = np.array(mercury.path_x)
    ry = np.array(mercury.path_y)
    rz = np.zeros_like(rx)
    
    # Нам нужны скорости. 
    # ВНИМАНИЕ: simulation.py по умолчанию сохраняет только ПУТЬ (r), но не скорость (u) на каждом шаге.
    # Чтобы посчитать вектор эксцентриситета точно, нам нужны скорости.
    # Мы можем восстановить их численно: v = (r(t+1) - r(t-1)) / 2dt
    # Или просто (r(i+1) - r(i)) / dt
    
    dt = 600 # Тот же, что в симуляции! ВАЖНО УКАЗАТЬ ВЕРНЫЙ
    
    # Численная производная для скорости
    vx = np.gradient(rx, dt)
    vy = np.gradient(ry, dt)
    vz = np.zeros_like(vx)
    
    r = np.stack([rx, ry, rz], axis=1)
    v = np.stack([vx, vy, vz], axis=1)
    
    # Считаем вектор эксцентриситета
    e_vecs = calculate_eccentricity_vector(r, v, G * M_SUN)
    
    # Угол вектора (аргумент перигелия)
    angles_rad = np.arctan2(e_vecs[:, 1], e_vecs[:, 0])
    
    # "Разворачиваем" фазу, чтобы не было скачков 360->0
    angles_unwrap = np.unwrap(angles_rad)
    
    # Переводим в угловые секунды (смещение от начала)
    # Вычитаем начальный угол, чтобы график шел из 0
    delta_angles = (angles_unwrap - angles_unwrap[0]) * ARCSEC_PER_RAD
    
    # Время в годах
    time_years = np.arange(len(delta_angles)) * dt / (365*24*3600)
    
    # --- СТРОИМ ГРАФИК ---
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    # Рисуем сырые данные (они будут шуметь из-за осцилляций орбиты)
    plt.plot(time_years, delta_angles, color='cyan', alpha=0.3, label='Raw Data (Oscillating)')
    
    # Сглаживаем (скользящее среднее), чтобы увидеть тренд
    # Окно сглаживания = 1 орбитальный период Меркурия (~0.24 года)
    window = int((0.24 * 365 * 24 * 3600) / dt) 
    if window < 1: window = 1
    
    smoothed = np.convolve(delta_angles, np.ones(window)/window, mode='valid')
    t_smooth = time_years[window-1:]
    
    plt.plot(t_smooth, smoothed, color='white', linewidth=2, label='Average Precession Trend')
    
    # --- ТЕОРИЯ ---
    # Теоретическая скорость: ~574 угловых секунд в столетие (Ньютон + Эйнштейн)
    # Если мы моделируем ВСЕ планеты.
    # Если только Солнце+Меркурий (без Юпитера), то будет 43" в столетие.
    
    theory_rate = 574.0 # arcsec / 100 years (Full System)
    # theory_rate = 43.0 # arcsec / 100 years (Only Sun+Mercury)
    
    theory_line = t_smooth * (theory_rate / 100.0)
    plt.plot(t_smooth, theory_line, '--', color='magenta', label=f'Theory ({theory_rate}"/cy)')

    plt.title(f'Mercury Perihelion Shift (Real C, dt={dt}s)')
    plt.xlabel('Time (Earth Years)')
    plt.ylabel('Shift (arcseconds)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('assets/scientific_proof.png')
    print("Graph saved to assets/scientific_proof.png")
    
    # Вывод результата наклона
    fit_slope = (smoothed[-1] - smoothed[0]) / (t_smooth[-1] - t_smooth[0]) * 100
    print(f"Measured Precession Rate: {fit_slope:.2f} arcsec/century")
    print(f"Theoretical Rate: {theory_rate:.2f} arcsec/century")

if __name__ == "__main__":
    analyze_precession()