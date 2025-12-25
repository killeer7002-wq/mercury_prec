import pickle
import numpy as np
import matplotlib.pyplot as plt
from models import Planet

# --- НАСТРОЙКИ ---
# ВАЖНО: Это число должно совпадать с dt в simulation.py!
DT = 600  # Если в симуляции был 1 час, ставьте 3600. Если 600 - ставьте 600.

# Константы
G = 6.674e-11
M_SUN = 1.989e30
ARCSEC_PER_RAD = 206265 # Угловых секунд в радиане

def load_data(filename: str = "assets/simulation_data.pkl") -> list[Planet]:
    with open(filename, 'rb') as f:
        planets = pickle.load(f)
    return planets

def calculate_eccentricity_vector(r, v, mu):
    """Считает вектор Рунге-Ленца (направление на перигелий)"""
    dist = np.linalg.norm(r, axis=1)[:, np.newaxis]
    h = np.cross(r, v)
    v_cross_h = np.cross(v, h)
    e_vec = (v_cross_h / mu) - (r / dist)
    return e_vec

def analyze_precession():
    planets = load_data()
    mercury = next(p for p in planets if p.name == "Mercury")
    
    print(f"Analyzing Mercury data with dt={DT}s...")
    
    # 1. Подготовка данных
    rx = np.array(mercury.path_x)
    ry = np.array(mercury.path_y)
    rz = np.zeros_like(rx) # Если симуляция была 2D
    
    # 2. Восстановление скоростей (V = dR / dt)
    vx = np.gradient(rx, DT)
    vy = np.gradient(ry, DT)
    vz = np.zeros_like(vx)
    
    r = np.stack([rx, ry, rz], axis=1)
    v = np.stack([vx, vy, vz], axis=1)
    
    # 3. Расчет угла перигелия
    e_vecs = calculate_eccentricity_vector(r, v, G * M_SUN)
    angles_rad = np.arctan2(e_vecs[:, 1], e_vecs[:, 0])
    angles_unwrap = np.unwrap(angles_rad)
    
    # Перевод в угловые секунды (смещение от начала)
    delta_angles = (angles_unwrap - angles_unwrap[0]) * ARCSEC_PER_RAD
    
    # Время в годах
    time_years = np.arange(len(delta_angles)) * DT / (365*24*3600)
    
    # 4. Визуализация (Стиль "Научная статья")
    plt.style.use('default') # Светлая тема
    plt.figure(figsize=(10, 7), dpi=120)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # --- Сырые данные (Oscillations) ---
    plt.plot(time_years, delta_angles, color='#b0bec5', alpha=0.5, linewidth=1, label='Raw Oscillation (Newtonian Noise)')
    
    # --- Линейная регрессия (Тренд) ---
    # Используем полином 1-й степени (y = kx + b) для поиска точного наклона
    slope, intercept = np.polyfit(time_years, delta_angles, 1)
    
    # Теоретическое значение (Полная прецессия: Юпитер + Венера + Земля + ОТО)
    THEORY_RATE = 574.1 # arcsec/century
    
    measured_rate = slope * 100 # переводим в "секунд за столетие"
    
    # Рисуем линию тренда
    plt.plot(time_years, slope * time_years + intercept, color='#d32f2f', linewidth=2.5, linestyle='-', label='Measured Trend')
    
    # Рисуем теоретическую линию (для сравнения)
    plt.plot(time_years, (THEORY_RATE/100)*time_years, color='#1976d2', linewidth=2, linestyle='--', label=f'Theory ({THEORY_RATE}"/cy)')

    # --- Оформление ---
    plt.title(f'Mercury Perihelion Precession Analysis\n(dt={DT}s)', fontsize=14, pad=15)
    plt.xlabel('Time (Earth Years)', fontsize=12)
    plt.ylabel('Perihelion Shift (arcseconds)', fontsize=12)
    
    # Информационное окно с цифрами
    info_text = (
        f"Measured Rate: {measured_rate:.2f}\"/century\n"
        f"Theoretical Rate: {THEORY_RATE:.2f}\"/century\n"
        f"Error: {abs(measured_rate - THEORY_RATE):.2f}\""
    )
    plt.gca().text(0.02, 0.95, info_text, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=1)
    plt.tight_layout()
    
    output_file = 'assets/scientific_proof.png'
    plt.savefig(output_file)
    print(f"Graph saved to {output_file}")
    print(f"--- RESULTS ---")
    print(info_text)

if __name__ == "__main__":
    analyze_precession()