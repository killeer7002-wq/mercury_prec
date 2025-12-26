from utils import load_data_binary
import numpy as np
import matplotlib.pyplot as plt
from models import Planet
from consts import *
import sys

def get_relative_vectors(mercury, sun):
    """
    Берем сохраненные точные данные из симуляции.
    Больше никаких np.gradient!
    """
    # 1. Позиции
    rx_m, ry_m, rz_m = np.array(mercury.path_x), np.array(mercury.path_y), np.array(mercury.path_z)
    rx_s, ry_s, rz_s = np.array(sun.path_x), np.array(sun.path_y), np.array(sun.path_z)
    
    r_rel = np.stack([rx_m - rx_s, ry_m - ry_s, rz_m - rz_s], axis=1)

    # 2. Скорости (Берем напрямую из истории!)
    vx_m, vy_m, vz_m = np.array(mercury.path_vx), np.array(mercury.path_vy), np.array(mercury.path_vz)
    vx_s, vy_s, vz_s = np.array(sun.path_vx), np.array(sun.path_vy), np.array(sun.path_vz)
    
    v_rel = np.stack([vx_m - vx_s, vy_m - vy_s, vz_m - vz_s], axis=1)
    
    return r_rel, v_rel

def analyze_precession():
    if len(sys.argv) == 1:
      planets = load_data_binary("assets/data_bin")
    else:
      planets = load_data_binary(f"assets/{sys.argv[1]}")
    if len(sys.argv) == 4:
      DT = float(sys.argv[3])
    mercury = next(p for p in planets if p.name == "Mercury")
    sun = next(p for p in planets if p.name == "Sun")
    
    print(f"Analyzing Data (Using EXACT velocities)...")
    
    r, v = get_relative_vectors(mercury, sun)
    
    # Вектор эксцентриситета
    dist = np.linalg.norm(r, axis=1)[:, np.newaxis]
    h = np.cross(r, v)
    v_cross_h = np.cross(v, h)
    e_vecs = (v_cross_h / (G * M_SUN)) - (r / dist)
    
    # Угол перигелия
    angles_rad = np.arctan2(e_vecs[:, 1], e_vecs[:, 0])
    angles_unwrap = np.unwrap(angles_rad)
    
    delta_angles = (angles_unwrap - angles_unwrap[0]) * ARCSEC_PER_RAD
    time_years = np.arange(len(delta_angles)) * DT / (365.25*24*3600)
    
    # Статистика
    slope, intercept = np.polyfit(time_years, delta_angles, 1)
    
    measured_rate = slope * 100
    theory_rate = 574.10 
    
    print("\n" + "="*40)
    print(f" MEASURED PRECESSION: {measured_rate:.2f} arcsec/cy")
    print(f" THEORETICAL TARGET:  {theory_rate:.2f} arcsec/cy")
    print(f" ERROR:               {abs(measured_rate - theory_rate):.2f} arcsec/cy")
    print("="*40 + "\n")

    # График
    plt.style.use('default')
    plt.figure(figsize=(10, 6), dpi=100)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # ОПТИМИЗАЦИЯ 1: Рисуем каждую 100-ю точку для "шума", иначе виснет
    step = 100 
    plt.plot(time_years[::step], delta_angles[::step], color='gray', alpha=0.3, label='Oscillation')
    
    # Линии тренда рисуем целиком (они прямые, там всего 2 точки по сути)
    plt.plot(time_years, slope * time_years + intercept, color='red', linewidth=2, label='Simulation')
    plt.plot(time_years, (theory_rate/100)*time_years + intercept, color='blue', linestyle='--', label='Theory')

    plt.title(f'Mercury Perihelion Precession (Exact Data)', fontsize=14)
    plt.xlabel('Time (Years)')
    plt.ylabel('Shift (arcsec)')
    
    # ОПТИМИЗАЦИЯ 2: Явно задаем место легенды, чтобы не искал "best"
    plt.legend(loc='upper left') 
    
    info = f"Result: {measured_rate:.1f}\"/cy\nTarget: {theory_rate:.1f}\"/cy"
    plt.gca().text(0.02, 0.50, info, transform=plt.gca().transAxes,
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    print("Saving plot...")
    if len(sys.argv) == 1:
      plt.savefig('assets/scientific_proof.png')
    else:
      plt.savefig(f'assets/{sys.argv[2]}.png')
    print("Graph saved.")

if __name__ == "__main__":
    analyze_precession()