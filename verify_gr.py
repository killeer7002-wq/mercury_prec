"""This script analyzes the simulation data to verify the general relativistic
precession of Mercury's perihelion. It compares the measured precession rate
from the simulation with the theoretical value."""
from utils import load_data_binary
import numpy as np
import matplotlib.pyplot as plt
from models import Planet
from consts import *
import sys

def get_relative_vectors(mercury, sun):
    """
    Calculates the relative position and velocity vectors of Mercury with respect to the Sun.

    Args:
        mercury (Planet): The Mercury planet object.
        sun (Planet): The Sun planet object.

    Returns:
        tuple: A tuple containing:
            - r_rel (np.ndarray): The relative position vectors.
            - v_rel (np.ndarray): The relative velocity vectors.
    """
    # 1. Positions
    rx_m, ry_m, rz_m = np.array(mercury.path_x), np.array(mercury.path_y), np.array(mercury.path_z)
    rx_s, ry_s, rz_s = np.array(sun.path_x), np.array(sun.path_y), np.array(sun.path_z)
    
    r_rel = np.stack([rx_m - rx_s, ry_m - ry_s, rz_m - rz_s], axis=1)

    # 2. Velocities (from history)
    vx_m, vy_m, vz_m = np.array(mercury.path_vx), np.array(mercury.path_vy), np.array(mercury.path_vz)
    vx_s, vy_s, vz_s = np.array(sun.path_vx), np.array(sun.path_vy), np.array(sun.path_vz)
    
    v_rel = np.stack([vx_m - vx_s, vy_m - vy_s, vz_m - vz_s], axis=1)
    
    return r_rel, v_rel

def analyze_precession(folder="assets/data_bin", output_filename="assets/scientific_proof.png", dt_sim=DT):
    """
    Analyzes the precession of Mercury's perihelion.

    Args:
        folder (str, optional): The folder containing the simulation data. Defaults to "assets/data_bin".
        output_filename (str, optional): The path to save the output plot. Defaults to "assets/scientific_proof.png".
        dt_sim (float, optional): The simulation time step. Defaults to the value from consts.py.
    """
    planets = load_data_binary(folder)
    mercury = next(p for p in planets if p.name == "Mercury")
    sun = next(p for p in planets if p.name == "Sun")
    
    print(f"Analyzing Data (Using EXACT velocities)...")
    
    r, v = get_relative_vectors(mercury, sun)
    
    # Eccentricity vector
    dist = np.linalg.norm(r, axis=1)[:, np.newaxis]
    h = np.cross(r, v)
    v_cross_h = np.cross(v, h)
    e_vecs = (v_cross_h / (G * M_SUN)) - (r / dist)
    
    # Angle of perihelion
    angles_rad = np.arctan2(e_vecs[:, 1], e_vecs[:, 0])
    angles_unwrap = np.unwrap(angles_rad)
    
    delta_angles = (angles_unwrap - angles_unwrap[0]) * ARCSEC_PER_RAD
    time_years = np.arange(len(delta_angles)) * dt_sim / (365.25*24*3600)
    
    # Statistics
    slope, intercept = np.polyfit(time_years, delta_angles, 1)
    
    measured_rate = slope * 100 # arcsec/century
    # The theoretical rate of 574.10 arcsec/century is the sum of contributions from all planets and GR.
    # The GR contribution alone is about 43 arcsec/century.
    theory_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 574.10
    
    print("\n" + "="*40)
    print(f" MEASURED PRECESSION: {measured_rate:.2f} arcsec/cy")
    print(f" THEORETICAL TARGET:  {theory_rate:.2f} arcsec/cy")
    print(f" ERROR:               {abs(measured_rate - theory_rate):.2f} arcsec/cy")
    print("="*40 + "\n")

    # Plotting
    plt.style.use('default')
    plt.figure(figsize=(10, 6), dpi=100)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Optimization 1: Plot every 100th point for the noisy data
    step = 100 
    plt.plot(time_years[::step], delta_angles[::step], color='gray', alpha=0.3, label='Oscillation')
    
    # Plot trend lines
    plt.plot(time_years, slope * time_years + intercept, color='red', linewidth=2, label='Simulation')
    plt.plot(time_years, (theory_rate/100)*time_years + intercept, color='blue', linestyle='--', label='Theory')

    plt.title(f'Mercury Perihelion Precession (Exact Data)', fontsize=14)
    plt.xlabel('Time (Years)')
    plt.ylabel('Shift (arcsec)')
    
    # Optimization 2: Explicitly set legend location
    plt.legend(loc='upper left') 
    
    info = f"Result: {measured_rate:.1f}\"/cy\nTarget: {theory_rate:.1f}\"/cy"
    plt.gca().text(0.02, 0.50, info, transform=plt.gca().transAxes,
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    print("Saving plot...")
    plt.savefig(output_filename)
    print("Graph saved.")

if __name__ == "__main__":
    folder_arg = sys.argv[1] if len(sys.argv) > 1 else "data_bin"
    output_arg = sys.argv[2] if len(sys.argv) > 2 else "assets/scientific_proof.png"
    dt_arg = float(sys.argv[3]) if len(sys.argv) > 3 else DT
    
    if not output_arg.endswith('.png'):
        output_arg = f"assets/{output_arg}.png"
        
    if not folder_arg.startswith('assets/'):
        folder_arg = f"assets/{folder_arg}"

    analyze_precession(folder=folder_arg, output_filename=output_arg, dt_sim=dt_arg)
