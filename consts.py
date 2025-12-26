import numpy as np

G = 6.674e-11
GM = 1.32712440018e20 # Гравитационный параметр Солнца (м^3/с^2)
C = 299.8e6
M_SUN = 1.989e30
AU = 1.496e11
A_MERCURY = 0.387098 * AU # Средняя большая полуось Меркурия (м)
DT = 60
YEARS_SIM = 100
DIM = np.array([0, 0, 0])
ARCSEC_PER_RAD = 206264.8