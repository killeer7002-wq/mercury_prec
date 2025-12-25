import numpy as np

# Гравитационный параметр Солнца (m^3/s^2)
GM = 1.32712440018e20 

def get_state_from_kepler(a_au, e, i_deg, w_deg, Omega_deg, M_deg):
    """
    Преобразует Кеплеровы элементы в векторы положения и скорости (в СИ).
    a_au: Большая полуось (в А.Е.)
    e: Эксцентриситет
    i_deg: Наклонение (градусы)
    w_deg: Аргумент перицентра (градусы)
    Omega_deg: Долгота восходящего узла (градусы)
    M_deg: Средняя аномалия (градусы)
    """
    
    # Константы
    AU = 1.496e11
    a = a_au * AU
    
    # Перевод градусов в радианы
    i = np.radians(i_deg)
    w = np.radians(w_deg)
    O = np.radians(Omega_deg)
    M = np.radians(M_deg)
    
    # 1. Решаем уравнение Кеплера для Эксцентрической аномалии (E)
    # Метод Ньютона-Рафсона
    E = M
    for _ in range(10):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < 1e-9: break
            
    # 2. Координаты в орбитальной плоскости (P, Q)
    # Истинная аномалия (nu) не нужна явно, считаем через E
    # r_orb = a * (1 - e * cos(E))
    
    cosE = np.cos(E)
    sinE = np.sin(E)
    
    # Положение в плоскости орбиты
    x_orb = a * (cosE - e)
    y_orb = a * np.sqrt(1 - e**2) * sinE
    
    # Скорость в плоскости орбиты
    # n = mean motion (rad/s)
    n = np.sqrt(GM / a**3)
    r_mag = a * (1 - e * cosE)
    
    vx_orb = - (n * a**2 / r_mag) * sinE
    vy_orb =   (n * a**2 * np.sqrt(1 - e**2) / r_mag) * cosE
    
    # 3. Поворот в эклиптическую систему координат (3D поворот)
    # Матрица поворота R = Rz(-O) * Rx(-i) * Rz(-w)
    
    def rotate(x, y, vx, vy):
        # Вектора в орбитальной плоскости (z=0)
        pos = np.array([x, y, 0.0])
        vel = np.array([vx, vy, 0.0])
        
        # Поворот на аргумент перицентра (w) вокруг Z
        R_w = np.array([
            [np.cos(w), -np.sin(w), 0],
            [np.sin(w),  np.cos(w), 0],
            [0,          0,         1]
        ])
        
        # Поворот на инклинуцию (i) вокруг X
        R_i = np.array([
            [1, 0,          0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i),  np.cos(i)]
        ])
        
        # Поворот на долготу узла (Omega) вокруг Z
        R_O = np.array([
            [np.cos(O), -np.sin(O), 0],
            [np.sin(O),  np.cos(O), 0],
            [0,          0,         1]
        ])
        
        # Полная матрица поворота (обратный порядок умножения для векторов-столбцов, но здесь мы умножаем вектора)
        # Правильная формула для вектора r_vec = R_O @ R_i @ R_w @ r_orb
        R = R_O @ R_i @ R_w
        
        return R @ pos, R @ vel

    r_vec, v_vec = rotate(x_orb, y_orb, vx_orb, vy_orb)
    
    return r_vec, v_vec

def get_j2000_state(name):
    """
    Возвращает r, v для эпохи J2000 (1 Jan 2000).
    Данные: NASA JPL Solar System Dynamics.
    Элементы: a (AU), e, i (deg), w (deg), Omega (deg), M (deg)
    """
    data = {
        "Mercury": (0.387098, 0.205630, 7.005,  29.124, 48.331, 174.796),
        "Venus":   (0.723332, 0.006772, 3.394,  54.884, 76.680, 50.115),
        "Earth":   (1.000000, 0.016708, 0.000,  102.937, 0.0,    357.517), # Для Земли Omega не определена, берем 0
        "Mars":    (1.523679, 0.093400, 1.850,  286.502, 49.558, 19.412),
        "Jupiter": (5.204267, 0.048498, 1.303,  273.867, 100.464, 20.020),
        "Saturn":  (9.582017, 0.055546, 2.485,  339.392, 113.665, 317.020),
        "Uranus":  (19.229411, 0.046381, 0.773, 96.9988, 74.006, 142.2386),
        "Neptune": (30.103661, 0.009456, 1.770, 276.336, 131.784, 256.228)
    }
    
    if name not in data:
        raise ValueError(f"No data for {name}")
        
    params = data[name]
    return get_state_from_kepler(*params)