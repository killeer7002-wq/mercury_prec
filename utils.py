import csv
import json
import os
import numpy as np

from models import Planet

def save_data(planets: list[Planet], folder: str = "assets/data"):
    """Сохраняет данные в CSV (траектории) и JSON (свойства планет)"""
    os.makedirs(folder, exist_ok=True)
    
    metadata = []
    
    print(f"Saving text data to {folder}/ ...")
    
    for p in planets:
        # 1. Сохраняем траекторию в CSV
        csv_filename = f"{p.name}.csv"
        csv_path = os.path.join(folder, csv_filename)
        
        # Считаем, что длины всех массивов совпадают
        rows = zip(p.path_x, p.path_y, p.path_z, p.path_vx, p.path_vy, p.path_vz)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'vx', 'vy', 'vz']) # Заголовки
            writer.writerows(rows)
            
        # 2. Готовим метаданные
        metadata.append({
            "name": p.name,
            "color": p.color,
            "mass": p.mass,
            "csv_file": csv_filename,
            # Сохраним последнее состояние для инициализации r/u при загрузке (опционально)
            "last_r": list(p.r),
            "last_u": list(p.u)
        })
        
    # 3. Сохраняем описание системы в JSON
    with open(os.path.join(folder, "system_manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved {len(planets)} planets.")

def load_data(folder: str = "assets/data") -> list[Planet]:
    manifest_path = os.path.join(folder, "system_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    planets = []
    print(f"Loading {len(metadata)} planets from text files...")

    for meta in metadata:
        # Восстанавливаем базовый объект
        # r/u здесь не так важны, так как мы сейчас перезапишем историю, 
        # но для корректности конструктора берем из метаданных или нули.
        r_init = np.array(meta.get("last_r", [0,0,0]), dtype=np.float64)
        u_init = np.array(meta.get("last_u", [0,0,0]), dtype=np.float64)
        
        p = Planet(
            mass=meta["mass"],
            r=r_init,
            u=u_init,
            name=meta["name"],
            color=meta["color"]
        )
        
        # Читаем CSV с траекторией. 
        # Используем pandas, если есть, или стандартный csv/numpy для скорости.
        # Ниже вариант на чистом numpy (быстрее для чисел):
        csv_path = os.path.join(folder, meta["csv_file"])
        
        # skiprows=1 пропускает заголовок
        # usecols=(0,1,2,3,4,5) читает x,y,z,vx,vy,vz
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        
        # Данные загружаются как (N, 6). Транспонируем для удобства
        # data[:, 0] -> весь столбец x
        p.path_x = data[:, 0].tolist()
        p.path_y = data[:, 1].tolist()
        p.path_z = data[:, 2].tolist()
        p.path_vx = data[:, 3].tolist()
        p.path_vy = data[:, 4].tolist()
        p.path_vz = data[:, 5].tolist()
        
        planets.append(p)
        print(f"  - Loaded {p.name}: {len(p.path_x)} points")
        
    return planets
