import csv
import json
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from models import Planet
from typing import Iterable, Optional

# --- НОВАЯ БИНАРНАЯ ЛОГИКА ---

def load_data_binary(
    folder: str = "assets/data_bin",
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> list[Planet]:
    """
    Загружает данные из .npy через mmap.
    include/exclude — фильтрация по именам планет (чтобы не собирать лишние Planet).
    """
    manifest_path = os.path.join(folder, "system_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Loading binary data from {folder} (mmap mode)...")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Backward/forward compatible schema:
    # - old: manifest is list[planet_meta]
    # - new: manifest is dict with key "planets"
    if isinstance(manifest, dict):
        metadata = manifest.get("planets", [])
    else:
        metadata = manifest

    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Bad manifest format: expected list of planets metadata.")

    include_set = set(include) if include is not None else None
    exclude_set = set(exclude) if exclude is not None else None

    # mmap arrays: shape (Planets, Steps, 3)
    try:
        hist_pos = np.load(os.path.join(folder, "positions.npy"), mmap_mode="r")
        hist_vel = np.load(os.path.join(folder, "velocities.npy"), mmap_mode="r")
    except FileNotFoundError:
        raise FileNotFoundError("Binary .npy files missing! Run simulation first.")

    planets: list[Planet] = []
    for i, meta in enumerate(metadata):
        name = meta["name"]

        if include_set is not None and name not in include_set:
            continue
        if exclude_set is not None and name in exclude_set:
            continue

        r_init = np.array(meta.get("last_r", [0, 0, 0]), dtype=np.float64)
        u_init = np.array(meta.get("last_u", [0, 0, 0]), dtype=np.float64)

        p = Planet(
            mass=meta["mass"],
            r=r_init,
            u=u_init,
            name=name,
            color=meta["color"],
        )

        # mmap views (NO COPY)
        p.path_x = hist_pos[i, :, 0]
        p.path_y = hist_pos[i, :, 1]
        p.path_z = hist_pos[i, :, 2]
        p.path_vx = hist_vel[i, :, 0]
        p.path_vy = hist_vel[i, :, 1]
        p.path_vz = hist_vel[i, :, 2]

        planets.append(p)

    print(f"Loaded {len(planets)} planets.")
    return planets


def save_data_binary(planets_meta: list[dict], hist_pos: np.ndarray, hist_vel: np.ndarray, folder: str = "assets/data_bin"):
    """
    Сохраняет данные в бинарном формате .npy (мгновенная запись).
    """
    os.makedirs(folder, exist_ok=True)
    print(f"Saving BINARY data to {folder}/ ...")

    # Транспонируем, если пришло (Steps, Planets, 3) -> (Planets, Steps, 3)
    if hist_pos.shape[0] != len(planets_meta):
        hist_pos = np.transpose(hist_pos, (1, 0, 2))
        hist_vel = np.transpose(hist_vel, (1, 0, 2))

    # Сохраняем огромные массивы одним куском (это ОЧЕНЬ быстро)
    # Формат: (Planets, Steps, 3)
    np.save(os.path.join(folder, "positions.npy"), hist_pos)
    np.save(os.path.join(folder, "velocities.npy"), hist_vel)

    # Сохраняем метаданные
    with open(os.path.join(folder, "system_manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(planets_meta, f, indent=2)
        
    print(f"Saved binary data for {len(planets_meta)} planets.")

def _write_planet_csv(args):
    """
    Вспомогательная функция для записи одного файла в отдельном процессе.
    """
    filename, pos_data, vel_data, header = args
    
    # Объединяем позиции и скорости: (N, 3) + (N, 3) -> (N, 6)
    # Это дешевая операция, так как создает view или копию только для одной планеты
    full_data = np.hstack((pos_data, vel_data))
    
    # np.savetxt работает быстрее циклов Python. 
    # fmt='%.6e' — научная нотация, достаточно 6 знаков (микроны для планет), это быстрее форматировать.
    np.savetxt(
        filename, 
        full_data, 
        delimiter=',', 
        header=header, 
        comments='', # Чтобы header не начинался с #
        fmt='%.6e' 
    )
    return filename

def save_data_from_arrays(
    planets_meta: list[dict], 
    hist_pos: np.ndarray, 
    hist_vel: np.ndarray, 
    folder: str = "assets/data"
):
    """
    Быстрое сохранение результатов numpy-симуляции.
    
    Args:
        planets_meta: Список словарей с метаданными (имя, цвет, масса...)
        hist_pos: Массив (Steps, Planets, 3) или (Planets, Steps, 3)
        hist_vel: Массив (Steps, Planets, 3)
        folder: Папка назначения
    """
    os.makedirs(folder, exist_ok=True)
    print(f"Preparing to save data to {folder}/ ...")

    # 1. Проверяем размерность. Если (Steps, Planets, 3), транспонируем в (Planets, Steps, 3)
    # чтобы было легко брать срезы по планетам.
    if hist_pos.shape[0] != len(planets_meta):
        # Значит первый dim — это steps
        hist_pos = np.transpose(hist_pos, (1, 0, 2))
        hist_vel = np.transpose(hist_vel, (1, 0, 2))

    tasks = []
    
    # 2. Формируем задачи для параллельной записи
    for i, meta in enumerate(planets_meta):
        name = meta['name']
        csv_filename = f"{name}.csv"
        csv_path = os.path.join(folder, csv_filename)
        
        # Обновляем имя файла в метаданных
        meta['csv_file'] = csv_filename
        
        # Данные конкретной планеты
        p_pos = hist_pos[i]
        p_vel = hist_vel[i]
        
        tasks.append((csv_path, p_pos, p_vel, "x,y,z,vx,vy,vz"))

    # 3. Пишем файлы параллельно
    # Используем ProcessPoolExecutor, чтобы обойти GIL и загрузить все ядра CPU форматированием текста
    print(f"Writing CSVs in parallel...")
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_write_planet_csv, tasks))
        
    # 4. Сохраняем манифест (JSON)
    # Нам нужно сохранить метаданные, но у нас нет объектов Planet. 
    # Используем переданный список словарей.
    with open(os.path.join(folder, "system_manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(planets_meta, f, indent=2)

    print(f"Successfully saved {len(results)} files.")

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
