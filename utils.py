import csv
import json
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from models import Planet
from typing import Iterable, Optional


def load_data_binary(
    folder: str = "assets/data_bin",
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> list[Planet]:
    """
    Загружает данные о планетах из бинарных файлов .npy с использованием memory-mapping.

    Этот метод значительно быстрее, чем загрузка из CSV, так как избегает парсинга текста
    и копирования данных в память.

    Args:
        folder (str): Путь к папке, содержащей файлы 'positions.npy', 'velocities.npy'
                      и 'system_manifest.json'.
        include (Optional[Iterable[str]]): Список имен планет для загрузки. Если указан,
                                           загружаются только эти планеты.
        exclude (Optional[Iterable[str]]): Список имен планет, которые нужно исключить
                                           из загрузки.

    Returns:
        list[Planet]: Список объектов Planet с загруженными траекториями.

    Raises:
        FileNotFoundError: Если не найден манифест или файлы .npy.
        ValueError: Если формат манифеста некорректен.
    """
    manifest_path = os.path.join(folder, "system_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Loading binary data from {folder} (mmap mode)...")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Обратная и прямая совместимость схемы:
    # - старая: manifest - это list[planet_meta]
    # - новая: manifest - это dict с ключом "planets"
    if isinstance(manifest, dict):
        metadata = manifest.get("planets", [])
    else:
        metadata = manifest

    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Bad manifest format: expected list of planets metadata.")

    include_set = set(include) if include is not None else None
    exclude_set = set(exclude) if exclude is not None else None

    # Загрузка массивов в режиме memory-mapping: форма (Planets, Steps, 3)
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

        # Создаются "виды" (views) через mmap без копирования данных в память
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
    Сохраняет результаты симуляции в бинарном формате .npy.

    Этот метод обеспечивает быструю запись больших объемов данных, сохраняя
    массивы позиций и скоростей в отдельные .npy файлы и метаданные в JSON.

    Args:
        planets_meta (list[dict]): Список словарей с метаданными планет
                                   (имя, масса, цвет и т.д.).
        hist_pos (np.ndarray): Массив с историей позиций. Ожидаемая форма:
                               (Planets, Steps, 3) или (Steps, Planets, 3).
        hist_vel (np.ndarray): Массив с историей скоростей. Ожидаемая форма:
                               (Planets, Steps, 3) или (Steps, Planets, 3).
        folder (str): Путь к папке для сохранения данных.
    """
    os.makedirs(folder, exist_ok=True)
    print(f"Saving BINARY data to {folder}/ ...")

    # Транспонирование, если данные пришли в формате (Steps, Planets, 3)
    if hist_pos.shape[0] != len(planets_meta):
        hist_pos = np.transpose(hist_pos, (1, 0, 2))
        hist_vel = np.transpose(hist_vel, (1, 0, 2))

    # Сохранение массивов в формате (Planets, Steps, 3)
    np.save(os.path.join(folder, "positions.npy"), hist_pos)
    np.save(os.path.join(folder, "velocities.npy"), hist_vel)

    # Сохранение метаданных
    with open(os.path.join(folder, "system_manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(planets_meta, f, indent=2)
        
    print(f"Saved binary data for {len(planets_meta)} planets.")

def _write_planet_csv(args):
    """
    Вспомогательная функция для параллельной записи данных одной планеты в CSV файл.

    Принимает кортеж с аргументами, чтобы быть совместимой с `executor.map`.

    Args:
        args (tuple): Кортеж, содержащий:
            - filename (str): Путь к CSV файлу.
            - pos_data (np.ndarray): Массив позиций (N, 3).
            - vel_data (np.ndarray): Массив скоростей (N, 6).
            - header (str): Заголовок для CSV файла.

    Returns:
        str: Путь к записанному файлу.
    """
    filename, pos_data, vel_data, header = args
    
    # Объединение данных о позициях и скоростях в одну матрицу
    full_data = np.hstack((pos_data, vel_data))
    
    # Использование np.savetxt для быстрой записи в файл
    np.savetxt(
        filename, 
        full_data, 
        delimiter=',', 
        header=header, 
        comments='', # Отключает добавление '#' к заголовку
        fmt='%.6e'  # Научная нотация для точности и скорости
    )
    return filename

def save_data_from_arrays(
    planets_meta: list[dict], 
    hist_pos: np.ndarray, 
    hist_vel: np.ndarray, 
    folder: str = "assets/data"
):
    """
    Сохраняет результаты симуляции из numpy-массивов в CSV файлы параллельно.

    Args:
        planets_meta (list[dict]): Список словарей с метаданными планет.
        hist_pos (np.ndarray): Массив с историей позиций. Ожидаемая форма:
                               (Steps, Planets, 3) или (Planets, Steps, 3).
        hist_vel (np.ndarray): Массив с историей скоростей. Ожидаемая форма:
                               (Steps, Planets, 3) или (Planets, Steps, 3).
        folder (str): Папка для сохранения CSV файлов и манифеста.
    """
    os.makedirs(folder, exist_ok=True)
    print(f"Preparing to save data to {folder}/ ...")

    # Если данные в формате (Steps, Planets, 3), транспонируем для удобства итерации по планетам
    if hist_pos.shape[0] != len(planets_meta):
        hist_pos = np.transpose(hist_pos, (1, 0, 2))
        hist_vel = np.transpose(hist_vel, (1, 0, 2))

    tasks = []
    
    # Формирование задач для параллельной записи
    for i, meta in enumerate(planets_meta):
        name = meta['name']
        csv_filename = f"{name}.csv"
        csv_path = os.path.join(folder, csv_filename)
        
        meta['csv_file'] = csv_filename
        
        # Данные для конкретной планеты
        p_pos = hist_pos[i]
        p_vel = hist_vel[i]
        
        tasks.append((csv_path, p_pos, p_vel, "x,y,z,vx,vy,vz"))

    # Параллельная запись файлов с использованием ProcessPoolExecutor для обхода GIL
    print(f"Writing CSVs in parallel...")
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_write_planet_csv, tasks))
        
    # Сохранение манифеста с метаданными
    with open(os.path.join(folder, "system_manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(planets_meta, f, indent=2)

    print(f"Successfully saved {len(results)} files.")

def save_data(planets: list[Planet], folder: str = "assets/data"):
    """
    Сохраняет данные о траекториях планет в CSV файлы и метаданные в JSON.

    Для каждой планеты создается отдельный CSV файл с историей ее позиций и скоростей.
    Также создается файл 'system_manifest.json', содержащий метаданные всех планет.

    Args:
        planets (list[Planet]): Список объектов Planet, данные которых нужно сохранить.
        folder (str): Папка для сохранения файлов.
    """
    os.makedirs(folder, exist_ok=True)
    
    metadata = []
    
    print(f"Saving text data to {folder}/ ...")
    
    for p in planets:
        # Сохранение траектории в CSV
        csv_filename = f"{p.name}.csv"
        csv_path = os.path.join(folder, csv_filename)
        
        rows = zip(p.path_x, p.path_y, p.path_z, p.path_vx, p.path_vy, p.path_vz)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'vx', 'vy', 'vz']) # Заголовок
            writer.writerows(rows)
            
        # Сбор метаданных
        metadata.append({
            "name": p.name,
            "color": p.color,
            "mass": p.mass,
            "csv_file": csv_filename,
            "last_r": list(p.r),
            "last_u": list(p.u)
        })
        
    # Сохранение метаданных системы в JSON
    with open(os.path.join(folder, "system_manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved {len(planets)} planets.")

def load_data(folder: str = "assets/data") -> list[Planet]:
    """
    Загружает данные о планетах из CSV файлов на основе манифеста.

    Args:
        folder (str): Папка, содержащая 'system_manifest.json' и соответствующие
                      CSV файлы с траекториями.

    Returns:
        list[Planet]: Список объектов Planet с загруженными данными.

    Raises:
        FileNotFoundError: Если файл манифеста не найден.
    """
    manifest_path = os.path.join(folder, "system_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    planets = []
    print(f"Loading {len(metadata)} planets from text files...")

    for meta in metadata:
        # Восстановление базового объекта Planet
        r_init = np.array(meta.get("last_r", [0,0,0]), dtype=np.float64)
        u_init = np.array(meta.get("last_u", [0,0,0]), dtype=np.float64)
        
        p = Planet(
            mass=meta["mass"],
            r=r_init,
            u=u_init,
            name=meta["name"],
            color=meta["color"]
        )
        
        # Чтение траектории из CSV файла с помощью numpy для скорости
        csv_path = os.path.join(folder, meta["csv_file"])
        
        # np.loadtxt читает числовые данные, пропуская заголовок
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        
        # Запись данных в списки объекта Planet
        p.path_x = data[:, 0].tolist()
        p.path_y = data[:, 1].tolist()
        p.path_z = data[:, 2].tolist()
        p.path_vx = data[:, 3].tolist()
        p.path_vy = data[:, 4].tolist()
        p.path_vz = data[:, 5].tolist()
        
        planets.append(p)
        print(f"  - Loaded {p.name}: {len(p.path_x)} points")
        
    return planets