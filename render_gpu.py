import numpy as np
import sys
import os
import time
import subprocess
import vispy
# Если были проблемы с Qt, раскомментируйте следующую строку
# vispy.use('egl') 
from vispy import app, scene, io
from vispy.color import Color

# Импорты проекта
from utils import load_data_binary
from consts import AU

# --- НАСТРОЙКИ ---
ORBITAL_PERIODS = {
    "Mercury": 0.241, "Venus": 0.615, "Earth": 1.0, "Mars": 1.88,
    "Jupiter": 11.86, "Saturn": 29.45, "Uranus": 84.0, "Neptune": 164.8
}

PLANET_COLORS = {
    "Mercury": "#A9A9A9", "Venus": "#FFD700", "Earth": "#00BFFF", "Mars": "#FF4500",
    "Jupiter": "#FFA500", "Saturn": "#F0E68C", "Uranus": "#AFEEEE", "Neptune": "#4169E1"
}

class SolarSystemVisualizer:
    def __init__(self, folder_name, num_planets, duration_years, dt_save, width=1920, height=1080):
        self.folder_name = folder_name
        self.num_planets = num_planets
        self.duration_years = duration_years
        self.dt = float(dt_save)
        self.width = width
        self.height = height
        
        # 1. Загрузка данных
        self.load_data()
        
        # 2. Настройка OpenGL Canvas
        # bgcolor='black' для космоса
        self.canvas = scene.SceneCanvas(keys=None, size=(width, height), show=False, bgcolor='black')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        # Устанавливаем камеру с небольшим запасом (1.1x от самой дальней планеты)
        self.view.camera.set_range(x=[-self.limits, self.limits], y=[-self.limits, self.limits])
        
        # 3. Визуал
        self.setup_visuals()
        
    def load_data(self):
        path = f"assets/{self.folder_name}"
        print(f"Loading data from {path}...")
        try:
            self.planets = load_data_binary(path)
        except FileNotFoundError:
            print(f"Error: Folder {path} not found.")
            sys.exit(1)
            
        max_dist = 0
        self.planet_proxies = []
        
        print(f"Using DT_SAVE = {self.dt} seconds.")
        print("Normalizing coordinates to AU (Astronomical Units) for GPU precision...")
        
        for i, p in enumerate(self.planets):
            if i >= self.num_planets: break
            
            # --- ВАЖНОЕ ИСПРАВЛЕНИЕ: ДЕЛИМ НА AU ---
            # Переводим метры в Астрономические Единицы
            px = np.array(p.path_x) / AU
            py = np.array(p.path_y) / AU
            
            # Масштаб сцены
            curr_max = np.max(np.abs(px[:1000])) 
            if curr_max > max_dist: max_dist = curr_max
            
            # Расчет длины хвоста
            period_sec = ORBITAL_PERIODS.get(p.name, 10.0) * 365.25 * 24 * 3600
            tail_indices = int((period_sec * 1.1) / self.dt)
            if tail_indices < 50: tail_indices = 50
            
            self.planet_proxies.append({
                'name': p.name,
                'x': px, 'y': py,
                'tail_len': tail_indices,
                'color': PLANET_COLORS.get(p.name, 'white')
            })
            
        self.limits = max_dist * 1.1
        self.total_steps = len(self.planets[0].path_x)
        print(f"Scene limits: {self.limits:.2f} AU")

    def setup_visuals(self):
        # Солнце (в центре 0,0)
        self.sun = scene.visuals.Markers(parent=self.view.scene)
        self.sun.set_data(np.array([[0, 0]]), face_color='yellow', size=15, edge_width=0)
        
        self.tails = []
        self.markers = []
        
        for p in self.planet_proxies:
            # Линия
            line = scene.visuals.Line(parent=self.view.scene, color=p['color'], width=1.5, method='gl')
            self.tails.append(line)
            # Точка планеты
            marker = scene.visuals.Markers(parent=self.view.scene)
            self.markers.append(marker)

        # Текст
        self.text = scene.visuals.Text("", parent=self.canvas.scene, color='white', 
                                       anchor_x='left', anchor_y='top', font_size=14, pos=(20, 20))

    def render_video(self, filename, fps=60):
        # Логика скорости видео
        if self.num_planets <= 4:
            years_per_sec = 0.5 
        else:
            years_per_sec = 5.0 
            
        total_video_frames = int(self.duration_years / years_per_sec * fps)
        seconds_in_year = 365.25 * 24 * 3600
        
        end_sim_time = self.duration_years * seconds_in_year
        end_index = int(end_sim_time / self.dt)
        if end_index > self.total_steps: end_index = self.total_steps
        
        step_stride = int(end_index / total_video_frames)
        if step_stride < 1: step_stride = 1
        
        print(f"Rendering {total_video_frames} frames via FFmpeg...")
        
        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}', '-pix_fmt', 'rgb24',
            '-r', str(fps), '-i', '-', '-c:v', 'libx264',
            '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
            filename
        ]
        
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        start_time = time.time()
        
        for frame in range(total_video_frames):
            idx = frame * step_stride
            if idx >= self.total_steps: break
            
            for i, p in enumerate(self.planet_proxies):
                tail_start = max(0, idx - p['tail_len'])
                
                # Оптимизация: берем не все точки, если их слишком много
                tail_len_actual = idx - tail_start
                draw_step = max(1, tail_len_actual // 500)
                
                x_data = p['x'][tail_start:idx:draw_step]
                y_data = p['y'][tail_start:idx:draw_step]
                
                if len(x_data) > 0:
                    pts = np.vstack((x_data, y_data)).T
                    self.tails[i].set_data(pos=pts)
                    
                    cur_pos = np.array([[p['x'][idx], p['y'][idx]]])
                    self.markers[i].set_data(pos=cur_pos, face_color=p['color'], size=8)
            
            years = idx * self.dt / seconds_in_year
            self.text.text = f"Time: {years:.2f} yr"
            
            img = self.canvas.render(alpha=False)
            process.stdin.write(img.tobytes())
            
            if frame % 30 == 0:
                perc = frame / total_video_frames * 100
                elapsed = time.time() - start_time
                fps_render = frame / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\rRendering: {perc:.1f}% | FPS: {fps_render:.1f}")
                sys.stdout.flush()
                
        process.stdin.close()
        process.wait()
        print(f"\nDone! Video saved to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python render_gpu.py <DATA_FOLDER> <NUM_PLANETS> <DT_SAVE_SECONDS>")
        print("Example: python render_gpu.py data_bin 4 3600")
    else:
        folder = sys.argv[1]
        n_pl = int(sys.argv[2])
        dt_val = float(sys.argv[3])
        
        dur = 10.0
        if n_pl > 4: dur = 50.0 
        
        output = f"assets/gpu_orbit_{n_pl}planets.mp4"
        viz = SolarSystemVisualizer(folder, n_pl, dur, dt_val)
        viz.render_video(output)