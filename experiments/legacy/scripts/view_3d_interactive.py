import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys
import os
import cv2
import matplotlib as mpl
from matplotlib.widgets import Slider

# Disable default matplotlib keyboard shortcuts for left/right arrows
for key in ['left', 'right']:
    if key in mpl.rcParams['keymap.back']: mpl.rcParams['keymap.back'].remove(key)
    if key in mpl.rcParams['keymap.forward']: mpl.rcParams['keymap.forward'].remove(key)

# Ensure local modules are found
sys.path.append(os.getcwd())
try:
    from src.analyzer.squat_analyzer_knee import SquatAnalyzer, SquatFrame
except ImportError:
    # Fallback if the user hasn't run the analyzer yet or structure is different
    from src.analyzer.squat_analyzer_knee import SquatAnalyzer

from src.config import FILTER_CONFIG, VIEWER_CONFIG


class SquatViewer3D:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.history_data = analyzer.history_data
        self.side = getattr(analyzer, 'side', 'Unknown').capitalize()
        
        if not self.history_data:
            print("No data to visualize!")
            sys.exit(1)
            
        self._apply_smoothing()
        self._sync_parameters(analyzer)
        
        # Rigid Body Lengths
        t_max = max(f.thigh_len_2d for f in self.history_data)
        s_max = max(f.shank_len_2d for f in self.history_data)
        self.final_thigh_len = max(t_max, s_max * 1.2)
        self.final_shank_len = self.final_thigh_len / 1.2
        
        print(f"Viewer Initialized. K={self.base_k:.3f}, ThighLen={self.final_thigh_len:.1f}")

    def _apply_smoothing(self):
        """Apply Savitzky-Golay filtering to the history data."""
        if not FILTER_CONFIG['ENABLED']:
            print("Smoothing Disabled.")
            return
            
        try:
            from scipy.signal import savgol_filter
            print("Applying Window Filtering (Savitzky-Golay)...")
            
            n_frames = len(self.history_data)
            window = FILTER_CONFIG['WINDOW_LENGTH']
            if n_frames < window: window = n_frames if n_frames % 2 == 1 else n_frames - 1
            poly = FILTER_CONFIG['POLY_ORDER']
            
            if window > 3:
                for joint in ['hip', 'knee', 'ankle']:
                    for axis in ['x', 'y', 'z']:
                        raw = [getattr(f, joint)[axis] for f in self.history_data]
                        smooth = savgol_filter(raw, window, poly)
                        for i, val in enumerate(smooth):
                            getattr(self.history_data[i], joint)[axis] = val
            print("Smoothing Complete.")
        except ImportError:
            print("SciPy not found. Skipping smoothing.")

    def _sync_parameters(self, analyzer):
        """Sync optimization parameters from the analyzer."""
        print("Running Analyzer Optimization to sync parameters...")
        _, _, _, params = analyzer.finalize_analysis()
        self.base_k = params.get('a_k', 1.0) # Use a_k as base_k from regressor
        self.azim = VIEWER_CONFIG['DEFAULT_AZIM']
        self.elev = VIEWER_CONFIG['DEFAULT_ELEV']
        self.is_paused = False
        self.current_frame = 0

    def _get_rigid_joint(self, frame, joint_name, target_len, knee_pos):
        """Calculate a joint position based on rigid body length constraints."""
        joint = getattr(frame, joint_name)
        dx, dy = joint['x'] - frame.knee['x'], joint['y'] - frame.knee['y']
        l2d_sq = dx**2 + dy**2
        z_diff_sq = target_len**2 - l2d_sq
        
        if z_diff_sq > 0:
            dz = np.sqrt(z_diff_sq)
            direction = 1 if joint['z'] > frame.knee['z'] else -1
            return knee_pos + np.array([dx, dy, direction * dz])
        else:
            # Vector normalization if 2D length exceeds target
            v_2d = np.array([dx, dy, 0])
            return knee_pos + (v_2d * (target_len / np.linalg.norm(v_2d)))

    def update_lines(self, num, lines, points, image_plot):
        frame = self.history_data[num]
        
        # 1. Update Video
        if image_plot and frame.drawn_image is not None:
            rgb = frame.drawn_image[..., ::-1]
            if rgb.shape[1] > 800:
                scale = 800 / rgb.shape[1]
                rgb = cv2.resize(rgb, (800, int(rgb.shape[0] * scale)))
            image_plot.set_data(rgb)

        # 2. Update 3D Skeleton
        kz = frame.knee['z'] * self.base_k
        p_knee = np.array([frame.knee['x'], frame.knee['y'], kz])
        p_ankle = self._get_rigid_joint(frame, 'ankle', self.final_shank_len, p_knee)
        
        offset = self.s_dr.val if hasattr(self, 's_dr') else 0
        p_hip = self._get_rigid_joint(frame, 'hip', self.final_thigh_len, p_knee) + np.array([0, 0, offset])
        
        # Transform for Plotting
        def to_plot(pt): return [pt[0] - p_ankle[0], pt[2] - p_ankle[2], -(pt[1] - p_ankle[1])]
        v_hip, v_knee, v_ankle = to_plot(p_hip), to_plot(p_knee), to_plot(p_ankle)
        
        lines[0].set_data([v_hip[0], v_knee[0]], [v_hip[1], v_knee[1]])
        lines[0].set_3d_properties([v_hip[2], v_knee[2]])
        lines[1].set_data([v_knee[0], v_ankle[0]], [v_knee[1], v_ankle[1]])
        lines[1].set_3d_properties([v_knee[2], v_ankle[2]])
        
        points.set_data([v_hip[0], v_knee[0], v_ankle[0]], [v_hip[1], v_knee[1], v_ankle[1]])
        points.set_3d_properties([v_hip[2], v_knee[2], v_ankle[2]])
        
        # 3. Update Text & View
        if hasattr(self, 'ax_3d'): self.ax_3d.view_init(elev=self.elev, azim=self.azim)
        if hasattr(self, 'time_text'):
            angle = self.analyzer.calculate_knee_angle(
                {'x':p_hip[0],'y':p_hip[1],'z':p_hip[2]}, 
                {'x':p_knee[0],'y':p_knee[1],'z':p_knee[2]}, 
                {'x':p_ankle[0],'y':p_ankle[1],'z':p_ankle[2]}
            )
            self.time_text.set_text(f"Leg: {self.side}\nFrame: {num}\nAngle: {angle:.1f}\xb0")

        return lines + [points, image_plot, self.time_text]

    def _setup_ui(self, fig):
        """Initialize all UI elements and widgets."""
        # Video Panel
        ax_video = fig.add_subplot(1, 2, 1)
        ax_video.set_title("Original Video")
        ax_video.axis('off')
        
        img = self.history_data[0].drawn_image or self.history_data[0].raw_image
        if img is not None:
            img_rgb = img[..., ::-1]
            if img_rgb.shape[1] > 800:
                scale = 800 / img_rgb.shape[1]
                img_rgb = cv2.resize(img_rgb, (800, int(img_rgb.shape[0] * scale)))
            self.image_plot = ax_video.imshow(img_rgb)
        
        # 3D Panel
        self.ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
        self.ax_3d.set_title("3D Motion (Click: L/R)")
        self.ax_3d.view_init(elev=self.elev, azim=self.azim)
        
        limit = max(VIEWER_CONFIG['AXIS_LIMIT'], (self.final_thigh_len + self.final_shank_len) * 1.4)
        self.ax_3d.set_xlim3d([-limit/2, limit/2])
        self.ax_3d.set_ylim3d([-limit/2, limit/2])
        self.ax_3d.set_zlim3d([0, limit])
        
        # Skeleton Elements
        self.lines = [self.ax_3d.plot([], [], [], 'b-', lw=4)[0] for _ in range(2)]
        self.points, = self.ax_3d.plot([], [], [], 'ro', markersize=8)
        self.time_text = self.ax_3d.text2D(0.05, 0.95, "", transform=self.ax_3d.transAxes, bbox=dict(facecolor='white', alpha=0.5))
        
        # Slider
        ax_slider = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
        self.s_dr = Slider(ax_slider, 'Hip Z Adj', -100.0, 100.0, valinit=0.0)

    def start(self):
        fig = plt.figure(figsize=(14, 8))
        self._setup_ui(fig)
        
        def on_click(event):
            if event.inaxes != self.ax_3d: return
            if event.button == 1: self.azim -= 5
            elif event.button == 3: self.azim += 5
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)

        def on_key(event):
            if event.key == ' ':
                if self.is_paused: ani.resume()
                else: ani.pause()
                self.is_paused = not self.is_paused
            elif event.key in ['left', 'right'] and self.is_paused:
                step = 1 if event.key == 'right' else -1
                self.current_frame = np.clip(self.current_frame + step, 0, len(self.history_data)-1)
                self.update_lines(self.current_frame, self.lines, self.points, self.image_plot)
                fig.canvas.draw_idle()

        def frame_gen():
            while True:
                if not self.is_paused:
                    self.current_frame = (self.current_frame + 1) % len(self.history_data)
                yield self.current_frame

        ani = animation.FuncAnimation(
            fig, self.update_lines, frames=frame_gen,
            fargs=(self.lines, self.points, self.image_plot),
            interval=VIEWER_CONFIG['ANIMATION_INTERVAL'], blit=False
        )

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.subplots_adjust(bottom=0.15)
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_3d_interactive.py <video_path>")
        sys.exit(1)
        
    print("Analyzing Video...")
    analyzer = SquatAnalyzer()
    analyzer.process_video(sys.argv[1])
    print("Launching 3D Viewer...")
    SquatViewer3D(analyzer).start()
