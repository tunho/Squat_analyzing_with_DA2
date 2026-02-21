import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import sys
import argparse
import os

# ==========================================
# CONFIGURATION
# ==========================================
VIEWER_CONFIG = {
    'WINDOW_SIZE': 5,            # Savitzky-Golay Window (Must be odd)
    'POLY_ORDER': 2,             # Savitzky-Golay Polynomial Order
    'ANIMATION_INTERVAL': 40,    # ms (25fps)
    'DEFAULT_ELEV': 10,
    'DEFAULT_AZIM': -90,          # -90 = Side View, 0 = Front View
    'AXIS_LIMIT': 500             # Plot boundary size
}

class LightweightSquatViewer:
    def __init__(self, pickle_path):
        print(f"Loading data from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        self.history_data = data['history_data']
        self.base_k = data['base_k']
        self.final_thigh_len = data['final_thigh_len']
        self.final_shank_len = data['final_shank_len']
        self.stable_radii = data['stable_radii']
        self.side = "UNKNOWN" # simplified
        print("Data successfully loaded! Launching Matplotlib Viewer...")

    def update_lines(self, num, lines, points, image_plot):
        frame = self.history_data[num]
        
        # 1. Update Video Frame
        if image_plot is not None and frame.get('drawn_image') is not None:
            rgb_frame = frame['drawn_image'][..., ::-1]
            image_plot.set_data(rgb_frame)
            
        # [A] Knee (Global Anchor)
        knee_z_metric = frame['knee']['z'] * self.base_k
        knee_z_center = knee_z_metric + self.stable_radii['knee']
        p_final_knee = np.array([frame['knee']['x'], frame['knee']['y'], knee_z_center])
        
        # [B] Ankle (Dynamic Anchor Connection from Knee)
        shank_len = np.linalg.norm([frame['ankle']['x'] - frame['knee']['x'], frame['ankle']['y'] - frame['knee']['y']])
        
        if self.final_shank_len > shank_len:
             z_needed_shank = np.sqrt(self.final_shank_len**2 - shank_len**2)
        else:
             z_needed_shank = 0.0
             
        direction_shank = 1 if frame['ankle']['z'] > frame['knee']['z'] else -1
        ankle_z_center = knee_z_center + (direction_shank * z_needed_shank)
        p_final_ankle = np.array([frame['ankle']['x'], frame['ankle']['y'], ankle_z_center])
        
        # [C] Hip (Dynamic Anchor Connection from Knee)
        thigh_len = np.linalg.norm([frame['hip']['x'] - frame['knee']['x'], frame['hip']['y'] - frame['knee']['y']])
        
        if self.final_thigh_len > thigh_len:
             z_needed_thigh = np.sqrt(self.final_thigh_len**2 - thigh_len**2)
        else:
             z_needed_thigh = 0.0
             
        direction_hip = 1 if frame['hip']['z'] > frame['knee']['z'] else -1
        hip_z_center = knee_z_center + (direction_hip * z_needed_thigh)
        
        # Apply Slider Offset on Hip Z for interactive visual tuning
        manual_offset = self.s_dr.val if hasattr(self, 's_dr') else 0
        p_final_hip = np.array([frame['hip']['x'], frame['hip']['y'], hip_z_center + manual_offset])
        
        # Needed for Difference visual logic checks
        knee_z_base = knee_z_center
        hip_z_base = frame['hip']['z'] * self.base_k
        
        # [Visualization Transformation]
        origin = p_final_ankle.copy()
        def to_plot(pt):
            # MP(x, y, z) -> Plot(x, z, -y)
            return [pt[0] - origin[0], pt[2] - origin[2], -(pt[1]-origin[1])]
            
        vis_hip = to_plot(p_final_hip)
        vis_knee = to_plot(p_final_knee)
        vis_ankle = to_plot(p_final_ankle)
        
        lines[0].set_data([vis_hip[0], vis_knee[0]], [vis_hip[1], vis_knee[1]])
        lines[0].set_3d_properties([vis_hip[2], vis_knee[2]])
        
        lines[1].set_data([vis_knee[0], vis_ankle[0]], [vis_knee[1], vis_ankle[1]])
        lines[1].set_3d_properties([vis_knee[2], vis_ankle[2]])
        
        xs = [vis_hip[0], vis_knee[0], vis_ankle[0]]
        ys = [vis_hip[1], vis_knee[1], vis_ankle[1]]
        zs = [vis_hip[2], vis_knee[2], vis_ankle[2]]
        
        points.set_data(xs, ys)
        points.set_3d_properties(zs)
        
        raw_knee_z_val = knee_z_base
        raw_hip_z_val = hip_z_base
        rigid_knee_z_val = p_final_knee[2]
        rigid_hip_z_val = p_final_hip[2] - manual_offset
        
        diff_knee = rigid_knee_z_val - raw_knee_z_val
        diff_hip = rigid_hip_z_val - raw_hip_z_val
        
        len_thigh_3d = np.linalg.norm(p_final_hip - p_final_knee)
        len_shank_3d = np.linalg.norm(p_final_knee - p_final_ankle)
        
        if hasattr(self, 'time_text'):
            self.time_text.set_text(
                f"Frame: {num}\n"
                f"Thigh: {len_thigh_3d:.1f} (Target: {self.final_thigh_len:.1f})\n"
                f"Shank: {len_shank_3d:.1f} (Target: {self.final_shank_len:.1f})\n"
                f"----------------\n"
                f"Hip Z Shift: {diff_hip:.1f} (Rigid - Raw)\n"
                f"Rigid: {rigid_hip_z_val:.1f} | Raw: {raw_hip_z_val:.1f}"
            )
        
        return lines + [points, image_plot, self.time_text if hasattr(self, 'time_text') else points]

    def start(self):
        fig = plt.figure(figsize=(14, 8))
        
        # 1. Video Panel
        ax_video = fig.add_subplot(1, 2, 1)
        ax_video.set_title("Original Video (Synced)")
        ax_video.axis('off')
        first_frame = self.history_data[0]['drawn_image'][..., ::-1]
        image_plot = ax_video.imshow(first_frame)
        
        # 2. 3D Panel
        self.ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
        self.ax_3d.set_title(f"3D Motion - {self.side} Knee (Click: L/R)")
        self.ax_3d.set_xlabel('X (Lateral)')
        self.ax_3d.set_ylabel('Y (Depth)')
        self.ax_3d.set_zlabel('Z (Height)')
        
        self.azim = VIEWER_CONFIG['DEFAULT_AZIM']
        self.elev = VIEWER_CONFIG['DEFAULT_ELEV']
        self.ax_3d.view_init(elev=self.elev, azim=self.azim)
        
        limit = VIEWER_CONFIG['AXIS_LIMIT']
        self.ax_3d.set_xlim3d([-limit/2, limit/2])
        self.ax_3d.set_ylim3d([-limit/2, limit/2])
        self.ax_3d.set_zlim3d([0, limit]) 
        
        floor_z = 0
        self.ax_3d.text(0, -200, floor_z, "FRONT (Camera)", color='green', fontweight='bold', ha='center')
        self.ax_3d.quiver(0, 0, floor_z, 0, -50, 0, color='green', arrow_length_ratio=0.3)

        lines = [self.ax_3d.plot([], [], [], 'b-', lw=4)[0] for _ in range(2)] 
        points, = self.ax_3d.plot([], [], [], 'ro', markersize=8)
        self.time_text = self.ax_3d.text2D(0.05, 0.95, "", transform=self.ax_3d.transAxes, color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        from matplotlib.widgets import Slider
        ax_dr = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
        self.s_dr = Slider(ax_dr, 'Hip Z Adjustment', -100.0, 100.0, valinit=0.0)
        
        def on_click(event):
            if event.inaxes != self.ax_3d: return
            if event.button == 1: self.azim -= 1 
            elif event.button == 3: self.azim += 1 
            elif event.button == 2: self.azim = -90 
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)
            self.ax_3d.set_title(f"Angle: {self.azim}° (Elev: {self.elev}°)")

        def on_scroll(event):
            if event.inaxes != self.ax_3d: return
            step = 10
            if event.button == 'up': self.azim += step
            elif event.button == 'down': self.azim -= step
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)
            self.ax_3d.set_title(f"Angle: {self.azim}° (Elev: {self.elev}°)")

        def on_move(event):
            if event.inaxes == self.ax_3d:
                self.ax_3d.view_init(elev=self.elev, azim=self.azim)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        fig.canvas.mpl_connect('motion_notify_event', on_move) 

        ani = animation.FuncAnimation(
            fig, self.update_lines, frames=len(self.history_data),
            fargs=(lines, points, image_plot), interval=VIEWER_CONFIG['ANIMATION_INTERVAL'], blit=False
        )
        
        # Pause/Resume State and Event
        self.is_paused = False
        def on_key(event):
            if event.key == ' ': 
                if self.is_paused:
                    ani.resume()
                else:
                    ani.pause()
                self.is_paused = not self.is_paused

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.subplots_adjust(bottom=0.15)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_path', type=str, nargs='?', default='squat_data.pkl')
    args = parser.parse_args()
    
    if not os.path.exists(args.pickle_path):
        print(f"Error: {args.pickle_path} not found.")
        sys.exit(1)
        
    viewer = LightweightSquatViewer(args.pickle_path)
    viewer.start()
