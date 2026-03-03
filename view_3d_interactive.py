import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys
import os
import cv2
import matplotlib as mpl

# Disable default matplotlib keyboard shortcuts for left/right arrows
if 'left' in mpl.rcParams['keymap.back']:
    mpl.rcParams['keymap.back'].remove('left')
if 'right' in mpl.rcParams['keymap.forward']:
    mpl.rcParams['keymap.forward'].remove('right')

# Ensure local modules are found
sys.path.append(os.getcwd())
from squat_analyzer_knee import SquatAnalyzer

from config import FILTER_CONFIG, VIEWER_CONFIG # [REF] Central Config

class SquatViewer3D:
    def __init__(self, analyzer):
        self.history_data = analyzer.history_data
        self.side = getattr(analyzer, 'side', 'Unknown').capitalize() # 'right' -> 'Right'
        
        # 3-Pass Logic to Get Best Parameters
        if not self.history_data:
            print("No data to visualize!")
            sys.exit(1)
            
        # [NEW] Apply Window Filtering (Savitzky-Golay Smoothing) to Raw Coordinates
        if FILTER_CONFIG['ENABLED']:
            try:
                from scipy.signal import savgol_filter
                print("Applying Window Filtering (Savitzky-Golay)...")
                
                joints = ['hip', 'knee', 'ankle']
                coords = ['x', 'y', 'z'] # 2D pixel x,y and MediaPipe z
                
                # Extract data arrays
                n_frames = len(self.history_data)
                window_length = FILTER_CONFIG['WINDOW_LENGTH'] 
                if n_frames < window_length: window_length = n_frames if n_frames % 2 == 1 else n_frames - 1
                polyorder = FILTER_CONFIG['POLY_ORDER']
                
                if window_length > 3:
                    for joint in joints:
                        for axis in coords:
                            raw_vals = [f[joint][axis] for f in self.history_data]
                            smooth_vals = savgol_filter(raw_vals, window_length, polyorder)
                            
                            # Update history_data with smoothed values
                            for i, val in enumerate(smooth_vals):
                                self.history_data[i][joint][axis] = val
                                
                print("Smoothing Complete.")
                
            except ImportError:
                print("SciPy not found. Skipping smoothing.")
        else:
            print("Smoothing (Filter) Disabled via Config.")
                            

            
        # [ALIGNMENT] Retrieve parameters directly from Analyzer logic
        # 3. Get Optimized Parameters from Analyzer
        print("Running Analyzer Optimization to sync parameters...")
        # finalize_analysis returns (counter, angles, debug_dict)
        # debug_dict contains {'base_k': ..., 'dr': ...}
        _, _, params = analyzer.finalize_analysis()
        
        self.base_k = params.get('base_k', 1.0)
        
        # 4. Define Rigid Body Lengths (Same logic as Analyzer)
        obs_max_thigh = max(f['thigh_len_2d'] for f in self.history_data)
        obs_max_shank = max(f['shank_len_2d'] for f in self.history_data)
        
        self.final_thigh_len = max(obs_max_thigh, obs_max_shank * 1.2)
        self.final_shank_len = self.final_thigh_len / 1.2
        
        print(f"Viewer Initialized. K={self.base_k:.3f}, ThighLen={self.final_thigh_len:.1f}")

    def update_lines(self, num, lines, points, image_plot):
        # Current Frame Data
        frame = self.history_data[num]
        
        # 1. Update Video Frame
        if image_plot is not None and frame.get('drawn_image') is not None:
            # Convert BGR to RGB for Matplotlib, using the image drawn by MediaPipe
            rgb_frame = frame['drawn_image'][..., ::-1]
            
            # [NEW] Resize to fix Matplotlib Video Stuttering (Lag)
            h, w = rgb_frame.shape[:2]
            if w > 800:
                scale = 800 / w
                rgb_frame = cv2.resize(rgb_frame, (800, int(h * scale)))
                
            image_plot.set_data(rgb_frame)
            


        # [A] Knee (Global Anchor - Surface Point Base)
        knee_z_metric = frame['knee']['z'] * self.base_k
        knee_z_center = knee_z_metric
        p_final_knee = np.array([frame['knee']['x'], frame['knee']['y'], knee_z_center])
        
        # [B] Ankle (Dynamic Anchor Connection from Knee)
        # 스무딩 등 좌표 변화가 있을 수 있으므로 현재 X, Y로 2D 길이를 다시 계산합니다.
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
        
        # [Difference Check] Compare Raw Depth vs Rigid Body Depth
        # Raw Depth (DAv2 * K + Radius)
        raw_knee_z_val = knee_z_base
        raw_hip_z_val = hip_z_base
        
        # Rigid Depth (Final Z after constraint)
        rigid_knee_z_val = p_final_knee[2]
        rigid_hip_z_val = p_final_hip[2] - manual_offset # Exclude manual offset for fair comparison
        
        diff_knee = rigid_knee_z_val - raw_knee_z_val
        diff_hip = rigid_hip_z_val - raw_hip_z_val
        
        # [NEW] Calculate Knee Angle (3D)
        len_thigh_3d = np.linalg.norm(p_final_hip - p_final_knee)
        len_shank_3d = np.linalg.norm(p_final_knee - p_final_ankle)

        vec_thigh = p_final_hip - p_final_knee
        vec_shank = p_final_ankle - p_final_knee
        
        norm_thigh = np.linalg.norm(vec_thigh)
        norm_shank = np.linalg.norm(vec_shank)
        
        if norm_thigh > 0 and norm_shank > 0:
            cos_theta = np.dot(vec_thigh, vec_shank) / (norm_thigh * norm_shank)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_rad = np.arccos(cos_theta)
            angle_deg = np.degrees(angle_rad)
        else:
            angle_deg = 0.0
            
        # Lock angle to user-controlled values (suppresses default mplot3d arrow key rotation)
        if hasattr(self, 'ax_3d') and hasattr(self, 'elev') and hasattr(self, 'azim'):
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)
        
        if hasattr(self, 'time_text'):
            self.time_text.set_text(
                f"Analysis: {self.side} Leg\n"
                f"Frame: {num}\n"
                f"Knee Angle: {angle_deg:.1f}\xb0\n"
                f"Thigh: {len_thigh_3d:.1f} (Target: {self.final_thigh_len:.1f})\n"
                f"Shank: {len_shank_3d:.1f} (Target: {self.final_shank_len:.1f})\n"
                f"----------------\n"
                f"Hip Z Shift: {diff_hip:.1f} (Rigid - Raw)\n"
                f"Rigid: {rigid_hip_z_val:.1f} | Raw: {raw_hip_z_val:.1f}"
            )
        
        return lines + [points, image_plot, self.time_text if hasattr(self, 'time_text') else points]

    def start(self):
        # Setup Figure with 2 Subplots
        fig = plt.figure(figsize=(14, 8)) # Increased height for slider
        
        # 1. Video Panel (Left)
        ax_video = fig.add_subplot(1, 2, 1)
        ax_video.set_title("Original Video (Synced)")
        ax_video.axis('off')
        
        # Initial Frame
        first_frame = self.history_data[0].get('drawn_image', self.history_data[0]['raw_image'])[..., ::-1]
        
        # [NEW] Resize Initial Frame as well
        h, w = first_frame.shape[:2]
        if w > 800:
            scale = 800 / w
            first_frame = cv2.resize(first_frame, (800, int(h * scale)))
            
        image_plot = ax_video.imshow(first_frame)
        
        # 2. 3D Panel (Right)
        self.ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
        self.ax_3d.set_title(f"3D Motion - {self.side} Knee (Click: L/R)")
        self.ax_3d.set_xlabel('X (Lateral)')
        self.ax_3d.set_ylabel('Y (Depth)')
        self.ax_3d.set_zlabel('Z (Height)')
        
        # Custom View State
        # Custom View State
        self.azim = VIEWER_CONFIG['DEFAULT_AZIM']
        self.elev = VIEWER_CONFIG['DEFAULT_ELEV']
        self.ax_3d.view_init(elev=self.elev, azim=self.azim)
        
        # Set Consistent Axis Limits (Dynamically scaled to fit long legs)
        total_len_estimate = self.final_thigh_len + self.final_shank_len
        limit = max(VIEWER_CONFIG['AXIS_LIMIT'], total_len_estimate * 1.4) 
        self.ax_3d.set_xlim3d([-limit/2, limit/2])
        self.ax_3d.set_ylim3d([-limit/2, limit/2])
        self.ax_3d.set_zlim3d([0, limit]) 
        
        # Orientation Guides
        floor_z = 0
        self.ax_3d.text(0, -200, floor_z, "FRONT (Camera)", color='green', fontweight='bold', ha='center')
        self.ax_3d.text(0, 200, floor_z, "BACK", color='gray', ha='center')
        self.ax_3d.text(-200, 0, floor_z, "LEFT", color='gray', ha='center')
        self.ax_3d.text(200, 0, floor_z, "RIGHT", color='gray', ha='center')
        self.ax_3d.quiver(0, 0, floor_z, 0, -50, 0, color='green', arrow_length_ratio=0.3)

        # Initialize Lines
        lines = [self.ax_3d.plot([], [], [], 'b-', lw=4)[0] for _ in range(2)] 
        points, = self.ax_3d.plot([], [], [], 'ro', markersize=8)
        
        # [NEW] Text for Length Debug
        self.time_text = self.ax_3d.text2D(0.05, 0.95, "", transform=self.ax_3d.transAxes, color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        # [NEW] Slider for Hip Depth Correction
        from matplotlib.widgets import Slider
        
        # Place slider at bottom
        ax_dr = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
        self.s_dr = Slider(ax_dr, 'Hip Z Adjustment', -100.0, 100.0, valinit=0.0)
        
        # Event Handler for Click Rotation
        def on_click(event):
            if event.inaxes != self.ax_3d: return
            if event.button == 1: self.azim -= 1 # Left Click
            elif event.button == 3: self.azim += 1 # Right Click
            elif event.button == 2: self.azim = -90 # Middle Click Reset
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)
            self.ax_3d.set_title(f"Angle: {self.azim}° (Elev: {self.elev}°)")
            # Redraw happens in animation loop

        # Scroll Event (Keep this if you want scroll to rotate faster)
        def on_scroll(event):
            if event.inaxes != self.ax_3d: return
            step = 10
            if event.button == 'up': self.azim += step
            elif event.button == 'down': self.azim -= step
            self.ax_3d.view_init(elev=self.elev, azim=self.azim)
            self.ax_3d.set_title(f"Angle: {self.azim}° (Elev: {self.elev}°)")

        # [NEW] Disable Default Mouse Drag Rotation
        # Matplotlib's 3D plot rotates on drag by default. We want to BLOCK this.
        def on_move(event):
            if event.inaxes == self.ax_3d:
                # Force reset view to our managed state if user tries to drag
                self.ax_3d.view_init(elev=self.elev, azim=self.azim)

        # [MODIFIED] Slider Update: Just force redraw if paused
        def update_slider(val):
            # No special logic needed, update_lines() reads self.s_dr.val every frame
            pass
            
        self.s_dr.on_changed(update_slider)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        fig.canvas.mpl_connect('motion_notify_event', on_move) # Block drag

        # [NEW] Pause/Resume State and Event
        self.is_paused = False
        self.current_frame = 0
        
        def on_key(event):
            if event.key == ' ': # 스페이스바(공백) 클릭 시
                if self.is_paused:
                    ani.resume()
                else:
                    ani.pause()
                self.is_paused = not self.is_paused
            elif event.key == 'right':
                if self.is_paused:
                    self.current_frame = min(len(self.history_data) - 1, self.current_frame + 1)
                    self.update_lines(self.current_frame, lines, points, image_plot)
                    fig.canvas.draw_idle()
            elif event.key == 'left':
                if self.is_paused:
                    self.current_frame = max(0, self.current_frame - 1)
                    self.update_lines(self.current_frame, lines, points, image_plot)
                    fig.canvas.draw_idle()

        # We need a small helper to keep current_frame in sync with ani
        def ani_frame_gen():
            while True:
                if not self.is_paused:
                    self.current_frame = (self.current_frame + 1) % len(self.history_data)
                yield self.current_frame

        # Create Animation
        ani = animation.FuncAnimation(
            fig, self.update_lines, frames=ani_frame_gen,
            fargs=(lines, points, image_plot), interval=VIEWER_CONFIG['ANIMATION_INTERVAL'], blit=False,
            save_count=len(self.history_data)
        )

        fig.canvas.mpl_connect('key_press_event', on_key) # Bind spacebar and arrows
        
        plt.subplots_adjust(bottom=0.15) # Make room for slider
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_3d_interactive.py <video_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    print("Analyzing Video...")
    analyzer = SquatAnalyzer()
    analyzer.process_video(video_path, show_window=False) 
    print("Launching Interactive 3D Viewer...")
    viewer = SquatViewer3D(analyzer)
    viewer.start()
