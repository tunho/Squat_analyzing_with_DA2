import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys
import os

# Ensure local modules are found
sys.path.append(os.getcwd())
from squat_analyzer_knee import SquatAnalyzer

class SquatViewer3D:
    def __init__(self, analyzer):
        self.history_data = analyzer.history_data
        
        # 3-Pass Logic to Get Best Parameters
        if not self.history_data:
            print("No data to visualize!")
            sys.exit(1)
            
        print("Calculating 3D Parameters...")
        
        # 1. Obs Max
        obs_max_thigh = max(f['thigh_len_2d'] for f in self.history_data)
        obs_max_shank = max(f['shank_len_2d'] for f in self.history_data)
        
        # 2. Rigid Body
        final_thigh_len = max(obs_max_thigh, obs_max_shank * 1.2)
        final_shank_len = final_thigh_len / 1.2
        
        # 3. Radii
        self.stable_radii = analyzer.corrector.get_radii(final_thigh_len, final_shank_len)
        dr_stable = self.stable_radii['knee'] - self.stable_radii['hip']
        
        # 4. Global K
        k_candidates = []
        for frame in self.history_data:
             thigh_len = frame['thigh_len_2d']
             target = final_thigh_len
             if target > thigh_len:
                 z_needed = np.sqrt(target**2 - thigh_len**2)
                 dZ_raw = frame['knee']['z'] - frame['hip']['z']
                 if abs(dZ_raw) > 1e-6:
                      k1 = (z_needed - dr_stable) / dZ_raw
                      if k1 > 0: k_candidates.append(k1)
        
        if k_candidates:
            self.base_k = float(np.median(k_candidates))
        else:
            self.base_k = 1.0
            
        print(f"Viewer Initialized. K={self.base_k:.3f}, Radii={self.stable_radii}")

    def update_lines(self, num, lines, points, image_plot):
        # Current Frame Data
        frame = self.history_data[num]
        
        # 1. Update Video Frame
        if image_plot is not None and frame.get('raw_image') is not None:
            # Convert BGR to RGB for Matplotlib
            rgb_frame = frame['raw_image'][..., ::-1]
            image_plot.set_data(rgb_frame)

        # 2. Update 3D Skeleton
        # Recover 3D Coords
        hip_z_raw = frame['hip']['z'] * self.base_k + self.stable_radii['hip']
        knee_z_raw = frame['knee']['z'] * self.base_k + self.stable_radii['knee']
        ankle_z_raw = frame['ankle']['z'] * self.base_k + self.stable_radii['ankle']
        
        # [UPDATED] Pivot: ANKLE (Center of world)
        origin_x = frame['ankle']['x']
        origin_y = frame['ankle']['y'] # Image Y (Down)
        origin_z = ankle_z_raw         # Depth Z
        
        def transform(pt, z_val):
            return [
                pt['x'] - origin_x,         # X (Relative to ankle)
                z_val - origin_z,           # Y (Depth relative to ankle)
                -(pt['y'] - origin_y)       # Z (Height from ankle, Inverted Y because image Y is down)
            ]
            
        # Get Points
        p_hip = transform(frame['hip'], hip_z_raw)
        p_knee = transform(frame['knee'], knee_z_raw)
        p_ankle = transform(frame['ankle'], ankle_z_raw) # Should be (0,0,0)
        
        # Bones List: [Hip-Knee, Knee-Ankle]
        # Line 0: Hip -> Knee
        lines[0].set_data([p_hip[0], p_knee[0]], [p_hip[1], p_knee[1]])
        lines[0].set_3d_properties([p_hip[2], p_knee[2]])
        
        # Line 1: Knee -> Ankle
        lines[1].set_data([p_knee[0], p_ankle[0]], [p_knee[1], p_ankle[1]])
        lines[1].set_3d_properties([p_knee[2], p_ankle[2]])
        
        # Update Points (Joints)
        xs = [p_hip[0], p_knee[0], p_ankle[0]]
        ys = [p_hip[1], p_knee[1], p_ankle[1]]
        zs = [p_hip[2], p_knee[2], p_ankle[2]]
        
        points.set_data(xs, ys)
        points.set_3d_properties(zs)
        
        return lines + [points, image_plot]

    def start(self):
        # Setup Figure with 2 Subplots
        fig = plt.figure(figsize=(14, 7))
        
        # 1. Video Panel (Left)
        ax_video = fig.add_subplot(1, 2, 1)
        ax_video.set_title("Original Video (Synced)")
        ax_video.axis('off')
        
        # Initial Frame
        first_frame = self.history_data[0]['raw_image'][..., ::-1]
        image_plot = ax_video.imshow(first_frame)
        
        # 2. 3D Panel (Right)
        ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
        ax_3d.set_title("3D Motion (Click to Rotate 1°)")
        ax_3d.set_xlabel('X (Lateral)')
        ax_3d.set_ylabel('Y (Depth)')
        ax_3d.set_zlabel('Z (Height)')
        
        # [NEW] Custom View State
        self.azim = -90  # Start at Front (-90 degrees is standard front in mpl)
        self.elev = 10   # Fixed elevation (slightly above eye level)
        ax_3d.view_init(elev=self.elev, azim=self.azim)
        
        # Disable default mouse rotation to prevent accidental mess uo
        ax_3d.disable_mouse_rotation()
        
        # Set Consistent Axis Limits
        limit = 400
        ax_3d.set_xlim3d([-limit/2, limit/2])
        ax_3d.set_ylim3d([-limit/2, limit/2])
        ax_3d.set_zlim3d([0, limit]) 
        
        # Add Orientation Guides (Floor Labels)
        # Z = 0 (Floor)
        floor_z = 0
        
        # Text alignment: 'center'
        ax_3d.text(0, -200, floor_z, "FRONT (Camera)", color='green', fontweight='bold', ha='center')
        ax_3d.text(0, 200, floor_z, "BACK", color='gray', ha='center')
        ax_3d.text(-200, 0, floor_z, "LEFT", color='gray', ha='center')
        ax_3d.text(200, 0, floor_z, "RIGHT", color='gray', ha='center')
        
        # Draw a small arrow pointing Front at origin
        ax_3d.quiver(0, 0, floor_z, 0, -50, 0, color='green', arrow_length_ratio=0.3)

        # Initialize Lines
        lines = [ax_3d.plot([], [], [], 'b-', lw=4)[0] for _ in range(2)] # Thigh, Shank
        points, = ax_3d.plot([], [], [], 'ro', markersize=8)
        
        # [NEW] Event Handler for Click Rotation
        def on_click(event):
            if event.inaxes != ax_3d: return
            
            # Left Click (button 1) -> Rotate Left (-1 deg)
            if event.button == 1:
                self.azim -= 1
            # Right Click (button 3) -> Rotate Right (+1 deg)
            elif event.button == 3:
                self.azim += 1
            # Middle Click (button 2) -> Reset to Front
            elif event.button == 2:
                self.azim = -90
                
            # Apply View
            ax_3d.view_init(elev=self.elev, azim=self.azim)
            # Update Title
            ax_3d.set_title(f"Angle: {self.azim}° (Elev: {self.elev}°)")
            fig.canvas.draw_idle()

        # [NEW] Scroll Event for Faster Rotation
        def on_scroll(event):
            if event.inaxes != ax_3d: return
            
            step = 10
            if event.button == 'up':
                self.azim += step
            elif event.button == 'down':
                self.azim -= step
                
            ax_3d.view_init(elev=self.elev, azim=self.azim)
            ax_3d.set_title(f"Angle: {self.azim}° (Elev: {self.elev}°)")
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('scroll_event', on_scroll)

        # Create Animation
        # fargs must match update_lines signature: (num, lines, points, image_plot)
        ani = animation.FuncAnimation(
            fig, self.update_lines, frames=len(self.history_data),
            fargs=(lines, points, image_plot), interval=50, blit=False
        )
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_3d_interactive.py <video_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    print("Analyzing Video...")
    analyzer = SquatAnalyzer()
    analyzer.process_video(video_path, show_window=False) # Only Pass 1 needed to get history
    
    print("Launching Interactive 3D Viewer...")
    viewer = SquatViewer3D(analyzer)
    viewer.start()
