import cv2
import os
import sys
from squat_analyzer_knee import SquatAnalyzer

def run_multiview_demo(video_path, output_dir="output_multiview"):
    """
    Runs the SquatAnalyzer (Knee) on a video and generates a multi-view 3D visualization.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Analyzer
    analyzer = SquatAnalyzer()
    
    # 1. Run Analysis (1-Pass Data Collection)
    # This also populates history_data
    # We don't need the immediate video output for this demo, but we can keep it inside output_dir
    temp_output = os.path.join(output_dir, "temp_process.mp4")
    print("Step 1: Running 1-Pass Analysis (Data Collection)...")
    analyzer.process_video(video_path, output_path=temp_output, show_window=False)
    
    # 2. Finalize Analysis (2-Pass & 3-Pass Optimization)
    print("Step 2: Running 2-Pass & 3-Pass Optimization...")
    count, final_angles, calibration_data = analyzer.finalize_analysis()
    
    base_k = calibration_data.get('base_k', 1.0)
    dr = calibration_data.get('dr', 0.0)
    # Note: stable_radii is not returned by finalize_analysis in the current signature, 
    # but we can re-calculate it or modify finalize_analysis to return it.
    # Alternatively, since we have the logic inside create_multiview_video to use parameters...
    # Wait, create_multiview_video needs stable_radii passed to it.
    # Let's peek at finalize_analysis updates. Ah, I mistakenly didn't return stable_radii in the previous edit?
    # Let's just re-calculate it here using the same logic for safety, OR update the class to store it.
    
    # Better approach: The class has history_data. calculating stable_radii is deterministic.
    # Let's replicate the logic quickly or assume the method does it internally?
    # Actually, let's update finalize_analysis to store stable_radii in self or return it.
    # Checking previous code... I returned {'base_k': base_k, 'dr': dr_stable}. Missing stable_radii.
    
    # Re-calculation logic (Rigid Body Strategy) logic mirror:
    g_max_thigh = max(f['thigh_len_2d'] for f in analyzer.history_data)
    g_max_shank = max(f['shank_len_2d'] for f in analyzer.history_data)
    final_thigh_len = max(g_max_thigh, g_max_shank * 1.2)
    final_shank_len = final_thigh_len / 1.2
    
    stable_radii = analyzer.corrector.get_radii(final_thigh_len, final_shank_len)
    
    print(f" Optimization Result: K={base_k:.3f}, Dr={dr:.1f}")
    
    # 3. Generate Multi-View Video
    print("Step 3: Generating Multi-View Visualization...")
    video_name = os.path.basename(video_path).split('.')[0]
    viz_output_path = os.path.join(output_dir, f"{video_name}_3d_multiview.mp4")
    
    analyzer.create_multiview_video(viz_output_path, base_k, dr, stable_radii)
    print(f"Done! Saved to {viz_output_path}")

if __name__ == "__main__":
    # Target Video
    target_video = "dataset/true/true_14.mp4" 
    
    if len(sys.argv) > 1:
        target_video = sys.argv[1]
        
    run_multiview_demo(target_video)
