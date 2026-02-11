
import cv2
import glob
import os
import torch
from squat_analyzer_knee import SquatAnalyzer

def main():
    print("=== Processing Sample Videos with KNEE ANGLE MODE (Simple) ===")
    
    # 1. Setup Folders
    sample_dir = "/home/lee/exe_est/sample"
    output_dir = "/home/lee/exe_est/sample_results/knee"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Initialize Analyzer (Knee Mode)
    print("Initializing SquatAnalyzer (Knee Mode)...")
    analyzer = SquatAnalyzer()
    
    # 3. Get Sample Videos
    video_files = sorted(glob.glob(os.path.join(sample_dir, "*.mp4")))
    print(f"Found {len(video_files)} sample videos.")
    
    if not video_files:
        print("No .mp4 files found in sample directory.")
        return

    # 4. Process Each Video
    for i, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        output_path = os.path.join(output_dir, f"knee_{filename}")
        
        print(f"\n[{i+1}/{len(video_files)}] Processing: {filename}")
        try:
            # --- Pass 1: Initial Run ---
            # We use show_window=False to speed up, but we want to save the video.
            counter1, _, issues1, _ = analyzer.process_video(
                video_path=video_path,
                output_path=output_path.replace("knee_", "knee_p1_"), # Save Pass 1
                depth_estimator=analyzer, 
                show_window=False
            )
            print(f"  > Pass 1 Done: {counter1} reps.")
            
            # --- Pass 2: Global Optimization ---
            # Extract Global Calibration Data
            counter_stats, _, calibration_data = analyzer.finalize_analysis()
            print(f"  > Pass 2 Calibration: {calibration_data}")
            
            # Re-run with Override Parameters (True 2-Pass Video)
            counter2, _, issues2, _ = analyzer.process_video(
                video_path=video_path,
                output_path=output_path.replace("knee_", "knee_p2_"), # Save Pass 2
                depth_estimator=analyzer,
                show_window=False,
                override_params=calibration_data # Inject Global Knowledge
            )
            print(f"  > Pass 2 Done: {counter2} reps.")
            
        except Exception as e:
            print(f"  > Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Knee Mode Sample Processing Complete! ===")
    print(f"Results are saved in: {output_dir}")

if __name__ == "__main__":
    main()
