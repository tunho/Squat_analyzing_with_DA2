import os
import glob
import torch
from squat_analyzer import SquatAnalyzer

def main():
    print("=== Processing Sample Videos with RAZS + Depth Anything ===")
    
    # 1. Setup Folders
    sample_dir = "/home/lee/exe_est/sample"
    output_dir = "/home/lee/exe_est/sample_results"
    pass1_dir = os.path.join(output_dir, "pass1")
    pass2_dir = os.path.join(output_dir, "pass2")
    os.makedirs(pass1_dir, exist_ok=True)
    os.makedirs(pass2_dir, exist_ok=True)
    
    # 2. Initialize Analyzer (Loads models once)
    print("Initializing SquatAnalyzer...")
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
        pass1_path = os.path.join(pass1_dir, f"pass1_{filename}")
        pass2_path = os.path.join(pass2_dir, f"pass2_{filename}")
        
        print(f"\n[{i+1}/{len(video_files)}] Processing (Pass 1): {filename}")
        try:
            # --- Pass 1: Initial Run ---
            # We use show_window=False to speed up, but we want to save the video.
            counter1, _, issues1, _ = analyzer.process_video(
                video_path=video_path,
                output_path=pass1_path, # Save Pass 1 video
                depth_estimator=analyzer, # [FIXED] Pass self as estimator
                show_window=False
            )
            print(f"  > Pass 1 Done: {counter1} reps. Issues: {issues1}")
            
            # --- Pass 2: Global Optimization ---
            # Extract Global Calibration Data
            counter2_stats, _, calibration_data = analyzer.finalize_analysis()
            print(f"  > Pass 2 Calibration: {calibration_data}")
            
            print(f"[{i+1}/{len(video_files)}] Processing (Pass 2): {filename}")
            # Re-run with Override Parameters (True 2-Pass Video)
            counter2_video, _, issues2, _ = analyzer.process_video(
                video_path=video_path,
                output_path=pass2_path, # Save Pass 2 video
                depth_estimator=analyzer, # [FIXED] Pass self as estimator
                show_window=False,
                override_params=calibration_data # Inject Global Knowledge
            )
            print(f"  > Pass 2 Video Done: {counter2_video} reps. Issues: {issues2}")

        except Exception as e:
            print(f"  > Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== All samples processed (Pass 1 & Pass 2)! ===")
    print(f"Results are saved in:\n - {pass1_dir}\n - {pass2_dir}")

if __name__ == "__main__":
    main()
