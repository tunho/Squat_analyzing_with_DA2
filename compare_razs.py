import os
import glob
import cv2
import pandas as pd
from squat_analyzer import SquatAnalyzer

def main():
    print("=== RAZS 1-Pass vs 2-Pass Comparison ===")
    
    # 1. Setup Folders
    # We will look directly into the dataset root's main categories
    # Assuming standard structure: dataset/true, dataset/false etc.
    dataset_root = "/home/lee/exe_est/dataset"
    output_dir = "/home/lee/exe_est/comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create base directories for 1-pass and 2-pass
    base_1pass = os.path.join(output_dir, "1pass")
    base_2pass = os.path.join(output_dir, "2pass")
    
    # Create subfolders for pass/non_pass in each base
    for base in [base_1pass, base_2pass]:
        os.makedirs(os.path.join(base, "pass"), exist_ok=True)
        os.makedirs(os.path.join(base, "non_pass"), exist_ok=True)
    
    # Categories to process
    categories = ["true", "false"] 
    
    print("Initializing SquatAnalyzer...")
    analyzer = SquatAnalyzer()
    
    results_list = []

    for cat in categories:
        cat_path = os.path.join(dataset_root, cat)
        if not os.path.isdir(cat_path):
            print(f"Skipping {cat} (Not a directory)")
            continue
            
        video_files = glob.glob(os.path.join(cat_path, "*.mp4"))
        print(f"\nProcessing Category: {cat} (Found {len(video_files)} videos)")
        
        for video_path in video_files:
            filename = os.path.basename(video_path)
            print(f"  > Processing: {filename}")
            
            # --- 1-PASS (Adaptive) ---
            analyzer.reset()
            
            # Temporary path for 1-pass video
            temp_1pass = os.path.join(output_dir, f"temp_1pass_{filename}")
            
            try:
                # Run 1-Pass
                c1, issues1, _, _ = analyzer.process_video(
                    video_path=video_path,
                    output_path=temp_1pass,
                    depth_estimator=analyzer,
                    show_window=False
                )
                
                # Sort 1-Pass Video
                status1 = "pass" if c1 >= 1 else "non_pass"
                dest_1pass = os.path.join(base_1pass, status1, filename)
                if os.path.exists(temp_1pass):
                    os.rename(temp_1pass, dest_1pass)
                
                # --- 2-PASS (Global Optimization) ---
                # 1. Get calibrated parameters from history
                count_2pass_stats, issues2_stats, calib_data = analyzer.finalize_analysis()
                
                # 2. Re-run process_video with FIXED parameters to generate 2-Pass Video
                # We use the calibration data to force the analyzer into a stable state
                
                status2 = "pass" if count_2pass_stats >= 1 else "non_pass"
                dest_2pass = os.path.join(base_2pass, status2, filename)
                
                print(f"    Running 2-Pass Video Generation... (Status: {status2})")
                
                # Re-run with override_params
                # We don't need to return count here as we trust finalize_analysis more, 
                # but we can check if they match.
                c2_video, _, _, _ = analyzer.process_video(
                    video_path=video_path,
                    output_path=dest_2pass,
                    depth_estimator=analyzer,
                    show_window=False,
                    override_params=calib_data
                )
                
                print(f"    Result: 1-Pass={c1} reps, 2-Pass(Stats)={count_2pass_stats}, 2-Pass(Video)={c2_video}")
                
                results_list.append({
                    "Category": cat,
                    "Filename": filename,
                    "1-Pass Status": status1,
                    "1-Pass Reps": c1,
                    "2-Pass Reps": count_2pass_stats, 
                    "1-Pass Issues": str(issues1),
                    "2-Pass Issues": str(issues2_stats)
                })
                
            except Exception as e:
                print(f"    Error processing {filename}: {e}")
                if os.path.exists(temp_1pass):
                    os.remove(temp_1pass)

    # 4. Save CSV Report
    if results_list:
        df = pd.DataFrame(results_list)
        report_path = os.path.join(output_dir, "comparison_report.csv")
        df.to_csv(report_path, index=False)
        print(f"\nComparison Report saved to: {report_path}")
        print(df)
    else:
        print("\nNo videos processed.")

if __name__ == "__main__":
    main()
