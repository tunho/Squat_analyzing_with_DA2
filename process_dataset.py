import os
import glob
import torch
import csv
from squat_analyzer import SquatAnalyzer

def main():
    print("=== Processing FULL DATASET with RAZS 2-Pass ===")
    
    dataset_base = "/home/lee/exe_est/dataset"
    output_base = "/home/lee/exe_est/dataset_results"
    
    # Subfolders
    categories = ["true", "false"]
    
    # Initialize Analyzer
    analyzer = SquatAnalyzer()
    
    results_summary = []
    
    for cat in categories:
        cat_dir = os.path.join(dataset_base, cat)
        target_output_pass1 = os.path.join(output_base, cat, "pass1")
        target_output_pass2 = os.path.join(output_base, cat, "pass2")
        
        os.makedirs(target_output_pass1, exist_ok=True)
        os.makedirs(target_output_pass2, exist_ok=True)
        
        video_files = sorted(glob.glob(os.path.join(cat_dir, "*.mp4")))
        print(f"\n--- Category: {cat} ({len(video_files)} files) ---")
        
        for i, video_path in enumerate(video_files):
            filename = os.path.basename(video_path)
            p1_out = os.path.join(target_output_pass1, f"p1_{filename}")
            p2_out = os.path.join(target_output_pass2, f"p2_{filename}")
            
            print(f"[{cat}] {i+1}/{len(video_files)}: {filename}")
            
            try:
                # Pass 1
                cnt1, _, issues1, _ = analyzer.process_video(video_path, p1_out, analyzer, show_window=False)
                p1_status = "COUNT" if cnt1 > 0 else "NO-COUNT"
                
                # Global Optimization (Pass 2)
                _, _, calibration = analyzer.finalize_analysis()
                
                # Pass 2 Video Generation
                cnt2, _, issues2, _ = analyzer.process_video(video_path, p2_out, analyzer, show_window=False, override_params=calibration)
                p2_status = "COUNT" if cnt2 > 0 else "NO-COUNT"

                results_summary.append({
                    "filename": filename,
                    "ground_truth": cat.upper(),
                    "p1_count": cnt1,
                    "p1_result": p1_status,
                    "p2_count": cnt2,
                    "p2_result": p2_status,
                    "base_k": f"{calibration.get('base_k', 0):.1f}"
                })
                
                print(f"  > P1: {p1_status}({cnt1}), P2: {p2_status}({cnt2}) | K: {calibration.get('base_k', 0):.1f}")
                
            except Exception as e:
                print(f"  > Error: {e}")
                
    # Save Summary CSV
    csv_path = os.path.join(output_base, "dataset_analysis_report.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "ground_truth", "p1_count", "p1_result", "p2_count", "p2_result", "base_k"])
        writer.writeheader()
        writer.writerows(results_summary)
        
    print(f"\nDone! Summary saved to {csv_path}")

if __name__ == "__main__":
    main()
