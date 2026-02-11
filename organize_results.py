import os
import shutil
import csv

def organize_by_count(csv_path, source_base, target_base_name):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    # Target Dirs
    base_dir = "/home/lee/exe_est/dataset_results"
    target_dir = os.path.join(base_dir, "sorted", target_base_name)
    
    dir_success = os.path.join(target_dir, "success_count")
    dir_fail = os.path.join(target_dir, "fail_no_count")
    
    os.makedirs(dir_success, exist_ok=True)
    os.makedirs(dir_fail, exist_ok=True)
    
    print(f"Organizing {target_base_name} into {target_dir}...")
    
    count_success = 0
    count_fail = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            # We look for Pass 2 video usually
            # Filename in CSV is 'true_1.mp4'
            # Actual file in dataset_results/true/pass2/p2_true_1.mp4
            
            ground_truth = row['ground_truth'].lower() # true / false
            
            # Construct Source Path
            # For Thigh Mode: dataset_results/{cat}/pass2/p2_{filename}
            # For Knee Mode: dataset_results/knee/{cat}/pass2/p2_{filename}
            
            if "knee" in target_base_name.lower():
                src_path = os.path.join(base_dir, "knee", ground_truth, "pass2", f"p2_{filename}")
            else:
                src_path = os.path.join(base_dir, ground_truth, "pass2", f"p2_{filename}")
            
            if not os.path.exists(src_path):
                print(f"Warning: Video not found {src_path}")
                continue
                
            # Check Count
            cnt = int(row.get('p2_count', 0))
            
            # Copy to Target based ONLY on Count
            # If Count > 0 -> "Counted" (Success Folder)
            # If Count == 0 -> "Not Counted" (Fail/No-Count Folder)
            
            if cnt > 0:
                dst = os.path.join(dir_success, f"[{ground_truth.upper()}]_CNT_{cnt}_{filename}")
                shutil.copy2(src_path, dst)
                count_success += 1
            else:
                dst = os.path.join(dir_fail, f"[{ground_truth.upper()}]_NOCNT_{filename}")
                shutil.copy2(src_path, dst)
                count_fail += 1

    print(f"  > Success (Counted): {count_success}")
    print(f"  > Fail (No Count): {count_fail}")
    print(f"  > Saved to: {target_dir}")

def main():
    # 1. Thigh Mode
    organize_by_count(
        "/home/lee/exe_est/dataset_results/dataset_analysis_report.csv", 
        "/home/lee/exe_est/dataset_results", 
        "thigh_mode"
    )
    
    # 2. Knee Mode
    organize_by_count(
        "/home/lee/exe_est/dataset_results/knee/dataset_summary_knee.csv", 
        "/home/lee/exe_est/dataset_results/knee",
        "knee_mode"
    )

if __name__ == "__main__":
    main()
