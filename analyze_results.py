import csv
import os

def analyze_csv(csv_path, label):
    if not os.path.exists(csv_path):
        print(f"[{label}] Report file not found: {csv_path}")
        return

    true_pass = 0
    true_fail = 0
    false_pass = 0
    false_fail = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_true_set = (row['ground_truth'].lower() == 'true')
            # Pass 2 Count Check
            count = int(row.get('p2_count', 0))
            is_success = (count > 0)
            
            if is_true_set:
                if is_success: true_pass += 1
                else: true_fail += 1
            else: # False Set
                if is_success: false_pass += 1
                else: false_fail += 1

    total_true = true_pass + true_fail
    total_false = false_pass + false_fail
    
    print(f"\n=== {label} Analysis ===")
    if total_true > 0:
        print(f"  [True Set]  Success: {true_pass} ({true_pass/total_true*100:.1f}%) / Fail: {true_fail}")
    if total_false > 0:
        print(f"  [False Set] Success: {false_pass} ({false_pass/total_false*100:.1f}%) / Fail: {false_fail}")
        # In False Set, 'Fail' to count is actually 'Good' rejection
        print(f"  > Rejection Rate (Good): {false_fail/total_false*100:.1f}%")

def main():
    # 1. Thigh Mode (Standard RAZS)
    analyze_csv("/home/lee/exe_est/dataset_results/dataset_analysis_report.csv", "Thigh Mode (RAZS)")
    
    # 2. Knee Mode (Knee Angle)
    analyze_csv("/home/lee/exe_est/dataset_results/knee/dataset_summary_knee.csv", "Knee Mode (New)")

if __name__ == "__main__":
    main()
