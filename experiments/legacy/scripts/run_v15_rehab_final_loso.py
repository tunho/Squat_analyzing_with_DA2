import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/lee/exe_est")
REHAB_DATA = PROJECT_ROOT / "outputs/ml_correction/v15_mega/universal_features_rehab_both_legs.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v15_rehab_final_benchmark"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio', 'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z', 
    'hip_depth_norm', 'hip_ankle_dist_norm', 'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2', 'mp_visibility'
]

PERSON_MAP = {
    'PM_008': 1, 'PM_021': 2, 'PM_022': 2, 'PM_028': 3, 'PM_029': 3, 
    'PM_037': 4, 'PM_038': 4, 'PM_042': 5, 'PM_043': 5, 'PM_104': 6, 
    'PM_105': 6, 'PM_125': 7, 'PM_126': 7, 'PM_112': 8, 'PM_113': 8, 
    'PM_117a': 9, 'PM_117b': 9, 'PM_118': 9
}

def run_rehab_final_benchmark():
    print("🚀 Running Final Rehab Benchmark (9 Subjects, View-Expert LOSO)...")
    df = pd.read_csv(REHAB_DATA).dropna(subset=FEATURES + ['gt_angle'])
    df['person_id'] = df['subject_id'].map(PERSON_MAP)
    df['view'] = df['view_type']
    
    pids = sorted(df['person_id'].dropna().unique())
    results = []

    for test_pid in pids:
        # Separate Front/Side models for peak precision
        for view_type in ['front', 'side']:
            view_data = df[df['view'] == view_type].copy()
            if view_data.empty: continue
            
            # Training: All other people, both legs (where visibility > 0.6)
            train_df = view_data[(view_data['person_id'] != test_pid) & (view_data['mp_visibility'] > 0.6)].copy()
            
            # Test: This person, only the best leg per frame
            test_df_full = view_data[view_data['person_id'] == test_pid].copy()
            if test_df_full.empty or train_df.empty: continue
            
            # Keep only best leg for test eval
            test_df = test_df_full.sort_values('mp_visibility', ascending=False).drop_duplicates(['subject_id', 'ex_id', 'frame_index'])
            
            # Train Expert Model
            model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(train_df[FEATURES], train_df['gt_angle'])
            
            test_df['pred'] = model.predict(test_df[FEATURES])
            
            for (sid, ex_id), group in test_df.groupby(['subject_id', 'ex_id']):
                group = group.sort_values('frame_index')
                raw = group['mp_knee_angle'].values
                gt = group['gt_angle'].values
                p = group['pred'].values
                
                if len(p) > 11: p_smooth = savgol_filter(p, 11, 3)
                else: p_smooth = p
                
                mae_raw = mean_absolute_error(gt, raw)
                mae_mod = mean_absolute_error(gt, p_smooth)
                
                results.append({
                    "Person": test_pid, "Subject": sid, "Ex": ex_id, "View": view_type,
                    "MAE_Raw": mae_raw, "MAE_Model": mae_mod
                })
        print(f"  Person {int(test_pid)} benchmarked.")

    res_df = pd.DataFrame(results)
    res_df.to_csv(SAVE_DIR / "final_rehab_benchmark_results.csv", index=False)
    
    print("\n--- ✅ Final Rehab Benchmark Success (9-Person Expert LOSO) ---")
    summary = res_df.groupby('View')[['MAE_Raw', 'MAE_Model']].mean()
    summary['Improvement'] = summary['MAE_Raw'] - summary['MAE_Model']
    print(summary)
    
    print("\n--- Subject-Wise Performance ---")
    subj_summary = res_df.groupby('Person')[['MAE_Raw', 'MAE_Model']].mean()
    print(subj_summary)

if __name__ == "__main__":
    run_rehab_final_benchmark()
