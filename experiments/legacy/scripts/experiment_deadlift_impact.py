import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/lee/exe_est")
REHAB_DATA = PROJECT_ROOT / "outputs/ml_correction/v15_mega/universal_features_rehab_both_legs.csv"
FIT3D_DATA = PROJECT_ROOT / "outputs/ml_correction/v15_mega/features_fit3d_v15_both_legs.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v15_deadlift_impact_experiment"
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

def run_experiment():
    print("🚀 Running Deadlift Impact Experiment (Person-level LOSO)...")
    
    # 1. Load and Map Data
    df_re = pd.read_csv(REHAB_DATA).dropna(subset=FEATURES + ['gt_angle'])
    df_re['person_id'] = df_re['subject_id'].map(PERSON_MAP)
    df_re['view'] = df_re['view_type'] # Already front/side
    
    df_fi = pd.read_csv(FIT3D_DATA).dropna(subset=FEATURES + ['gt_angle'])
    cam_map = {50591643: 'front', 58860488: 'side', 65906101: 'side'}
    df_fi['view'] = df_fi['view_type'].map(cam_map)
    df_fi = df_fi[df_fi['view'].notna()].copy()
    # Map Fit3D subjects to 10, 11, 12
    sid_map = {'fit3d_s03': 10, 'fit3d_s04': 11, 'fit3d_s05': 12}
    df_fi['person_id'] = df_fi['subject_id'].map(sid_map)
    
    # Combine
    full_df = pd.concat([df_re, df_fi], ignore_index=True)
    all_pids = sorted(full_df['person_id'].dropna().unique())
    
    results = []
    print(f"  Total Subjects: {len(all_pids)}")

    for test_pid in all_pids:
        # A person might have multiple exercises (5, 6, 7)
        # Test Person and Training Candidates
        test_person_full = full_df[full_df['person_id'] == test_pid].copy()
        other_people_full = full_df[full_df['person_id'] != test_pid].copy()
        
        # Scenario A: Train only on Squat(6) and Lunge(5) from others
        train_A = other_people_full[other_people_full['ex_id'].isin([5, 6])]
        
        # Scenario B: Train on Squat(6), Lunge(5), AND Deadlift(7) from others
        train_B = other_people_full[other_people_full['ex_id'].isin([5, 6, 7])]
        
        # Test Set: ONLY Squat and Lunge from Test Person (as requested)
        test_data = test_person_full[test_person_full['ex_id'].isin([5, 6])].copy()
        if test_data.empty: continue
            
        # Model A
        model_A = ExtraTreesRegressor(n_estimators=30, random_state=42, n_jobs=-1)
        model_A.fit(train_A[FEATURES], train_A['gt_angle'])
        test_data['pred_A'] = model_A.predict(test_data[FEATURES])
        
        # Model B
        model_B = ExtraTreesRegressor(n_estimators=30, random_state=42, n_jobs=-1)
        model_B.fit(train_B[FEATURES], train_B['gt_angle'])
        test_data['pred_B'] = model_B.predict(test_data[FEATURES])
        
        # --- Visibility Selection ---
        # Pick best leg per frame
        best_leg = test_data.sort_values('mp_visibility', ascending=False).drop_duplicates(['subject_id', 'ex_id', 'view', 'frame_index'])
        
        for (sid, ex_id, view), group in best_leg.groupby(['subject_id', 'ex_id', 'view']):
            group = group.sort_values('frame_index')
            gt = group.gt_angle.values
            raw = group.mp_knee_angle.values
            pA = group.pred_A.values
            pB = group.pred_B.values
            
            if len(pA) > 11:
                pA = savgol_filter(pA, 11, 3)
                pB = savgol_filter(pB, 11, 3)
                
            results.append({
                "Person": test_pid, "Video": sid, "Ex": ex_id, "View": view,
                "MAE_Raw": mean_absolute_error(gt, raw),
                "MAE_Model_NoDL": mean_absolute_error(gt, pA),
                "MAE_Model_WithDL": mean_absolute_error(gt, pB)
            })
            
        print(f"  Person {int(test_pid)} done.")

    res_df = pd.DataFrame(results)
    res_df.to_csv(SAVE_DIR / "deadlift_impact_results.csv", index=False)
    
    # Summarize
    summary_view = res_df.groupby('View')[['MAE_Raw', 'MAE_Model_NoDL', 'MAE_Model_WithDL']].mean().reset_index()
    print("\n--- 🔥 Overall Impact of Deadlift on Squat/Lunge Accuracy ---")
    print(summary_view.to_string(index=False))
    
    total_avg = res_df[['MAE_Raw', 'MAE_Model_NoDL', 'MAE_Model_WithDL']].mean()
    print("\n--- Total Average Performance ---")
    print(total_avg)

if __name__ == "__main__":
    run_experiment()
