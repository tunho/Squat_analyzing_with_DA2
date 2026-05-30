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
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v15_view_expert_experiment"
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

def run_expert_experiment():
    print("🚀 Running View-Specific Expert Experiment (Front Model vs Side Model)...")
    
    # 1. Load and Map Data
    df_re = pd.read_csv(REHAB_DATA).dropna(subset=FEATURES + ['gt_angle'])
    df_re['person_id'] = df_re['subject_id'].map(PERSON_MAP)
    df_re['view'] = df_re['view_type']
    
    df_fi = pd.read_csv(FIT3D_DATA).dropna(subset=FEATURES + ['gt_angle'])
    cam_map = {50591643: 'front', 58860488: 'side', 65906101: 'side'}
    df_fi['view'] = df_fi['view_type'].map(cam_map)
    df_fi = df_fi[df_fi['view'].notna()].copy()
    sid_map = {'fit3d_s03': 10, 'fit3d_s04': 11, 'fit3d_s05': 12}
    df_fi['person_id'] = df_fi['subject_id'].map(sid_map)
    
    # Combine and filter ONLY Squat (6) and Lunge (5)
    full_df = pd.concat([df_re, df_fi], ignore_index=True)
    full_df = full_df[full_df['ex_id'].isin([5, 6])].copy()
    
    # --- NEW: Best-Leg-Only Strategy (Training & Inference) ---
    # For EVERY frame, keep ONLY the leg with the highest visibility
    full_df = full_df.sort_values('mp_visibility', ascending=False).drop_duplicates(
        ['subject_id', 'ex_id', 'view', 'frame_index', 'person_id']
    )
    
    all_pids = sorted(full_df['person_id'].dropna().unique())
    results = []

    for test_pid in all_pids:
        for view_type in ['front', 'side']:
            view_data = full_df[full_df['view'] == view_type].copy()
            if view_data.empty: continue
            
            # Training set: Others, high-visibility only
            train_df = view_data[(view_data['person_id'] != test_pid) & (view_data['mp_visibility'] > 0.6)]
            
            # Test set: This person
            test_df = view_data[view_data['person_id'] == test_pid].copy()
            if test_df.empty or train_df.empty: continue
            
            # Expert Model for this View
            model = ExtraTreesRegressor(n_estimators=30, random_state=42, n_jobs=-1)
            model.fit(train_df[FEATURES], train_df['gt_angle'])
            test_df['pred'] = model.predict(test_df[FEATURES])
            
            for (sid, ex_id), group in test_df.groupby(['subject_id', 'ex_id']):
                group = group.sort_values('frame_index')
                gt = group.gt_angle.values
                raw = group.mp_knee_angle.values
                p = group.pred.values
                
                if len(p) > 11:
                    p = savgol_filter(p, 11, 3)
                
                results.append({
                    "Person": test_pid, "View": view_type, "Ex": ex_id,
                    "MAE_Raw": mean_absolute_error(gt, raw),
                    "MAE_Expert": mean_absolute_error(gt, p)
                })
        print(f"  Person {int(test_pid)} done.")

    res_df = pd.DataFrame(results)
    res_df.to_csv(SAVE_DIR / "view_expert_results.csv", index=False)
    
    print("\n--- 🔥 View-Specific Expert LOSO Results ---")
    summary = res_df.groupby('View')[['MAE_Raw', 'MAE_Expert']].mean().reset_index()
    summary['Improvement'] = summary['MAE_Raw'] - summary['MAE_Expert']
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_expert_experiment()
