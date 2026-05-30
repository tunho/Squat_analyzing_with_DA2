import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/lee/exe_est")
DATA_PATH = PROJECT_ROOT / "outputs/ml_correction/v13_mega/universal_features_v13_mega.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v13_mega_loso_9subjects"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Standard features
FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio',
    'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z',
    'hip_depth_norm', 'hip_ankle_dist_norm',
    'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2'
]

# Mapping PM subjects to unique 9 Person IDs
PERSON_MAP = {
    'PM_008': 1, 
    'PM_021': 2, 'PM_022': 2, 
    'PM_028': 3, 'PM_029': 3, 
    'PM_037': 4, 'PM_038': 4, 
    'PM_042': 5, 'PM_043': 5, 
    'PM_104': 6, 'PM_105': 6, 
    'PM_125': 7, 'PM_126': 7,
    'PM_112': 8, 'PM_113': 8, 
    'PM_117a': 9, 'PM_117b': 9, 'PM_118': 9
}

def refine_bone_lengths(df):
    refined_dfs = []
    groups = df.groupby(['subject_id', 'ex_id', 'view_type'])
    for _, group in groups:
        group = group.copy().sort_values('frame_index')
        if len(group) < 2:
            refined_dfs.append(group)
            continue
        k = group[['k_x', 'k_y', 'k_z']].values
        a = group[['a_x', 'a_y', 'a_z']].values
        thigh_lens = np.linalg.norm(k, axis=1)
        shank_lens = np.linalg.norm(a - k, axis=1)
        m_thigh = np.median(thigh_lens)
        m_shank = np.median(shank_lens)
        k_refined = (k / (thigh_lens[:, None] + 1e-8)) * m_thigh
        a_dir = (a - k) / (shank_lens[:, None] + 1e-8)
        a_refined = k_refined + (a_dir * m_shank)
        group[['k_x', 'k_y', 'k_z']] = k_refined
        group[['a_x', 'a_y', 'a_z']] = a_refined
        refined_dfs.append(group)
    return pd.concat(refined_dfs)

def run_experiment_9subjects():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return
        
    full_df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    
    # 1. Filter: STRICTLY EX5, EX6 and Front/Side views only
    df = full_df[full_df['view_type'].isin(['front', 'side'])].copy()
    
    # 2. Map person_id
    df['person_id'] = df['subject_id'].map(PERSON_MAP)
    
    # Fit3D subjects (not in PERSON_MAP) get their own IDs
    fit3d_mask = df['subject_id'].str.contains('fit3d')
    df.loc[fit3d_mask, 'person_id'] = df.loc[fit3d_mask, 'subject_id'].apply(lambda x: 10 + int(x.split('_s')[1]))
    
    # Subjects to test on: Persons 1 to 9 (Rehab Subjects)
    rehab_persons = sorted([p for p in df['person_id'].unique() if p <= 9])
    
    print(f"🚀 Starting LOSO for {len(rehab_persons)} Rehab Persons")
    print(f"Dataset Size: {len(df)} frames (Front/Side, Ex 5/6)")

    results = []

    for test_pid in rehab_persons:
        train_df = df[df['person_id'] != test_pid]
        test_df = df[df['person_id'] == test_pid].copy()
        
        # Train model on all other persons (Rehab + Fit3D)
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(train_df[FEATURES], train_df['gt_angle'])
        
        # Predict on test person
        test_df_refined = refine_bone_lengths(test_df)
        preds = model.predict(test_df_refined[FEATURES])
        test_df_refined['pred_angle'] = preds
        
        # Sequence-based evaluation per video
        # (Person PID has multiple subject_id/ex_id/view_type videos)
        for (sid, ex_id, view), sub_group in test_df_refined.groupby(['subject_id', 'ex_id', 'view_type']):
            p_seq = sub_group['pred_angle'].values
            raw_seq = sub_group['mp_knee_angle'].values
            gt_seq = sub_group['gt_angle'].values
            
            if len(p_seq) > 11:
                p_seq_smoothed = savgol_filter(p_seq, 11, 3)
            else:
                p_seq_smoothed = p_seq
            
            mae_model = mean_absolute_error(gt_seq, p_seq_smoothed)
            mae_raw = mean_absolute_error(gt_seq, raw_seq)
            
            results.append({
                "person_id": test_pid,
                "video_id": sid, 
                "ex_id": ex_id, 
                "view": view, 
                "mae_model": mae_model, 
                "mae_raw": mae_raw
            })
            
        print(f"  Person {test_pid} done.", flush=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(SAVE_DIR / "loso_9subjects_metrics.csv", index=False)
    
    # Final Summary
    summary = res_df.groupby(['view'])[['mae_raw', 'mae_model']].mean().reset_index()
    summary['improvement'] = summary['mae_raw'] - summary['mae_model']
    
    print("\n--- 🔥 9-Subject LOSO FINAL RESULTS (Front/Side ONLY) ---")
    print(summary.round(4).to_string(index=False))
    
    # Person-wise summary
    p_summary = res_df.groupby('person_id')[['mae_raw', 'mae_model']].mean().reset_index()
    print("\n--- Person-wise Performance ---")
    print(p_summary.round(4).to_string(index=False))

if __name__ == "__main__":
    run_experiment_9subjects()
