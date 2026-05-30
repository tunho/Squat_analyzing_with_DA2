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
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v13_mega_loso_comparison"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Standard features
FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio',
    'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z',
    'hip_depth_norm', 'hip_ankle_dist_norm',
    'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2'
]

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

def run_experiment():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return
        
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    subjects = sorted(df['subject_id'].unique())
    results = []
    
    print(f"🚀 Starting v13 Mega LOSO Comparison ({len(subjects)} subjects, {len(df)} frames)")

    for i, test_sid in enumerate(subjects):
        train_df = df[df['subject_id'] != test_sid]
        test_df = df[df['subject_id'] == test_sid].copy()
        
        if len(test_df) == 0:
            continue

        # Fit model on all subjects except the test subject
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(train_df[FEATURES], train_df['gt_angle'])
        
        # Predict on test subject
        test_df_refined = refine_bone_lengths(test_df)
        preds = model.predict(test_df_refined[FEATURES])
        test_df_refined['pred_angle'] = preds
        
        # Sequence-based smoothing and evaluation
        for (ex_id, view), sub_group in test_df_refined.groupby(['ex_id', 'view_type']):
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
                "subject": test_sid, 
                "ex_id": ex_id, 
                "view": view, 
                "mae_model": mae_model, 
                "mae_raw": mae_raw,
                "improvement": mae_raw - mae_model
            })
            
        print(f"  [{i+1}/{len(subjects)}] Subject {test_sid} done.", flush=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(SAVE_DIR / "v13_loso_comparison_metrics.csv", index=False)
    
    # Summary by view
    summary = res_df.groupby(['view'])[['mae_raw', 'mae_model']].mean().reset_index()
    summary['improvement'] = summary['mae_raw'] - summary['mae_model']
    summary['percent_reduction'] = (summary['improvement'] / summary['mae_raw']) * 100
    
    print("\n--- 🔥 v13 Final Mega Benchmark: Model vs Raw (21-Fold LOSO) ---")
    print(summary.round(4).to_string(index=False))
    
    print(f"\nFinal Overall Model MAE: {res_df['mae_model'].mean():.4f}")
    print(f"Final Overall Raw MAE: {res_df['mae_raw'].mean():.4f}")

if __name__ == "__main__":
    run_experiment()
