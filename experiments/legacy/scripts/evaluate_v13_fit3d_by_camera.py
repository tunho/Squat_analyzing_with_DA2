import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/lee/exe_est")
MEGA_PATH = PROJECT_ROOT / "outputs/ml_correction/v13_mega/universal_features_v13_mega.csv"
FIT3D_PATH = PROJECT_ROOT / "outputs/ml_correction/v13_mega/features_fit3d_v13_all.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v13_fit3d_benchmark"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Features
FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio',
    'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z',
    'hip_depth_norm', 'hip_ankle_dist_norm',
    'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2'
]

# Cam ID Map for reporting
CAM_NAMES = {
    50591643: "Front (5059...)",
    58860488: "Side_R (5886...)",
    60457274: "Back (6045...)",
    65906101: "Side_L (6590...)"
}

def refine_bone_lengths(df):
    refined_dfs = []
    # If using Fit3D raw as test, columns are same but view_type is camera_id
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

def evaluate_fit3d():
    if not MEGA_PATH.exists() or not FIT3D_PATH.exists():
        print("Required datasets not found.")
        return
        
    df_mega = pd.read_csv(MEGA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    df_fit_raw = pd.read_csv(FIT3D_PATH).dropna(subset=FEATURES + ['gt_angle'])
    
    # We will iterate through Fit3D subjects in the raw file to maintain camera IDs
    fit3d_subjects = sorted(df_fit_raw['subject_id'].unique())
    results = []
    
    print(f"🚀 Evaluating Performance on Fit3D ({len(fit3d_subjects)} subjects) by Camera Angle")

    for test_sid in fit3d_subjects:
        # Train on EVERYTHING except this Fit3D subject.
        # Note: df_mega uses mapped view_type ('front'/'side'), but that's fine for training.
        train_df = df_mega[df_mega['subject_id'] != test_sid]
        
        # Test on the target Fit3D subject (using raw metadata for camera IDs)
        test_df = df_fit_raw[df_fit_raw['subject_id'] == test_sid].copy()
        
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(train_df[FEATURES], train_df['gt_angle'])
        
        # Predict
        test_df_refined = refine_bone_lengths(test_df)
        preds = model.predict(test_df_refined[FEATURES])
        test_df_refined['pred_angle'] = preds
        
        # Breakdown by camera
        for (cam_id, ex_id), sub_group in test_df_refined.groupby(['view_type', 'ex_id']):
            p_seq = sub_group['pred_angle'].values
            raw_seq = sub_group['mp_knee_angle'].values
            gt_seq = sub_group['gt_angle'].values
            
            if len(p_seq) > 11:
                p_seq_smoothed = savgol_filter(p_seq, 11, 3)
            else:
                p_seq_smoothed = p_seq
                
            mae_mod = mean_absolute_error(gt_seq, p_seq_smoothed)
            mae_raw = mean_absolute_error(gt_seq, raw_seq)
            
            results.append({
                "subject": test_sid,
                "camera_id": str(cam_id),
                "camera_name": CAM_NAMES.get(int(cam_id), str(cam_id)),
                "ex_id": ex_id,
                "mae_model": mae_mod,
                "mae_raw": mae_raw
            })
            
        print(f"  Subject {test_sid} done.")

    res_df = pd.DataFrame(results)
    res_df.to_csv(SAVE_DIR / "fit3d_camera_benchmark_v13.csv", index=False)
    
    # Summary by Camera
    summary = res_df.groupby('camera_name')[['mae_raw', 'mae_model']].mean().reset_index()
    summary['improvement'] = summary['mae_raw'] - summary['mae_model']
    
    print("\n--- 🔥 Fit3D Benchmark: Model vs Raw by Camera ---")
    print(summary.round(4).to_string(index=False))
    
    # Summary by Camera and Exercise
    summary_ex = res_df.groupby(['camera_name', 'ex_id'])[['mae_raw', 'mae_model']].mean().reset_index()
    print("\n--- Detailed by Exercise ---")
    print(summary_ex.round(4).to_string(index=False))

if __name__ == "__main__":
    evaluate_fit3d()
