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
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v15_rehab_only_on_fit3d"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio', 'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z', 
    'hip_depth_norm', 'hip_ankle_dist_norm', 'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2', 'mp_visibility'
]

CAM_MAP = {50591643: 'front', 58860488: 'side', 65906101: 'side'}

def runner():
    print("🚀 Training Rehab-Only Model (Augmented) and Evaluating on Fit3D...")
    df_rehab = pd.read_csv(REHAB_DATA).dropna(subset=FEATURES + ['gt_angle'])
    df_fit = pd.read_csv(FIT3D_DATA).dropna(subset=FEATURES + ['gt_angle'])
    
    # Train on ALL Rehab
    model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(df_rehab[FEATURES], df_rehab['gt_angle'])
    
    # Evaluate on each Fit3D subject
    fit3d_subjects = sorted(df_fit['subject_id'].unique())
    results = []

    for sid in fit3d_subjects:
        test_df = df_fit[df_fit['subject_id'] == sid].copy()
        
        # Predict
        test_df['pred'] = model.predict(test_df[FEATURES])
        
        # Max visibility selection per frame
        best = test_df.sort_values('mp_visibility', ascending=False).drop_duplicates(['ex_id', 'view_type', 'frame_index'])
        
        for (cam_id, ex_id), group in best.groupby(['view_type', 'ex_id']):
            view_name = CAM_MAP.get(int(cam_id), 'unknown')
            if view_name == 'unknown': continue
            
            group = group.sort_values('frame_index')
            p = group['pred'].values
            raw = group['mp_knee_angle'].values
            gt = group['gt_angle'].values
            
            if len(p) > 11: p_smooth = savgol_filter(p, 11, 3)
            else: p_smooth = p
            
            mae_mod = mean_absolute_error(gt, p_smooth)
            mae_raw = mean_absolute_error(gt, raw)
            
            results.append({
                "subject": sid, "view": view_name, "ex_id": ex_id,
                "mae_model": mae_mod, "mae_raw": mae_raw
            })
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(SAVE_DIR / "rehab_only_on_fit3d_results.csv", index=False)
    
    # Summary by Subject and View
    summary = res_df.groupby(['subject', 'view'])[['mae_raw', 'mae_model']].mean().reset_index()
    print("\n--- Final Result: Rehab-Only Model on Fit3D ---")
    print(summary.round(4).to_string(index=False))
    
    # Summary by Exercise
    ex_summary = res_df.groupby(['ex_id', 'view'])[['mae_raw', 'mae_model']].mean().reset_index()
    print("\n--- Performance by Exercise (5=Lunge, 6=Squat, 7=Deadlift) ---")
    print(ex_summary.round(4).to_string(index=False))

if __name__ == "__main__":
    runner()
