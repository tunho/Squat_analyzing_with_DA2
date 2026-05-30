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

FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio', 'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z', 
    'hip_depth_norm', 'hip_ankle_dist_norm', 'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2', 'mp_visibility'
]

CAM_MAP = {
    50591643: "Front",
    58860488: "Side_R",
    65906101: "Side_L"
}

EX_MAP = {
    5: "Lunge",
    6: "Squat",
    7: "Deadlift"
}

def generate_report():
    print("🚀 Generating Comprehensive Fit3D MAE Report (Rehab-Only Training)...")
    df_re = pd.read_csv(REHAB_DATA).dropna(subset=FEATURES + ['gt_angle'])
    df_fi = pd.read_csv(FIT3D_DATA).dropna(subset=FEATURES + ['gt_angle'])
    
    # Train on EVERYTHING in Rehab
    model = ExtraTreesRegressor(n_estimators=20, random_state=42, n_jobs=1)
    model.fit(df_re[FEATURES], df_re['gt_angle'])
    
    # Evaluate on ALL Fit3D videos
    results = []
    
    # Sort for consistent table
    subjects = sorted(df_fi['subject_id'].unique())
    
    for sid in subjects:
        sub_df = df_fi[df_fi['subject_id'] == sid].copy()
        sub_df['pred'] = model.predict(sub_df[FEATURES])
        
        # Best leg per frame
        best_df = sub_df.sort_values('mp_visibility', ascending=False).drop_duplicates(['ex_id', 'view_type', 'frame_index'])
        
        for (ex_id, cam_id), group in best_df.groupby(['ex_id', 'view_type']):
            view_name = CAM_MAP.get(int(cam_id), "Other")
            if view_name == "Other": continue
            
            ex_name = EX_MAP.get(int(ex_id), f"Ex{ex_id}")
            
            group = group.sort_values('frame_index')
            p = group['pred'].values
            raw = group['mp_knee_angle'].values
            gt = group['gt_angle'].values
            
            if len(p) > 11: p_s = savgol_filter(p, 11, 3)
            else: p_s = p
            
            mae_mod = mean_absolute_error(gt, p_s)
            mae_raw = mean_absolute_error(gt, raw)
            
            results.append({
                "Subject": sid.replace("fit3d_", ""),
                "Exercise": ex_name,
                "View": view_name,
                "MAE_Raw": mae_raw,
                "MAE_Model": mae_mod,
                "Improvement": mae_raw - mae_mod
            })
            
    final_df = pd.DataFrame(results)
    output_path = PROJECT_ROOT / "outputs/ml_correction/v15_mega/fit3d_all_videos_mae.csv"
    final_df.to_csv(output_path, index=False)
    
    print("\n--- 📊 Fit3D Every Video MAE Result (Sorted) ---")
    print(final_df.sort_values(['Subject', 'Exercise', 'View']).to_string(index=False))
    print(f"\n✅ Full CSV saved to: {output_path}")

if __name__ == "__main__":
    generate_report()
