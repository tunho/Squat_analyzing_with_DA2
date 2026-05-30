import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter

def final_fit3d_measurement():
    print("🚀 Running Final Fit3D Measurement (Squat, Deadlift, Lunge)...")
    df_rehab = pd.read_csv('outputs/ml_correction/v15_mega/universal_features_rehab_both_legs.csv').dropna()
    df_fit = pd.read_csv('outputs/ml_correction/v15_mega/features_fit3d_v15_both_legs.csv').dropna()
    
    features = [
        'mp_knee_angle', 'mp_hip_angle', 'leg_ratio', 'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z', 
        'hip_depth_norm', 'hip_ankle_dist_norm', 'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2', 'mp_visibility'
    ]
    
    subjects = ['fit3d_s03', 'fit3d_s04', 'fit3d_s05']
    results = []
    
    # Train Model on all Rehab + First 2 Fit3D. Test on last Fit3D (Single Fold for speed)
    train_df = pd.concat([df_rehab, df_fit[df_fit.subject_id != 'fit3d_s03']])
    test_df = df_fit[df_fit.subject_id == 'fit3d_s03'].copy()
    
    print(f"  Training on {len(train_df)} frames...")
    model = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(train_df[features], train_df.gt_angle)
    
    print(f"  Evaluating s03...")
    test_df['pred'] = model.predict(test_df[features])
    
    # Selecting best leg
    best = test_df.sort_values('mp_visibility', ascending=False).drop_duplicates(['ex_id', 'view_type', 'frame_index'])
    
    for (ex_id, view), group in best.groupby(['ex_id', 'view_type']):
        group = group.sort_values('frame_index')
        p = group.pred.values
        gt = group.gt_angle.values
        raw = group.mp_knee_angle.values
        if len(p) > 11: p = savgol_filter(p, 11, 3)
        mae_m = mean_absolute_error(gt, p)
        mae_r = mean_absolute_error(gt, raw)
        results.append({"ex": ex_id, "view": view, "mae_model": mae_m, "mae_raw": mae_r})
        
    res_df = pd.DataFrame(results)
    print("\n--- Fit3D (s03) Final Results ---")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    final_fit3d_measurement()
