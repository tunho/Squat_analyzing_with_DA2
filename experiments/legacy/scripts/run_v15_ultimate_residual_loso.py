import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Paths
PROJECT_ROOT = Path("/home/lee/exe_est")
DATA_PATH = PROJECT_ROOT / "outputs/ml_correction/v15_mega/ultimate_features_all.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v15_ultimate_residual_benchmark"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "mp_knee_angle", "mp_hip_angle", "leg_ratio", "angle_vel_smooth",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "mp_visibility", "hip_y", "hip_ankle_dist_norm",
    "knee_lag1", "knee_lag2"
]

PERSON_MAP = {
    'PM_008': 1, 'PM_021': 2, 'PM_022': 2, 'PM_028': 3, 'PM_029': 3, 
    'PM_037': 4, 'PM_038': 4, 'PM_042': 5, 'PM_043': 5, 'PM_104': 6, 
    'PM_105': 6, 'PM_125': 7, 'PM_126': 7, 'PM_112': 8, 'PM_113': 8, 
    'PM_117a': 9, 'PM_117b': 9, 'PM_118': 9,
    'fit3d_s03': 10, 'fit3d_s04': 11, 'fit3d_s05': 12
}

def run_residual_loso(mode="rehab_only"):
    print(f"🚀 Running Residual LOSO Benchmark (Mode: {mode})...")
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ['gt_angle', 'mp_knee_angle'])
    df['person_id'] = df['subject_id'].map(PERSON_MAP)
    
    # Calculate Residual
    df['residual'] = df['gt_angle'] - df['mp_knee_angle']
    
    if mode == "rehab_only":
        df = df[df['person_id'] <= 9].copy()
    
    pids = sorted(df['person_id'].dropna().unique())
    results = []

    for test_pid in pids:
        # Separate Views
        for view_type in ['front', 'side']:
            view_df = df[df['view'] == view_type].copy()
            if view_df.empty: continue
            
            # Train: Others
            train_df = view_df[view_df['person_id'] != test_pid].copy()
            # Test: This person (Best Leg Only)
            test_df_full = view_df[view_df['person_id'] == test_pid].copy()
            if train_df.empty or test_df_full.empty: continue
            
            test_df = test_df_full.sort_values('mp_visibility', ascending=False).drop_duplicates(['subject_id', 'ex_id', 'frame_index'])
            
            # Scaler (matching v5)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[FEATURES])
            X_test = scaler.transform(test_df[FEATURES])
            
            # Train Residual Model
            model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, train_df['residual'])
            
            # Predict Residual
            test_df['residual_pred'] = model.predict(X_test)
            test_df['pred_angle'] = test_df['mp_knee_angle'] + test_df['residual_pred']
            
            for (sid, ex_id), group in test_df.groupby(['subject_id', 'ex_id']):
                group = group.sort_values('frame_index')
                gt = group.gt_angle.values
                raw = group.mp_knee_angle.values
                p = group.pred_angle.values
                if len(p) > 11: p = savgol_filter(p, 11, 3)
                
                mae_raw = mean_absolute_error(gt, raw)
                mae_mod = mean_absolute_error(gt, p)
                results.append({"Person": test_pid, "View": view_type, "Ex": ex_id, "MAE_Raw": mae_raw, "MAE_Model": mae_mod})
                
        print(f"  Person {int(test_pid)} done.")

    res_df = pd.DataFrame(results)
    summary = res_df.groupby('View')[['MAE_Raw', 'MAE_Model']].mean()
    print(f"\n--- {mode.upper()} Result ---")
    print(summary)
    return res_df

if __name__ == "__main__":
    if DATA_PATH.exists():
        r_rehab = run_residual_loso("rehab_only")
        r_unified = run_residual_loso("unified")
        
        # Save results
        r_rehab.to_csv(SAVE_DIR / "rehab_only_residual_results.csv", index=False)
        r_unified.to_csv(SAVE_DIR / "unified_residual_results.csv", index=False)
