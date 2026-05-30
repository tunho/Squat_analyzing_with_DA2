import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# --- SETTINGS ---
PROJECT_ROOT = Path("/home/lee/exe_est")
DATA_PATH = PROJECT_ROOT / "outputs/ml_correction/v15_mega/ultimate_features_all.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v15_ultimate_expert_bench"
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

def run_exercise_expert_loso(ex_id):
    ex_name = "Squat" if ex_id == 6 else "Lunge"
    print(f"🎬 Running {ex_name} Expert LOSO Benchmark...")
    
    df_raw = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    df_raw['person_id'] = df_raw['subject_id'].map(PERSON_MAP)
    # Filter only this Exercise (All 12 Subjects: Rehab + Fit3D)
    df = df_raw[df_raw['ex_id'] == ex_id].copy()
    
    # Target: Residual
    df['residual'] = df['gt_angle'] - df['mp_knee_angle']
    
    pids = sorted(df['person_id'].dropna().unique())
    results = []

    for test_pid in pids:
        # Train: All other subjects, this exercise (BOTH Front and Side)
        train_df = df[df['person_id'] != test_pid].copy()
        # Test: This subject, this exercise (BOTH Front and Side)
        test_df_full = df[df['person_id'] == test_pid].copy()
        
        if train_df.empty or test_df_full.empty: continue
        
        # Keep only best leg per frame for evaluation
        test_df = test_df_full.sort_values('mp_visibility', ascending=False).drop_duplicates(['subject_id', 'view', 'frame_index'])
        
        # Scaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[FEATURES])
        X_test = scaler.transform(test_df[FEATURES])
        
        # Model (Single Expert Model for ALL Views)
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, train_df['residual'])
        
        # Predict
        test_df['residual_pred'] = model.predict(X_test)
        test_df['pred_angle'] = test_df['mp_knee_angle'] + test_df['residual_pred']
        
        # Evaluate separately by view to show metrics
        for (sid, view_type), group in test_df.groupby(['subject_id', 'view']):
            group = group.sort_values('frame_index')
            gt = group.gt_angle.values
            raw = group.mp_knee_angle.values
            p = group.pred_angle.values
            if len(p) > 11: p = savgol_filter(p, 11, 3)
            
            mae_raw = mean_absolute_error(gt, raw)
            mae_mod = mean_absolute_error(gt, p)
            results.append({"Person": test_pid, "Subject": sid, "Ex": ex_name, "View": view_type, "MAE_Raw": mae_raw, "MAE_Model": mae_mod})
            
        print(f"  Person {int(test_pid)} {ex_name} done.")

    return pd.DataFrame(results)

if __name__ == "__main__":
    if DATA_PATH.exists():
        r_squat = run_exercise_expert_loso(6)
        r_lunge = run_exercise_expert_loso(5)
        
        final_res = pd.concat([r_squat, r_lunge], ignore_index=True)
        final_res.to_csv(SAVE_DIR / "final_expert_bench_results.csv", index=False)
        
        # Summary
        print("\n--- 🔥 FINAL EXPERT ENSEMBLE RESULTS ---")
        summary = final_res.groupby(['Ex', 'View'])[['MAE_Raw', 'MAE_Model']].mean().reset_index()
        summary['Improvement'] = summary['MAE_Raw'] - summary['MAE_Model']
        print(summary.to_string(index=False))
        
        total_avg = final_res[['MAE_Raw', 'MAE_Model']].mean()
        print(f"\n🚀 Total Average MAE: {total_avg['MAE_Model']:.2f}° (Raw: {total_avg['MAE_Raw']:.2f}°)")
