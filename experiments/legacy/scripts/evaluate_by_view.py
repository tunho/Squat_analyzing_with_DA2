import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path("/home/lee/exe_est")
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"
MODEL_PATH = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final/paper_correction_model_v5.pkl"

def prepare_v5_data(df):
    df = df.sort_values(['subject_id', 'view_type', 'frame_index'])
    df['femur_len'] = np.sqrt(df['k_x']**2 + df['k_y']**2 + df['k_z']**2)
    df['tibia_len'] = np.sqrt((df['a_x']-df['k_x'])**2 + (df['a_y']-df['k_y'])**2 + (df['a_z']-df['k_z'])**2)
    subj_ratio = df.groupby('subject_id')[['femur_len', 'tibia_len']].mean()
    subj_ratio['leg_ratio'] = subj_ratio['femur_len'] / (subj_ratio['tibia_len'] + 1e-6)
    df['leg_ratio'] = df['subject_id'].map(subj_ratio['leg_ratio'])
    df['angle_velocity'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].diff().fillna(0)
    df['angle_vel_smooth'] = df.groupby(['subject_id', 'view_type'])['angle_velocity'].transform(lambda x: x.rolling(5, center=True).mean()).fillna(0)
    df['knee_lag1'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(1)
    df['knee_lag2'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(2)
    return df.dropna(subset=['knee_lag1', 'knee_lag2'])

FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle", "leg_ratio", "angle_vel_smooth",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis", "hip_depth_norm", "hip_ankle_dist_norm",
    "knee_lag1", "knee_lag2"
]

def main():
    df = pd.read_csv(INPUT_CSV)
    df = prepare_v5_data(df)
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model, scaler = bundle["model"], bundle["scaler"]
    
    X_s = scaler.transform(df[FEATURE_COLS])
    df['pred_residual'] = model.predict(X_s)
    df['corrected_angle'] = df['mp_knee_angle'] + df['pred_residual']
    
    print("\n--- VIEW TYPE BREAKDOWN (v5 Paper Model) ---")
    for vt in ['front', 'side']:
        v_df = df[df['view_type'] == vt]
        mae_raw = mean_absolute_error(v_df['gt_angle'], v_df['mp_knee_angle'])
        mae_corr = mean_absolute_error(v_df['gt_angle'], v_df['corrected_angle'])
        print(f"[{vt.upper()}] Raw: {mae_raw:.2f}° -> Corrected: {mae_corr:.2f}° (Improvement: {mae_raw-mae_corr:.2f}°)")

if __name__ == "__main__":
    main()
