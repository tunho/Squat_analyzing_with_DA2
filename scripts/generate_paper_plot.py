import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from pathlib import Path

# Config
PROJECT_ROOT = Path("/home/lee/exe_est")
DATA_PATH = PROJECT_ROOT / "outputs/ml_correction/v15_mega/ultimate_features_all.csv"
ARTIFACT_DIR = Path("/home/lee/.gemini/antigravity/brain/39a1a6e9-f4b7-4441-8548-50413c11fba7")
SAVE_PATH = ARTIFACT_DIR / "fig1_squat_correction.png"

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
    'PM_117a': 9, 'PM_117b': 9, 'PM_118': 9
}

def generate_plot():
    print("🎨 Generating high-quality paper plot...")
    
    # 1. Load Data (Rehab Only, Squat Only)
    df_raw = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    df_raw['person_id'] = df_raw['subject_id'].map(PERSON_MAP)
    df = df_raw[(df_raw['person_id'] <= 9) & (df_raw['ex_id'] == 6)].copy() # Squat
    df['residual'] = df['gt_angle'] - df['mp_knee_angle']
    
    # Target: Subject PM_043 (Person 5), View: Side
    test_pid = 5 
    target_subject = 'PM_043'
    target_view = 'side'
    
    # 2. Train Expert Model (LOSO split)
    train_df = df[df['person_id'] != test_pid].copy()
    test_df_full = df[df['person_id'] == test_pid].copy()
    
    test_df = test_df_full.sort_values('mp_visibility', ascending=False).drop_duplicates(['subject_id', 'view', 'frame_index'])
    target_test_df = test_df[test_df['view'] == target_view].copy()
    target_test_df = target_test_df.sort_values('frame_index')
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURES])
    X_test = scaler.transform(target_test_df[FEATURES])
    
    model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, train_df['residual'])
    
    target_test_df['residual_pred'] = model.predict(X_test)
    pred_angle = target_test_df['mp_knee_angle'] + target_test_df['residual_pred']
    target_test_df['pred_angle'] = savgol_filter(pred_angle, 11, 3) # Smoothing
    
    # 3. Plotting (Academic Theme)
    frames = target_test_df['frame_index'].values
    gt = target_test_df['gt_angle'].values
    raw = target_test_df['mp_knee_angle'].values
    pred = target_test_df['pred_angle'].values
    
    # MAE calculation for title
    mae_raw = np.abs(gt - raw).mean()
    mae_pred = np.abs(gt - pred).mean()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    ax.plot(frames, gt, label=f'Ground Truth (GT)', color='black', linewidth=3, linestyle='-', zorder=3)
    ax.plot(frames, raw, label=f'Raw MediaPipe (MAE: {mae_raw:.2f}°)', color='#FF5A5F', linewidth=2, linestyle='--', alpha=0.8, zorder=1)
    ax.plot(frames, pred, label=f'Proposed Expert Model (MAE: {mae_pred:.2f}°)', color='#007AFF', linewidth=2.5, linestyle='-', zorder=2)
    
    # Fill between to show error visually
    ax.fill_between(frames, gt, pred, color='#007AFF', alpha=0.1)
    ax.fill_between(frames, gt, raw, color='#FF5A5F', alpha=0.05)
    
    ax.set_title("Figure 1. Knee Angle Correction Trajectory (Squat - Side View)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Frame Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("Knee Angle (Degrees)", fontsize=12, fontweight='bold')
    
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved plot to {SAVE_PATH}")

if __name__ == "__main__":
    generate_plot()
