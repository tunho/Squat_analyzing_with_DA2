import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/home/lee/exe_est")
FIT3D_FEATURES = PROJECT_ROOT / "outputs/ml_correction/v13_mega/features_fit3d_v13_all.csv"
ORIGINAL_FEATURES = PROJECT_ROOT / "outputs/ml_correction/v7_universal/universal_features_complete.csv"
OUTPUT_MEGA = PROJECT_ROOT / "outputs/ml_correction/v13_mega/universal_features_v13_mega.csv"

def merge():
    if not FIT3D_FEATURES.exists():
        print("Fit3D features not found yet.")
        return

    print("📖 Loading Original Dataset...")
    df_orig = pd.read_csv(ORIGINAL_FEATURES)
    # Filter for Ex5 (Lunge), Ex6 (Squat)
    df_orig = df_orig[df_orig['ex_id'].isin([5, 6])]
    print(f"Original frames: {len(df_orig)}")

    print("📖 Loading Fit3D Dataset...")
    df_fit3d = pd.read_csv(FIT3D_FEATURES)
    
    # Map Fit3D Camera IDs to View Types
    # 50591643: Front, 58860488: Side_R, 60457274: Back, 65906101: Side_L (Approx)
    cam_map = {
        "50591643": "front",
        "58860488": "side",
        "60457274": "back",
        "65906101": "side"
    }
    df_fit3d['view_type'] = df_fit3d['view_type'].astype(str).map(cam_map)
    print(f"Fit3D frames: {len(df_fit3d)}")

    # Align columns
    cols = [
        'subject_id', 'view_type', 'ex_id', 'frame_index',
        'mp_knee_angle', 'mp_hip_angle', 'leg_ratio',
        'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z',
        'hip_depth_norm', 'hip_ankle_dist_norm',
        'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2',
        'gt_angle'
    ]
    
    df_orig = df_orig[cols]
    df_fit3d = df_fit3d[cols]

    # Combine
    df_mega = pd.concat([df_orig, df_fit3d], ignore_index=True)
    print(f"✅ Total Mega Dataset frames: {len(df_mega)}")
    print(f"Subjects: {df_mega['subject_id'].nunique()}")

    df_mega.to_csv(OUTPUT_MEGA, index=False)
    print(f"✨ Saved to: {OUTPUT_MEGA}")

if __name__ == "__main__":
    merge()
