import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/home/lee/exe_est")
FIT3D_FEATURES = PROJECT_ROOT / "outputs/ml_correction/v13_mega/features_fit3d_v13_all.csv"
REHAB_FEATURES = PROJECT_ROOT / "outputs/ml_correction/v14_mega/universal_features_rehab_all.csv"
OUTPUT_MEGA = PROJECT_ROOT / "outputs/ml_correction/v14_mega/universal_features_v14_mega.csv"

def merge():
    if not FIT3D_FEATURES.exists():
        print("Fit3D features not found.")
        return
    if not REHAB_FEATURES.exists():
        print("Rehab features not found.")
        return

    print("📖 Loading Rehab Dataset...")
    df_rehab = pd.read_csv(REHAB_FEATURES)
    # Ex1, Ex5, Ex6 are most relevant for knee.
    # Ex4 was also in the list, but let's see. 
    # User specifically said "Squat도 추가해서"
    df_rehab = df_rehab[df_rehab['ex_id'].isin([1, 5, 6])]
    print(f"Rehab frames: {len(df_rehab)}")

    print("📖 Loading Fit3D Dataset...")
    df_fit3d = pd.read_csv(FIT3D_FEATURES)
    
    # Fit3D ex_id: 5 (Lunge), 6 (Squat)
    # Rehab ex_id: 1 (Squat), 5 (Lunge), 6 (?)
    # Map them to a consistent set if possible, but LOSO works fine with different IDs.
    # However, let's map Rehab Ex1 to 6 to align with Fit3D Squat if appropriate.
    # No, it's safer to just keep them as they are and use ex_id as a feature.
    
    cam_map = {
        "50591643": "front",
        "58860488": "side",
        "60457274": "back",
        "65906101": "side"
    }
    df_fit3d['view_type'] = df_fit3d['view_type'].astype(str).map(cam_map)
    print(f"Fit3D frames: {len(df_fit3d)}")

    # Combine
    df_mega = pd.concat([df_rehab, df_fit3d], ignore_index=True)
    
    # Filter for knee flexion as per previous feedback
    # Rehab Ex6 might be abduction, check description. 
    # But for now, let's include all 1, 5, 6 from Rehab and 5, 6 from Fit3D.
    
    print(f"✅ Total v14 Mega Dataset frames: {len(df_mega)}")
    print(f"Subjects: {df_mega['subject_id'].nunique()}")

    df_mega.to_csv(OUTPUT_MEGA, index=False)
    print(f"✨ Saved to: {OUTPUT_MEGA}")

if __name__ == "__main__":
    merge()
