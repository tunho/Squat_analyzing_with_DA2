import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/lee/exe_est")
AUG_DATA_PATH = PROJECT_ROOT / "outputs/ml_correction/v15_mega/universal_features_rehab_both_legs.csv"
FIT3D_DATA_PATH = PROJECT_ROOT / "outputs/ml_correction/v15_mega/features_fit3d_v15_both_legs.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v15_augmented_loso"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Standard features
FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio',
    'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z',
    'hip_depth_norm', 'hip_ankle_dist_norm',
    'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2',
    'mp_visibility'
]

# Mapping PM subjects (18 video IDs) to 9 unique Person IDs
PERSON_MAP = {
    'PM_008': 1, 
    'PM_021': 2, 'PM_022': 2, 
    'PM_028': 3, 'PM_029': 3, 
    'PM_037': 4, 'PM_038': 4, 
    'PM_042': 5, 'PM_043': 5, 
    'PM_104': 6, 'PM_105': 6, 
    'PM_125': 7, 'PM_126': 7,
    'PM_112': 8, 'PM_113': 8, 
    'PM_117a': 9, 'PM_117b': 9, 'PM_118': 9
}

def refine_bone_lengths(df):
    refined_dfs = []
    groups = df.groupby(['subject_id', 'ex_id', 'view_type', 'leg_side'])
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

def run_augmented_loso():
    if not AUG_DATA_PATH.exists():
        print(f"Error: {AUG_DATA_PATH} not found.")
        return
        
    df_aug = pd.read_csv(AUG_DATA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    df_aug['person_id'] = df_aug['subject_id'].map(PERSON_MAP)
    
    # Optionally: Augment Fit3D too?
    # For now, let's just use original Fit3D (already 57k frames) to speed up.
    # We will combine Rehab (Augmented) + Fit3D (Base)
    print("📖 Loading Fit3D Dataset...")
    df_fit = pd.read_csv(FIT3D_DATA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    
    # Map Fit3D Camera IDs
    cam_map = {50591643:'front', 58860488:'side', 65906101:'side'}
    df_fit['view_type'] = df_fit['view_type'].map(cam_map)
    df_fit = df_fit[df_fit['view_type'].notna()].copy()
    
    df_fit['person_id'] = 10 + df_fit['subject_id'].astype('category').cat.codes
    
    # Final combined dataset
    df = pd.concat([df_aug, df_fit], ignore_index=True)
    
    rehab_pids = sorted(df_aug['person_id'].unique())
    print(f"🚀 Augmented LOSO for {len(rehab_pids)} Rehab Subjects (2x frames per vid)")

    results = []
    for test_pid in rehab_pids:
        # --- NEW: Filter Training data for quality (Visibility > 0.5) ---
        train_df_full = df[df['person_id'] != test_pid]
        train_df = train_df_full[train_df_full['mp_visibility'] > 0.5].copy()
        
        test_df = df[df['person_id'] == test_pid].copy()
        
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(train_df[FEATURES], train_df['gt_angle'])
        
        test_df_refined = refine_bone_lengths(test_df)
        preds = model.predict(test_df_refined[FEATURES])
        test_df_refined['pred_angle'] = preds
        
        # --- OPTIMIZED: Visibility-based Selection per frame ---
        # Sort by visibility and keep only the top one for each (video, frame)
        test_df_best = test_df_refined.sort_values('mp_visibility', ascending=False).drop_duplicates(['subject_id', 'ex_id', 'view_type', 'frame_index'])
        test_df_best = test_df_best.sort_values(['subject_id', 'ex_id', 'view_type', 'frame_index'])
        
        # Sequence-based evaluation on the BEST leg only
        for (sid, ex_id, view), sub_group in test_df_best.groupby(['subject_id', 'ex_id', 'view_type']):
            p_seq = sub_group['pred_angle'].values
            raw_seq = sub_group['mp_knee_angle'].values
            gt_seq = sub_group['gt_angle'].values
            
            if len(p_seq) > 11: p_seq_smoothed = savgol_filter(p_seq, 11, 3)
            else: p_seq_smoothed = p_seq
            
            results.append({
                "person_id": test_pid, "video_id": sid, "view": view, "ex_id": ex_id, 
                "mae_model": mean_absolute_error(gt_seq, p_seq_smoothed),
                "mae_raw": mean_absolute_error(gt_seq, raw_seq)
            })
        print(f"  Person {test_pid} done.")

    res_df = pd.DataFrame(results)
    res_df.to_csv(SAVE_DIR / "res_augmented_loso.csv", index=False)
    
    print("\n--- 🔥 AUGMENTED LOSO Results ---")
    print(res_df.groupby('view')[['mae_raw', 'mae_model']].mean())

if __name__ == "__main__":
    run_augmented_loso()
