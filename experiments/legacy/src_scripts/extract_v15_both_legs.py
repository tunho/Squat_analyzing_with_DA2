import sys
import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path

PROJECT_ROOT = Path("/home/lee/exe_est")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import point_to_np
from squat.pose_estimation import PoseEstimator
from squat.pipeline.frame_processor import _extract_world_landmarks
from squat.domain.types import Point3D

EX_LIST = ["Ex5", "Ex6"]
OUTPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/v15_mega/universal_features_rehab_both_legs.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
SEGMENT_CSV = PROJECT_ROOT / "dataset/raw/REHAB24-6/Segmentation.csv"

def get_gt_indices(leg_side):
    # Left: 16, 17, 18; Right: 21, 22, 23
    if leg_side == "left":
        return 16, 17, 18
    return 21, 22, 23

def get_mp_indices(leg_side):
    # Left: 23, 25, 27; Right: 24, 26, 28
    if leg_side == "left":
        return 23, 25, 27
    return 24, 26, 28

def extract_features_per_video_both_legs(pe, video_path, ex_type, sid, gt_arr):
    cap = cv2.VideoCapture(str(video_path))
    
    view_type = "side" if "Camera18" in video_path.name else "front"
    ex_id = int(ex_type.replace("Ex", ""))
    
    frames_raw = []
    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or f_idx >= len(gt_arr): break
        extracted = pe.extract_keypoints_only(frame)
        landmarks, res_obj = (extracted if isinstance(extracted, tuple) else (extracted, None))
        if landmarks:
            world_lms = _extract_world_landmarks(res_obj)
            if world_lms and len(world_lms) > 30:
                frames_raw.append({"f_idx": f_idx, "world_lms": world_lms})
        f_idx += 1
    cap.release()

    if not frames_raw: return pd.DataFrame()

    all_leg_rows = []
    
    # Process BOTH legs
    for leg_side in ["left", "right"]:
        m_h, m_k, m_a = get_mp_indices(leg_side)
        g_h, g_k, g_a = get_gt_indices(leg_side)
        
        # Calculate leg-specific median lengths for normalization
        thigh_lens, shank_lens = [], []
        leg_frames = []
        
        for f in frames_raw:
            w = f["world_lms"]
            h = np.array([w[m_h]['x'], w[m_h]['y'], w[m_h]['z']])
            k = np.array([w[m_k]['x'], w[m_k]['y'], w[m_k]['z']])
            a = np.array([w[m_a]['x'], w[m_a]['y'], w[m_a]['z']])
            thigh_lens.append(np.linalg.norm(h - k))
            shank_lens.append(np.linalg.norm(a - k))
            leg_frames.append({"f_idx": f["f_idx"], "h": h, "k": k, "a": a})
            
        med_total = np.median(thigh_lens) + np.median(shank_lens) if thigh_lens else 1.0
        leg_ratio = np.median(thigh_lens) / (np.median(shank_lens) + 1e-6) if thigh_lens else 1.0
        
        # Vertical range (for hip depth norm)
        hips_y = [f["h"][1] for f in leg_frames]
        y_min, y_max = min(hips_y), max(hips_y)
        y_range = max(y_max - y_min, 1e-6)
        
        flip_x = -1.0 if leg_side == "right" else 1.0
        
        def ang(a,b,c):
            ba, bc = a-b, c-b
            return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8), -1, 1)))

        rows_for_leg = [] # Renamed to avoid conflict with the 'rows' variable in the original snippet
        for f in leg_frames:
            w = f["world_lms"] # Get world_lms from the frame
            
            m_h, m_k, m_a = (23, 25, 27) if leg_side == "left" else (24, 26, 28)
            m_s = 11 if leg_side == "left" else 12 # Shoulder
            
            h = np.array([w[m_h]['x'], w[m_h]['y'], w[m_h]['z']])
            k = np.array([w[m_k]['x'], w[m_k]['y'], w[m_k]['z']])
            a = np.array([w[m_a]['x'], w[m_a]['y'], w[m_a]['z']])
            s = np.array([w[m_s]['x'], w[m_s]['y'], w[m_s]['z']])
            
            gt_h = gt_arr[f["f_idx"], g_h, :3]
            gt_k = gt_arr[f["f_idx"], g_k, :3]
            gt_a = gt_arr[f["f_idx"], g_a, :3]
            
            med_total_float = float(med_total) # Ensure med_total is float for division
            
            rows_for_leg.append({
                "subject_id": sid, "view_type": view_type, "ex_id": ex_id,
                "frame_index": f['f_idx'],
                "leg_side": leg_side,
                "mp_knee_angle": ang(h, k, a),
                "mp_hip_angle": ang(s, h, k), 
                "mp_visibility": (w[m_h]['visibility'] + w[m_k]['visibility'] + w[m_a]['visibility'] + w[m_s]['visibility']) / 4.0,
                "leg_ratio": leg_ratio,
                "k_x": (k[0]-h[0])*flip_x/med_total_float, 
                "k_y": (k[1]-h[1])/med_total_float, 
                "k_z": (k[2]-h[2])/med_total_float,
                "a_x": (a[0]-h[0])*flip_x/med_total_float, 
                "a_y": (a[1]-h[1])/med_total_float, 
                "a_z": (a[2]-h[2])/med_total_float,
                "s_x": (s[0]-h[0])*flip_x/med_total_float,
                "s_y": (s[1]-h[1])/med_total_float,
                "s_z": (s[2]-h[2])/med_total_float,
                "hip_depth_norm": (y_max - h[1]) / y_range, # Keep original hip_depth_norm
                "hip_y": h[1],
                "hip_ankle_dist_norm": np.linalg.norm(h-a)/med_total_float, # Keep original hip_ankle_dist_norm
                "hip_ankle_dist": np.linalg.norm(h-a),
                "gt_angle": ang(gt_h, gt_k, gt_a) # Use gt_h, gt_k, gt_a directly
            })
        all_leg_rows.extend(rows_for_leg) # Use extend to add all rows from this leg
            
    df = pd.DataFrame(all_leg_rows)
    # Add velocities and lags per leg
    final_dfs = []
    for leg in ["left", "right"]:
        sub = df[df['leg_side'] == leg].copy().sort_values("frame_index")
        if sub.empty: continue
        sub['angle_velocity'] = sub['mp_knee_angle'].diff().fillna(0)
        sub['angle_vel_smooth'] = sub['angle_velocity'].rolling(5, center=True).mean().fillna(0)
        sub['knee_lag1'] = sub['mp_knee_angle'].shift(1).fillna(method='bfill')
        sub['knee_lag2'] = sub['mp_knee_angle'].shift(2).fillna(method='bfill')
        final_dfs.append(sub)
        
    return pd.concat(final_dfs)

def run_augmentation_extraction():
    all_dfs = []
    print("🚀 Symmetric Data Augmentation (L/R Legs) Extraction Starting...")
    
    pe = PoseEstimator(static_image_mode=True)
    try:
        TARGET_IDS = ['PM_008', 'PM_021', 'PM_022', 'PM_028', 'PM_029', 'PM_037', 'PM_038', 'PM_042', 'PM_043', 'PM_104', 'PM_105', 'PM_125', 'PM_126', 'PM_112', 'PM_113', 'PM_117a', 'PM_117b', 'PM_118']
        
        for ex_type in EX_LIST:
            v_dir = PROJECT_ROOT / f"dataset/raw/REHAB24-6/videos/{ex_type}"
            gt_dir = PROJECT_ROOT / f"dataset/raw/REHAB24-6/3d_joints/{ex_type}"
            
            for v_path in sorted(list(v_dir.glob("*.mp4"))):
                sid = v_path.name.split('-')[0]
                if sid not in TARGET_IDS: continue
                
                gt_path = gt_dir / f"{sid}-30fps.npy"
                if not gt_path.exists(): continue
                
                # Check mapping for 9 people later, but extract all for now
                gt_arr = np.load(gt_path)
                
                print(f"  🎬 Processing {ex_type} | {v_path.name} (Both Legs)")
                df_video = extract_features_per_video_both_legs(pe, v_path, ex_type, sid, gt_arr)
                if not df_video.empty:
                    all_dfs.append(df_video)
                    pd.concat(all_dfs).to_csv(OUTPUT_CSV, index=False)
    finally:
        pe.close()
    
    print(f"\n✨ Extraction Complete! -> {OUTPUT_CSV}")

if __name__ == "__main__":
    run_augmentation_extraction()
