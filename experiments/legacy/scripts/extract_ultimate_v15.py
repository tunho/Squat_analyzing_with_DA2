import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import json
import sys
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path("/home/lee/exe_est")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squat.pose_estimation import PoseEstimator
from squat.pipeline.frame_processor import _extract_world_landmarks
from squat.geometry.angles import calculate_knee_angle
from squat.domain.types import Point3D

# --- CONFIG ---
EX_LIST = [5, 6] # Lunge, Squat
OUTPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/v15_mega/ultimate_features_all.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

REHAB_IDS = ['PM_008', 'PM_021', 'PM_022', 'PM_028', 'PM_029', 'PM_037', 'PM_038', 'PM_042', 'PM_043', 'PM_104', 'PM_105', 'PM_125', 'PM_126', 'PM_112', 'PM_113', 'PM_117a', 'PM_117b', 'PM_118']
FIT3D_IDS = ['s03', 's04', 's05']

def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_all_ultimate():
    pose_estimator = PoseEstimator(static_image_mode=True)
    all_rows = []

    # 1. Process Rehab
    print("🎬 Processing Rehab for Ultimate Benchmark...")
    for sub_id in REHAB_IDS:
        for ex in EX_LIST:
            v_dir = PROJECT_ROOT / f"dataset/raw/REHAB24-6/videos/Ex{ex}"
            gt_dir = PROJECT_ROOT / f"dataset/raw/REHAB24-6/3d_joints/Ex{ex}"
            
            # Rehab has Camera17 (Front) and Camera18 (Side)
            for cam in ["Camera17", "Camera18"]:
                v_path = next(v_dir.glob(f"{sub_id}-{cam}*.mp4"), None)
                if not v_path: continue
                
                gt_path = gt_dir / f"{sub_id}-30fps.npy"
                if not gt_path.exists(): continue
                gt_arr = np.load(gt_path)
                
                # Leg ID Mapping (Simplified: Left 16,17,18 / Right 21,22,23)
                leg_map = {"left": (16, 17, 18), "right": (21, 22, 23)}
                mp_map = {"left": (23, 25, 27, 11), "right": (24, 26, 28, 12)} # Hip, Knee, Ankle, Shoulder
                
                cap = cv2.VideoCapture(str(v_path))
                f_idx = 0
                temp_frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or f_idx >= len(gt_arr): break
                    extracted = pose_estimator.extract_keypoints_only(frame)
                    _, res = (extracted if isinstance(extracted, tuple) else (extracted, None))
                    world_lms = _extract_world_landmarks(res)
                    if world_lms and len(world_lms) > 28:
                        temp_frames.append({"f_idx": f_idx, "world_lms": world_lms})
                    f_idx += 1
                cap.release()
                
                if not temp_frames: continue
                
                for side, (g_h, g_k, g_a) in leg_map.items():
                    m_h, m_k, m_a, m_s = mp_map[side]
                    flip_x = -1.0 if side == "right" else 1.0
                    
                    leg_rows = []
                    thigh_lens, shank_lens = [], []
                    for f in temp_frames:
                        w = f["world_lms"]
                        h = [w[m_h]['x'], w[m_h]['y'], w[m_h]['z']]
                        k = [w[m_k]['x'], w[m_k]['y'], w[m_k]['z']]
                        a = [w[m_a]['x'], w[m_a]['y'], w[m_a]['z']]
                        thigh_lens.append(np.linalg.norm(np.array(h)-np.array(k)))
                        shank_lens.append(np.linalg.norm(np.array(a)-np.array(k)))
                    
                    med_total = np.median(thigh_lens) + np.median(shank_lens) + 1e-6
                    leg_ratio = np.median(thigh_lens) / (np.median(shank_lens) + 1e-6)
                    
                    for f in temp_frames:
                        w = f["world_lms"]
                        h = [w[m_h]['x'], w[m_h]['y'], w[m_h]['z']]
                        k = [w[m_k]['x'], w[m_k]['y'], w[m_k]['z']]
                        a = [w[m_a]['x'], w[m_a]['y'], w[m_a]['z']]
                        s = [w[m_s]['x'], w[m_s]['y'], w[m_s]['z']]
                        
                        gt_h, gt_k, gt_a = gt_arr[f['f_idx'], g_h, :3], gt_arr[f['f_idx'], g_k, :3], gt_arr[f['f_idx'], g_a, :3]
                        
                        leg_rows.append({
                            "subject_id": sub_id, "view": "front" if "Camera17" in cam else "side", "ex_id": ex,
                            "frame_index": f['f_idx'], "leg_side": side,
                            "mp_knee_angle": get_angle(h, k, a),
                            "gt_angle": get_angle(gt_h, gt_k, gt_a),
                            "mp_hip_angle": get_angle(s, h, k),
                            "mp_visibility": (w[m_h]['visibility'] + w[m_k]['visibility'] + w[m_a]['visibility'] + w[m_s]['visibility']) / 4.0,
                            "leg_ratio": leg_ratio,
                            "k_x": (k[0]-h[0])*flip_x/med_total, "k_y": (k[1]-h[1])/med_total, "k_z": (k[2]-h[2])/med_total,
                            "a_x": (a[0]-h[0])*flip_x/med_total, "a_y": (a[1]-h[1])/med_total, "a_z": (a[2]-h[2])/med_total,
                            "s_x": (s[0]-h[0])*flip_x/med_total, "s_y": (s[1]-h[1])/med_total, "s_z": (s[2]-h[2])/med_total,
                            "hip_y": h[1], "hip_ankle_dist_norm": np.linalg.norm(np.array(h)-np.array(a))/med_total
                        })
                    
                    if leg_rows:
                        df_l = pd.DataFrame(leg_rows)
                        df_l['angle_velocity'] = df_l['mp_knee_angle'].diff().fillna(0)
                        df_l['angle_vel_smooth'] = df_l['angle_velocity'].rolling(5, center=True).mean().fillna(0)
                        df_l['knee_lag1'] = df_l['mp_knee_angle'].shift(1).ffill().bfill()
                        df_l['knee_lag2'] = df_l['mp_knee_angle'].shift(2).ffill().bfill()
                        all_rows.append(df_l)
                        pd.concat(all_rows).to_csv(OUTPUT_CSV, index=False)
                        print(f"    - Sub {sub_id} Ex{ex} {cam} {side} done.")

    # 2. Process Fit3D同样...
    print("🎬 Processing Fit3D for Ultimate Benchmark...")
    # (Similarity logic follows here for Fit3D)
    # Mapping Fit3D exercises: squat (6), dumbbell_reverse_lunge (5)
    FIT3S_EX = {"squat": 6, "dumbbell_reverse_lunge": 5}
    FIT3D_CAMS = {50591643: "front", 58860488: "side", 65906101: "side"}
    FIT3D_LEG_GT = {"left": (4, 5, 6), "right": (1, 2, 3)}

    for sid in FIT3D_IDS:
        s_dir = PROJECT_ROOT / f"dataset/raw/fit3d/fit3d_train/train/{sid}"
        for ex_name, ex_id in FIT3S_EX.items():
            json_path = s_dir / f"joints3d_25/{ex_name}.json"
            if not json_path.exists(): continue
            with open(json_path, 'r') as f: gt_frames = json.load(f)['joints3d_25']
            
            for cam_id, view_name in FIT3D_CAMS.items():
                v_path = s_dir / f"videos/{cam_id}/{ex_name}.mp4"
                if not v_path.exists(): continue
                
                cap = cv2.VideoCapture(str(v_path))
                f_idx = 0
                temp_frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or f_idx >= len(gt_frames): break
                    extracted = pose_estimator.extract_keypoints_only(frame)
                    _, res = (extracted if isinstance(extracted, tuple) else (extracted, None))
                    world_lms = _extract_world_landmarks(res)
                    if world_lms and len(world_lms) > 28:
                        temp_frames.append({"f_idx": f_idx, "world_lms": world_lms})
                    f_idx += 1
                cap.release()
                
                for side, (g_h, g_k, g_a) in FIT3D_LEG_GT.items():
                    m_h, m_k, m_a, m_s = (23, 25, 27, 11) if side == "left" else (24, 26, 28, 12)
                    flip_x = -1.0 if side == "right" else 1.0
                    leg_rows = []
                    thigh_lens, shank_lens = [], []
                    for f in temp_frames:
                        w = f["world_lms"]; h = [w[m_h]['x'], w[m_h]['y'], w[m_h]['z']]; k = [w[m_k]['x'], w[m_k]['y'], w[m_k]['z']]
                        thigh_lens.append(np.linalg.norm(np.array(h)-np.array(k)))
                        shank_lens.append(np.linalg.norm([w[m_a]['x'], w[m_a]['y'], w[m_a]['z']]-np.array(k)))
                    
                    med_total = np.median(thigh_lens) + np.median(shank_lens) + 1e-6
                    leg_ratio = np.median(thigh_lens) / (np.median(shank_lens) + 1e-6)
                    
                    for f in temp_frames:
                        w = f["world_lms"]
                        h = [w[m_h]['x'], w[m_h]['y'], w[m_h]['z']]; k = [w[m_k]['x'], w[m_k]['y'], w[m_k]['z']]; a = [w[m_a]['x'], w[m_a]['y'], w[m_a]['z']]; s = [w[m_s]['x'], w[m_s]['y'], w[m_s]['z']]
                        gt_h, gt_k, gt_a = gt_frames[f['f_idx']][g_h], gt_frames[f['f_idx']][g_k], gt_frames[f['f_idx']][g_a]
                        
                        leg_rows.append({
                            "subject_id": f"fit3d_{sid}", "view": view_name, "ex_id": ex_id,
                            "frame_index": f['f_idx'], "leg_side": side,
                            "mp_knee_angle": get_angle(h, k, a), "gt_angle": get_angle(gt_h, gt_k, gt_a),
                            "mp_hip_angle": get_angle(s, h, k),
                            "mp_visibility": (w[m_h]['visibility'] + w[m_k]['visibility'] + w[m_a]['visibility'] + w[m_s]['visibility']) / 4.0,
                            "leg_ratio": leg_ratio,
                            "k_x": (k[0]-h[0])*flip_x/med_total, "k_y": (k[1]-h[1])/med_total, "k_z": (k[2]-h[2])/med_total,
                            "a_x": (a[0]-h[0])*flip_x/med_total, "a_y": (a[1]-h[1])/med_total, "a_z": (a[2]-h[2])/med_total,
                            "s_x": (s[0]-h[0])*flip_x/med_total, "s_y": (s[1]-h[1])/med_total, "s_z": (s[2]-h[2])/med_total,
                            "hip_y": h[1], "hip_ankle_dist_norm": np.linalg.norm(np.array(h)-np.array(a))/med_total
                        })
                    if leg_rows:
                        df_l = pd.DataFrame(leg_rows); df_l['angle_velocity'] = df_l['mp_knee_angle'].diff().fillna(0)
                        df_l['angle_vel_smooth'] = df_l['angle_velocity'].rolling(5, center=True).mean().fillna(0)
                        df_l['knee_lag1'] = df_l['mp_knee_angle'].shift(1).ffill().bfill()
                        df_l['knee_lag2'] = df_l['mp_knee_angle'].shift(2).ffill().bfill()
                        all_rows.append(df_l)
                        pd.concat(all_rows).to_csv(OUTPUT_CSV, index=False)
                        print(f"    - Sub fit3d_{sid} Ex{ex_id} {cam_id} {side} done.")

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✨ ALL DATA EXTRACTED TO {OUTPUT_CSV}")
    pose_estimator.close()

if __name__ == "__main__":
    extract_all_ultimate()
