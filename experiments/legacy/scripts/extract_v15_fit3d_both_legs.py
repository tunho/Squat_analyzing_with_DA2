import os
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import sys

# 프로젝트 루트 및 소스 경로 설정
PROJECT_ROOT = Path("/home/lee/exe_est")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squat.pose_estimation import PoseEstimator
from squat.pipeline.frame_processor import _extract_world_landmarks

def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# 설정
SUBJECTS = ["s03", "s04", "s05"]
CAMS = ["50591643", "58860488", "65906101"] # Front, Side_R, Side_L (Excluding Back)
EX_MAP = {"squat": 6, "dumbbell_reverse_lunge": 5, "deadlift": 7}
OUTPUT_FILE = PROJECT_ROOT / "outputs/ml_correction/v15_mega/features_fit3d_v15_both_legs.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def find_sub_dir(sub):
    d1 = PROJECT_ROOT / f"dataset/raw/fit3d/fit3d_train/train/{sub}"
    if d1.exists(): return d1
    return None

def process_fit3d_both_legs():
    all_rows = []
    pose_estimator = PoseEstimator(static_image_mode=True)
    
    # GT Joints mapping (Approximate for MediaPipe alignment)
    # Fit3D indices: 5 Hip, 6 Knee, 7 Ankle? (Need to verify)
    # Actually fit3d_joints indices (H3.6M format usually):
    # Left: 4 Hip, 5 Knee, 6 Ankle
    # Right: 1 Hip, 2 Knee, 3 Ankle
    LEG_GT_MAP = {
        "left": (4, 5, 6),
        "right": (1, 2, 3)
    }
    LEG_MP_MAP = {
        "left": (23, 25, 27), # Hip, Knee, Ankle
        "right": (24, 26, 28)
    }

    for sub in SUBJECTS:
        sub_dir = find_sub_dir(sub)
        if not sub_dir: continue
            
        for ex_name, ex_id in EX_MAP.items():
            json_path = sub_dir / f"joints3d_25/{ex_name}.json"
            if not json_path.exists(): continue
            
            with open(json_path, 'r') as f:
                gt_frames = json.load(f)['joints3d_25']

            for cam_id in CAMS:
                video_path = sub_dir / f"videos/{cam_id}/{ex_name}.mp4"
                if not video_path.exists(): continue

                print(f"🚀 Processing Fit3D {sub} - {ex_name} - {cam_id} (Both Legs)")
                cap = cv2.VideoCapture(str(video_path))
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
                
                if not temp_frames: continue

                # Post-process per leg
                for side in ["left", "right"]:
                    m_hi, m_ki, m_ai = LEG_MP_MAP[side]
                    g_hi, g_ki, g_ai = LEG_GT_MAP[side]
                    flip_x = -1.0 if side == "right" else 1.0
                    
                    leg_rows = []
                    thigh_lens, shank_lens = [], []
                    
                    for f in temp_frames:
                        w = f["world_lms"]
                        h = [w[m_hi]['x'], w[m_hi]['y'], w[m_hi]['z']]
                        k = [w[m_ki]['x'], w[m_ki]['y'], w[m_ki]['z']]
                        a = [w[m_ai]['x'], w[m_ai]['y'], w[m_ai]['z']]
                        
                        thigh_lens.append(np.linalg.norm(np.array(k)-np.array(h)))
                        shank_lens.append(np.linalg.norm(np.array(a)-np.array(k)))
                        
                        mp_knee = get_angle(h, k, a)
                        gt_knee = get_angle(gt_frames[f['f_idx']][g_hi], gt_frames[f['f_idx']][g_ki], gt_frames[f['f_idx']][g_ai])
                        
                        leg_rows.append({
                            "subject_id": f"fit3d_{sub}", "ex_id": ex_id, "view_type": cam_id, "frame_index": f['f_idx'],
                            "leg_side": side, "mp_knee_angle": mp_knee, "gt_angle": gt_knee,
                            "mp_visibility": (w[m_hi]['visibility'] + w[m_ki]['visibility'] + w[m_ai]['visibility']) / 3.0,
                            "k_x": (k[0]-h[0])*flip_x, "k_y": k[1]-h[1], "k_z": k[2]-h[2],
                            "a_x": (a[0]-h[0])*flip_x, "a_y": a[1]-h[1], "a_z": a[2]-h[2],
                            "hip_y": h[1], "hip_ankle_dist": np.linalg.norm(np.array(h)-np.array(a)),
                            "mp_hip_angle": 180.0 # Placeholder
                        })
                        
                    if leg_rows:
                        df_v = pd.DataFrame(leg_rows)
                        med_total = np.median(thigh_lens) + np.median(shank_lens)
                        df_v['leg_ratio'] = np.median(thigh_lens) / (np.median(shank_lens) + 1e-6)
                        for c in ['k_x','k_y','k_z','a_x','a_y','a_z']: df_v[c] /= med_total
                        y_min, y_max = df_v['hip_y'].min(), df_v['hip_y'].max()
                        df_v['hip_depth_norm'] = (y_max - df_v['hip_y']) / max(y_max - y_min, 1e-6)
                        df_v['hip_ankle_dist_norm'] = df_v['hip_ankle_dist'] / med_total
                        df_v['angle_velocity'] = df_v['mp_knee_angle'].diff().fillna(0)
                        df_v['angle_vel_smooth'] = df_v['angle_velocity'].rolling(5, center=True).mean().fillna(0)
                        df_v['knee_lag1'] = df_v['mp_knee_angle'].shift(1).ffill().bfill()
                        df_v['knee_lag2'] = df_v['mp_knee_angle'].shift(2).ffill().bfill()
                        all_rows.append(df_v)
                        pd.concat(all_rows).to_csv(OUTPUT_FILE, index=False)

    pose_estimator.close()
    print(f"✨ Fit3D Augmented Extraction Complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_fit3d_both_legs()
