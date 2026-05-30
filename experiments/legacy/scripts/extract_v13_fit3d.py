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
SUBJECTS = ["s03", "s04", "s05", "s02", "s12", "s13"]
CAMS = ["50591643", "58860488", "60457274", "65906101"]
EX_MAP = {"squat": 6, "dumbbell_reverse_lunge": 5}
OUTPUT_FILE = PROJECT_ROOT / "outputs/ml_correction/v13_mega/features_fit3d_v13_all.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def find_sub_dir(sub):
    d1 = PROJECT_ROOT / f"dataset/raw/fit3d/fit3d_train/train/{sub}"
    if d1.exists(): return d1
    d2 = PROJECT_ROOT / f"dataset/raw/fit3d/fit3d_test/test/{sub}"
    if d2.exists(): return d2
    return None

def process_fit3d():
    all_rows = []
    existing_subs_vids = set()
    if OUTPUT_FILE.exists():
        try:
            df_ex = pd.read_csv(OUTPUT_FILE)
            for _, r in df_ex.iterrows():
                existing_subs_vids.add((r['subject_id'], r['ex_id'], r['view_type']))
            all_rows.append(df_ex)
            print(f"📄 Found existing data with {len(df_ex)} frames.")
        except: pass

    pose_estimator = PoseEstimator(static_image_mode=False)
    
    for sub in SUBJECTS:
        sub_dir = find_sub_dir(sub)
        if not sub_dir: continue
            
        for ex_name, ex_id in EX_MAP.items():
            json_path = sub_dir / f"joints3d_25/{ex_name}.json"
            if not json_path.exists(): continue
            
            with open(json_path, 'r') as f:
                gt_frames = json.load(f)['joints3d_25']

            # Determine video paths
            v_paths = []
            for cam in CAMS:
                p = sub_dir / f"videos/{cam}/{ex_name}.mp4"
                if p.exists(): v_paths.append((p, cam))
            
            if not v_paths:
                p = sub_dir / f"videos/{ex_name}.mp4"
                if p.exists(): v_paths.append((p, "default"))

            for video_path, cam_id in v_paths:
                fid = f"fit3d_{sub}"
                if (fid, ex_id, cam_id) in existing_subs_vids:
                    print(f"⏩ Skipping {sub} - {ex_name} - {cam_id}")
                    continue

                print(f"🚀 Processing {sub} - {ex_name} - {cam_id}")
                cap = cv2.VideoCapture(str(video_path))
                f_idx = 0
                temp_rows = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or f_idx >= len(gt_frames): break
                    
                    extracted = pose_estimator.extract_keypoints_only(frame)
                    _, res = (extracted if isinstance(extracted, tuple) else (extracted, None))
                    world_lms = _extract_world_landmarks(res)

                    if world_lms and len(world_lms) > 28:
                        hl = [world_lms[23]['x'], world_lms[23]['y'], world_lms[23]['z']]
                        kl = [world_lms[25]['x'], world_lms[25]['y'], world_lms[25]['z']]
                        al = [world_lms[27]['x'], world_lms[27]['y'], world_lms[27]['z']]
                        sl = [world_lms[11]['x'], world_lms[11]['y'], world_lms[11]['z']]
                        
                        mp_knee = get_angle(hl, kl, al)
                        mp_hip = get_angle(sl, hl, kl)
                        gt_knee = get_angle(gt_frames[f_idx][4], gt_frames[f_idx][5], gt_frames[f_idx][6])
                        
                        temp_rows.append({
                            "subject_id": fid, "ex_id": ex_id, "view_type": cam_id, "frame_index": f_idx,
                            "mp_knee_angle": mp_knee, "mp_hip_angle": mp_hip, "gt_angle": gt_knee,
                            "k_x": kl[0]-hl[0], "k_y": kl[1]-hl[1], "k_z": kl[2]-hl[2],
                            "a_x": al[0]-hl[0], "a_y": al[1]-hl[1], "a_z": al[2]-hl[2],
                            "thigh_l": np.linalg.norm(np.array(kl)-np.array(hl)),
                            "shank_l": np.linalg.norm(np.array(al)-np.array(kl)),
                            "hip_y": hl[1], "hip_ankle_dist": np.linalg.norm(np.array(hl)-np.array(al))
                        })
                    f_idx += 1
                cap.release()
                
                if temp_rows:
                    df_v = pd.DataFrame(temp_rows)
                    med_t, med_s = df_v['thigh_l'].median(), df_v['shank_l'].median()
                    leg_ratio = med_t / (med_s + 1e-6)
                    med_total = med_t + med_s
                    df_v['leg_ratio'] = leg_ratio
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
    print(f"✨ All Fit3D Extraction complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_fit3d()
