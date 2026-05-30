import sys
import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squat.analyzer import MPWorldRawSquatAnalyzer
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import point_to_np
from squat.pose_estimation import PoseEstimator
from squat.pipeline.frame_builder import build_squat_frame
from squat.pipeline.frame_processor import _extract_world_landmarks
from squat.domain.types import Point3D

EX_LIST = ["Ex5", "Ex6"]
OUTPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/v13_mega/rehab_ex5_ex6_only.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
SEGMENT_CSV = PROJECT_ROOT / "dataset/raw/REHAB24-6/Segmentation.csv"

def get_leg_indices(video_id, ex_type):
    try:
        df_seg = pd.read_csv(SEGMENT_CSV, sep=';')
        sid = video_id.split('-')[0]
        ex_id_int = int(str(ex_type).replace("Ex", ""))
        sub_seg = df_seg[(df_seg['video_id'] == sid) & (df_seg['exercise_id'] == ex_id_int)]
        if not sub_seg.empty:
            subtype = str(sub_seg.iloc[0]['exercise_subtype']).lower()
            if 'right' in subtype: return 21, 22, 23
    except Exception: pass
    return 16, 17, 18

def extract_features_per_video(pe, video_path, ex_type, sid, gt_angles):
    cap = cv2.VideoCapture(str(video_path))
    history_data = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(gt_angles): break
        extracted = pe.extract_keypoints_only(frame, frame_timestamp_ms=frame_idx*33)
        landmarks, res_obj = (extracted if isinstance(extracted, tuple) else (extracted, None))
        if landmarks:
            world_lms = _extract_world_landmarks(res_obj)
            sq_frame = build_squat_frame(frame_idx, "left", landmarks, None, frame, None, False, False, world_lms)
            if sq_frame and sq_frame.mp_world_hip: 
                history_data.append(sq_frame)
        frame_idx += 1
    cap.release()

    if not history_data: return pd.DataFrame()

    rows = []
    view_type = "side" if "Camera18" in video_path.name else "front"
    ex_id = int(ex_type.replace("Ex", ""))
    
    hips_y = [f.mp_world_hip.y for f in history_data if f.mp_world_hip]
    y_min, y_max = (min(hips_y), max(hips_y)) if hips_y else (0, 1)
    y_range = max(y_max - y_min, 1e-6)

    thigh_lens, shank_lens = [], []
    for f in history_data:
        if f.mp_world_hip and f.mp_world_knee and f.mp_world_ankle:
            h, k, a = point_to_np(f.mp_world_hip), point_to_np(f.mp_world_knee), point_to_np(f.mp_world_ankle)
            thigh_lens.append(np.linalg.norm(h - k))
            shank_lens.append(np.linalg.norm(a - k))
    
    med_total = np.median(thigh_lens) + np.median(shank_lens) if thigh_lens else 1.0
    leg_ratio = np.median(thigh_lens) / (np.median(shank_lens) + 1e-6) if thigh_lens else 1.0

    for f in history_data:
        if f.mp_world_hip and f.mp_world_knee and f.mp_world_ankle:
            h, k, a = point_to_np(f.mp_world_hip), point_to_np(f.mp_world_knee), point_to_np(f.mp_world_ankle)
            s = point_to_np(f.shoulder) if f.shoulder else h
            
            rows.append({
                "frame_index": f.frame_index,
                "subject_id": sid,
                "view_type": view_type,
                "ex_id": ex_id,
                "mp_knee_angle": calculate_knee_angle(f.mp_world_hip, f.mp_world_knee, f.mp_world_ankle),
                "mp_hip_angle": calculate_knee_angle(f.shoulder or f.mp_world_hip, f.mp_world_hip, f.mp_world_knee),
                "leg_ratio": leg_ratio,
                "k_x": (k[0]-h[0])/med_total, "k_y": (k[1]-h[1])/med_total, "k_z": (k[2]-h[2])/med_total,
                "a_x": (a[0]-h[0])/med_total, "a_y": (a[1]-h[1])/med_total, "a_z": (a[2]-h[2])/med_total,
                "hip_depth_norm": (y_max - f.mp_world_hip.y) / y_range,
                "hip_ankle_dist_norm": np.linalg.norm(h-a)/med_total,
                "gt_angle": gt_angles[int(f.frame_index)] if int(f.frame_index) < len(gt_angles) else np.nan
            })
    
    df = pd.DataFrame(rows).dropna(subset=['gt_angle'])
    if df.empty: return df
    
    df['angle_velocity'] = df['mp_knee_angle'].diff().fillna(0)
    df['angle_vel_smooth'] = df['angle_velocity'].rolling(5, center=True).mean().fillna(0)
    df['knee_lag1'] = df['mp_knee_angle'].shift(1)
    df['knee_lag2'] = df['mp_knee_angle'].shift(2)
    return df.dropna()

def run_main():
    all_dfs = []
    print("🚀 통합 데이터 전수 추출 시작...")
    
    existing_combinations = set()
    if OUTPUT_CSV.exists():
        try:
            df_old = pd.read_csv(OUTPUT_CSV)
            for _, row in df_old.iterrows():
                existing_combinations.add((row['subject_id'], row['ex_id'], row['view_type']))
            all_dfs.append(df_old)
            print(f"📄 기존 데이터 {len(df_old)} 프레임 로드 완료.")
        except: pass

    pe = PoseEstimator(static_image_mode=True)
    try:
        for ex_type in EX_LIST:
            v_dir = PROJECT_ROOT / f"dataset/raw/REHAB24-6/videos/{ex_type}"
            gt_dir = PROJECT_ROOT / f"dataset/raw/REHAB24-6/3d_joints/{ex_type}"
            
            for v_path in sorted(list(v_dir.glob("*.mp4"))):
                sid = v_path.name.split('-')[0]
                view_type = "side" if "Camera18" in v_path.name else "front"
                ex_id_int = int(ex_type.replace("Ex", ""))
                
                if (sid, ex_id_int, view_type) in existing_combinations:
                    print(f"  ⏩ {ex_type} | {v_path.name} 스킵")
                    continue

                gt_path = gt_dir / f"{sid}-30fps.npy"
                if not gt_path.exists(): continue

                h_idx, k_idx, a_idx = get_leg_indices(v_path.name, ex_type)
                gt_arr = np.load(gt_path)
                gt_angles = [calculate_knee_angle(Point3D(*gt_arr[t, h_idx, :3].tolist(), 1.0),
                                                Point3D(*gt_arr[t, k_idx, :3].tolist(), 1.0),
                                                Point3D(*gt_arr[t, a_idx, :3].tolist(), 1.0))
                            for t in range(len(gt_arr))]

                df_video = extract_features_per_video(pe, v_path, ex_type, sid, gt_angles)
                if not df_video.empty:
                    all_dfs.append(df_video)
                    print(f"  ✅ {ex_type} | {v_path.name} 완료 ({len(df_video)} frames)")
                    pd.concat(all_dfs).to_csv(OUTPUT_CSV, index=False)
    finally:
        pe.close()
    
    print(f"\n✨ 모든 데이터 추출 완료! -> {OUTPUT_CSV}")

if __name__ == "__main__":
    run_main()
