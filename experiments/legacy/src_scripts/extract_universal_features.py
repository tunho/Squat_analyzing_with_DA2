import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squat.analyzer import MPWorldRawSquatAnalyzer
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import point_to_np
from squat.pose_estimation import PoseEstimator
from squat.pipeline.frame_builder import build_squat_frame
from squat.pipeline.frame_processor import _extract_world_landmarks
from squat.domain.types import Point3D

# 설정
EX_LIST = ["Ex4", "Ex5", "Ex6"]
OUTPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/v7_universal/universal_features_ex456.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
SEGMENT_CSV = PROJECT_ROOT / "dataset/raw/REHAB24-6/Segmentation.csv"

def get_leg_indices(video_id, ex_type):
    try:
        df_seg = pd.read_csv(SEGMENT_CSV, sep=';')
        sid = video_id.split('-')[0]
        ex_id = int(ex_type.replace("Ex", ""))
        sub_seg = df_seg[(df_seg['video_id'] == sid) & (df_seg['exercise_id'] == ex_id)]
        if sub_seg.empty: return 16, 17, 18
        subtype = str(sub_seg.iloc[0]['exercise_subtype']).lower()
        if 'right' in subtype: return 21, 22, 23
    except: pass
    return 16, 17, 18

def extract_features_v7(analyzer, ex_type, sid, video_tag):
    rows = []
    if not analyzer.history_data: return pd.DataFrame()
    
    view_type = "side" if "Camera18" in video_tag else "front"
    ex_id = int(ex_type.replace("Ex", ""))
    
    hips_y = [f.mp_world_hip.y for f in analyzer.history_data if f.mp_world_hip]
    if not hips_y: return pd.DataFrame()
    y_min, y_max = min(hips_y), max(hips_y)
    y_range = max(y_max - y_min, 1e-6)
    
    thigh_lens, shank_lens = [], []
    for f in analyzer.history_data:
        if f.mp_world_hip and f.mp_world_knee and f.mp_world_ankle:
            h, k, a = point_to_np(f.mp_world_hip), point_to_np(f.mp_world_knee), point_to_np(f.mp_world_ankle)
            thigh_lens.append(np.linalg.norm(h - k))
            shank_lens.append(np.linalg.norm(a - k))
    
    if not thigh_lens: return pd.DataFrame()
    med_total = np.median(thigh_lens) + np.median(shank_lens)
    leg_ratio = np.median(thigh_lens) / (np.median(shank_lens) + 1e-6)

    for f in analyzer.history_data:
        if f.mp_world_hip and f.mp_world_knee and f.mp_world_ankle:
            h, k, a = point_to_np(f.mp_world_hip), point_to_np(f.mp_world_knee), point_to_np(f.mp_world_ankle)
            s = point_to_np(f.shoulder) if f.shoulder else h
            mp_knee_angle = calculate_knee_angle(f.mp_world_hip, f.mp_world_knee, f.mp_world_ankle)
            mp_hip_angle = calculate_knee_angle(f.shoulder or f.mp_world_hip, f.mp_world_hip, f.mp_world_knee)
            
            rows.append({
                "frame_index": f.frame_index,
                "subject_id": sid,
                "view_type": view_type,
                "ex_id": ex_id,
                "mp_knee_angle": mp_knee_angle,
                "mp_hip_angle": mp_hip_angle,
                "leg_ratio": leg_ratio,
                "k_x": (k[0]-h[0])/med_total, "k_y": (k[1]-h[1])/med_total, "k_z": (k[2]-h[2])/med_total,
                "a_x": (a[0]-h[0])/med_total, "a_y": (a[1]-h[1])/med_total, "a_z": (a[2]-h[2])/med_total,
                "s_x": (s[0]-h[0])/med_total, "s_y": (s[1]-h[1])/med_total, "s_z": (s[2]-h[2])/med_total,
                "h_vis": f.mp_world_hip.vis, "k_vis": f.mp_world_knee.vis, "a_vis": f.mp_world_ankle.vis, "s_vis": f.shoulder.vis if f.shoulder else 0.0,
                "hip_depth_norm": (y_max - f.mp_world_hip.y) / y_range,
                "hip_ankle_dist_norm": np.linalg.norm(h-a)/med_total
            })
    
    df = pd.DataFrame(rows)
    df['angle_velocity'] = df['mp_knee_angle'].diff().fillna(0)
    df['angle_vel_smooth'] = df['angle_velocity'].rolling(5, center=True).mean().fillna(0)
    df['knee_lag1'] = df['mp_knee_angle'].shift(1)
    df['knee_lag2'] = df['mp_knee_angle'].shift(2)
    return df.dropna(subset=['knee_lag1', 'knee_lag2'])

def run_extraction():
    all_rows = []
    pose_estimator = PoseEstimator(static_image_mode=False)

    for ex_type in EX_LIST:
        print(f"\n📂 {ex_type} 추출 시작...")
        v_dir = PROJECT_ROOT / f"dataset/raw/REHAB24-6/videos/{ex_type}"
        gt_dir = PROJECT_ROOT / f"dataset/raw/REHAB24-6/3d_joints/{ex_type}"
        
        for v_path in sorted(list(v_dir.glob("*.mp4"))):
            sid = v_path.name.split('-')[0]
            gt_path = gt_dir / f"{sid}-30fps.npy"
            if not gt_path.exists(): continue

            # GT
            h_idx, k_idx, a_idx = get_leg_indices(v_path.name, ex_type)
            gt_arr = np.load(gt_path)
            gt_angles = []
            for t in range(len(gt_arr)):
                hip = Point3D(*gt_arr[t, h_idx, :3].tolist(), 1.0)
                knee = Point3D(*gt_arr[t, k_idx, :3].tolist(), 1.0)
                ankle = Point3D(*gt_arr[t, a_idx, :3].tolist(), 1.0)
                gt_angles.append(calculate_knee_angle(hip, knee, ankle))

            cap = cv2.VideoCapture(str(v_path))
            analyzer = MPWorldRawSquatAnalyzer()
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_idx >= len(gt_angles): break
                extracted = pose_estimator.extract_keypoints_only(frame, frame_timestamp_ms=frame_idx*33)
                landmarks, res_obj = (extracted if isinstance(extracted, tuple) else (extracted, None))
                if landmarks:
                    world_lms = _extract_world_landmarks(res_obj)
                    sq_frame = build_squat_frame(frame_idx, "left", landmarks, None, frame, None, False, False, world_lms)
                    if sq_frame and sq_frame.mp_world_hip: analyzer.history_data.append(sq_frame)
                frame_idx += 1
            cap.release()

            if not analyzer.history_data: continue
            df = extract_features_v7(analyzer, ex_type, sid, v_path.name)
            if df.empty: continue
            df['gt_angle'] = df['frame_index'].apply(lambda idx: gt_angles[int(idx)] if int(idx) < len(gt_angles) else np.nan)
            df = df.dropna(subset=['gt_angle'])
            all_rows.append(df)
            print(f"  ✅ {v_path.name} 완료 ({len(df)} frames)")
            
            # 중간 저장 (데이터 보호)
            pd.concat(all_rows).to_csv(OUTPUT_CSV, index=False)

    pose_estimator.close()
    print(f"\n✨ 모든 통합 피처 추출 완료 -> {OUTPUT_CSV}")

if __name__ == "__main__":
    run_extraction()
