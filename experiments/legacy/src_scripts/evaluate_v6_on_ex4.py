import sys
import pickle
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squat.analyzer import MPWorldRawSquatAnalyzer
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import point_to_np
from squat.pose_estimation import PoseEstimator
from squat.pipeline.frame_builder import build_squat_frame
from squat.pipeline.frame_processor import _extract_world_landmarks
from squat.domain.types import Point3D

# =========================================================
# 설정
# =========================================================
EX_TYPE = "Ex4"  # 다리 옆으로 들기 (Abduction)
VIDEO_DIR = PROJECT_ROOT / f"dataset/raw/REHAB24-6/videos/{EX_TYPE}"
GT_DIR = PROJECT_ROOT / f"dataset/raw/REHAB24-6/3d_joints/{EX_TYPE}"
MODEL_PATH = PROJECT_ROOT / "outputs/ml_correction/v6_single_final/squat_single_model_v6.pkl"
SEGMENT_CSV = PROJECT_ROOT / "dataset/raw/REHAB24-6/Segmentation.csv"

# 피처 리스트 (V6 모델과 동일)
FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle", "leg_ratio", "angle_vel_smooth",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis", "hip_depth_norm", "hip_ankle_dist_norm",
    "knee_lag1", "knee_lag2"
]

def get_leg_indices(video_id):
    try:
        df_seg = pd.read_csv(SEGMENT_CSV, sep=';')
        sid = video_id.split('-')[0]
        sub_seg = df_seg[(df_seg['video_id'] == sid) & (df_seg['exercise_id'] == 4)]
        if sub_seg.empty: return 16, 17, 18
        subtype = str(sub_seg.iloc[0]['exercise_subtype']).lower()
        if 'right' in subtype: return 21, 22, 23
    except: pass
    return 16, 17, 18

def extract_v6_features(analyzer):
    rows = []
    if not analyzer.history_data: return pd.DataFrame()
    hips_y = [f.mp_world_hip.y for f in analyzer.history_data if f.mp_world_hip]
    if not hips_y: return pd.DataFrame()
    y_min, y_max = min(hips_y), max(hips_y)
    y_range = max(y_max - y_min, 1e-6)
    thigh_lens, shank_lens = [], []
    for f in analyzer.history_data:
        if f.mp_world_hip and f.mp_world_knee and f.mp_world_ankle:
            h, k, a = point_to_np(f.mp_world_hip), point_to_np(f.mp_world_knee), point_to_np(f.mp_world_ankle)
            thigh_lens.append(np.linalg.norm(h - k)); shank_lens.append(np.linalg.norm(a - k))
    if not thigh_lens: return pd.DataFrame()
    med_total = np.median(thigh_lens) + np.median(shank_lens); leg_ratio = np.median(thigh_lens) / (np.median(shank_lens) + 1e-6)
    for f in analyzer.history_data:
        if f.mp_world_hip and f.mp_world_knee and f.mp_world_ankle:
            h, k, a = point_to_np(f.mp_world_hip), point_to_np(f.mp_world_knee), point_to_np(f.mp_world_ankle)
            s = point_to_np(f.shoulder) if f.shoulder else h
            mp_knee_angle = calculate_knee_angle(f.mp_world_hip, f.mp_world_knee, f.mp_world_ankle)
            mp_hip_angle = calculate_knee_angle(f.shoulder or f.mp_world_hip, f.mp_world_hip, f.mp_world_knee)
            rows.append({
                "frame_index": f.frame_index, "mp_knee_angle": mp_knee_angle, "mp_hip_angle": mp_hip_angle, "leg_ratio": leg_ratio,
                "k_x": (k[0]-h[0])/med_total, "k_y": (k[1]-h[1])/med_total, "k_z": (k[2]-h[2])/med_total,
                "a_x": (a[0]-h[0])/med_total, "a_y": (a[1]-h[1])/med_total, "a_z": (a[2]-h[2])/med_total,
                "s_x": (s[0]-h[0])/med_total, "s_y": (s[1]-h[1])/med_total, "s_z": (s[2]-h[2])/med_total,
                "h_vis": f.mp_world_hip.vis, "k_vis": f.mp_world_knee.vis, "a_vis": f.mp_world_ankle.vis, "s_vis": f.shoulder.vis if f.shoulder else 0.0,
                "hip_depth_norm": (y_max - f.mp_world_hip.y) / y_range, "hip_ankle_dist_norm": np.linalg.norm(h-a)/med_total
            })
    df = pd.DataFrame(rows)
    df['angle_velocity'] = df['mp_knee_angle'].diff().fillna(0)
    df['angle_vel_smooth'] = df['angle_velocity'].rolling(5, center=True).mean().fillna(0)
    df['knee_lag1'] = df['mp_knee_angle'].shift(1); df['knee_lag2'] = df['mp_knee_angle'].shift(2)
    return df.dropna(subset=['knee_lag1', 'knee_lag2'])

def run_evaluation():
    with open(MODEL_PATH, "rb") as f: bundle = pickle.load(f)
    model, scaler = bundle["model"], bundle["scaler"]
    video_files = sorted(list(VIDEO_DIR.glob("*.mp4")))
    results = []
    print(f"\n🚀 [Ex4 v6 통합모델 테스트] 시작\n")
    for v_path in video_files:
        sid = v_path.name.split('-')[0]
        gt_path = GT_DIR / f"{sid}-30fps.npy"
        if not gt_path.exists(): continue
        h_idx, k_idx, a_idx = get_leg_indices(v_path.name); gt_arr = np.load(gt_path)
        cap = cv2.VideoCapture(str(v_path)); pose_estimator = PoseEstimator(static_image_mode=False); analyzer = MPWorldRawSquatAnalyzer()
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= len(gt_arr): break
            extracted = pose_estimator.extract_keypoints_only(frame, frame_timestamp_ms=frame_idx*33)
            landmarks, res_obj = (extracted if isinstance(extracted, tuple) else (extracted, None))
            if landmarks:
                world_lms = _extract_world_landmarks(res_obj)
                sq_frame = build_squat_frame(frame_idx, "left", landmarks, None, frame, None, False, False, world_lms)
                if sq_frame and sq_frame.mp_world_hip: analyzer.history_data.append(sq_frame)
            frame_idx += 1
        cap.release(); pose_estimator.close()
        if not analyzer.history_data: continue
        df = extract_v6_features(analyzer)
        if df.empty: continue
        gt_angles = []
        for t in range(len(gt_arr)):
            hip = Point3D(*gt_arr[t, h_idx, :3].tolist(), 1.0); knee = Point3D(*gt_arr[t, k_idx, :3].tolist(), 1.0); ankle = Point3D(*gt_arr[t, a_idx, :3].tolist(), 1.0)
            gt_angles.append(calculate_knee_angle(hip, knee, ankle))
        df['gt_angle'] = df['frame_index'].apply(lambda idx: gt_angles[int(idx)] if int(idx) < len(gt_angles) else np.nan); df = df.dropna(subset=['gt_angle'])
        if df.empty: continue
        df['v6_corrected_angle'] = df['mp_knee_angle'] + model.predict(scaler.transform(df[FEATURE_COLS]))
        mae_raw = np.abs(df['gt_angle'] - df['mp_knee_angle']).mean(); mae_v6 = np.abs(df['gt_angle'] - df['v6_corrected_angle']).mean()
        print(f"  ✅ {v_path.name}: Raw {mae_raw:.2f}° -> v6 {mae_v6:.2f}°")
        results.append({"video": v_path.name, "mae_raw": mae_raw, "mae_v6": mae_v6})
        pd.DataFrame(results).to_csv(PROJECT_ROOT / "outputs/ml_correction/v6_single_final/ex4_evaluation_v6.csv", index=False)
    print("\n✅ Ex4 v6 평가 완료")

if __name__ == "__main__": run_evaluation()
