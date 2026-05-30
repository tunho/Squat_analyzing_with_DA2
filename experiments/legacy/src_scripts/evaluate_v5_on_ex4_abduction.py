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
EX_TYPE = "Ex4"  # 다리 옆으로 들어 올리기 (Abduction)
VIDEO_DIR = PROJECT_ROOT / f"dataset/raw/REHAB24-6/videos/{EX_TYPE}"
GT_DIR = PROJECT_ROOT / f"dataset/raw/REHAB24-6/3d_joints/{EX_TYPE}"
MODEL_PATH = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final/paper_correction_model_v5.pkl"
SEGMENT_CSV = PROJECT_ROOT / "dataset/raw/REHAB24-6/Segmentation.csv"

# 피처 리스트 (V5 모델 학습 시 사용된 것과 동일해야 함)
FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle", "leg_ratio", "angle_vel_smooth",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis", "hip_depth_norm", "hip_ankle_dist_norm",
    "knee_lag1", "knee_lag2"
]

def get_leg_indices(video_id):
    """Segmentation.csv에서 해당 영상이 사용하는 다리 방향을 확인하여 인덱스 반환"""
    try:
        df_seg = pd.read_csv(SEGMENT_CSV, sep=';')
        # video_id match (video_id column matches filename prefix)
        sid = video_id.split('-')[0]
        sub_seg = df_seg[(df_seg['video_id'] == sid) & (df_seg['exercise_id'] == 4)]
        if sub_seg.empty:
            return 16, 17, 18 # Default: Left
        
        subtype = str(sub_seg.iloc[0]['exercise_subtype']).lower()
        if 'right' in subtype:
            return 21, 22, 23 # Right UpLeg, Leg, Foot
    except Exception:
        pass
    return 16, 17, 18     # Left UpLeg, Leg, Foot

def load_v5_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def extract_v5_features(analyzer, view_type):
    """V5 모델용 피처 추출 (Leg Ratio, Velocity, Lag 포함)"""
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
    # Velocity & Lag 산출
    df['angle_velocity'] = df['mp_knee_angle'].diff().fillna(0)
    df['angle_vel_smooth'] = df['angle_velocity'].rolling(5, center=True).mean().fillna(0)
    df['knee_lag1'] = df['mp_knee_angle'].shift(1)
    df['knee_lag2'] = df['mp_knee_angle'].shift(2)
    return df.dropna(subset=['knee_lag1', 'knee_lag2'])

def run_evaluation():
    if not MODEL_PATH.exists():
        print(f"❌ 모델 파일이 없습니다: {MODEL_PATH}")
        return

    print("--- Loading Model Bundle ---")
    bundle = load_v5_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    
    video_files = sorted(list(VIDEO_DIR.glob("*.mp4")))
    results = []

    print(f"\n🚀 [Ex4: 다리 들기] v5 모델 성능 평가 시작 (총 {len(video_files)}개 영상)\n")

    for v_path in video_files:
        sid = v_path.name.split('-')[0]
        
        # 1. GT 로드 및 관절 선택
        gt_path = GT_DIR / f"{sid}-30fps.npy"
        if not gt_path.exists(): 
            print(f"      [WARN] GT 없음: {gt_path.name}")
            continue
        
        h_idx, k_idx, a_idx = get_leg_indices(v_path.name)
        try:
            gt_arr = np.load(gt_path)
        except Exception as e:
            print(f"      [ERROR] GT 로드 실패 {gt_path.name}: {e}")
            continue
        
        # 2. MediaPipe 추출 (Fast Scan)
        cap = cv2.VideoCapture(str(v_path))
        pose_estimator = PoseEstimator(static_image_mode=False)
        analyzer = MPWorldRawSquatAnalyzer()
        
        print(f"    → {v_path.name} 처리 중...")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= len(gt_arr): break
            
            extracted = pose_estimator.extract_keypoints_only(frame, frame_timestamp_ms=frame_idx*33)
            landmarks, res_obj = (extracted if isinstance(extracted, tuple) else (extracted, None))
            
            if landmarks:
                world_lms = _extract_world_landmarks(res_obj)
                sq_frame = build_squat_frame(frame_idx, "left", landmarks, None, frame, None, False, False, world_lms)
                if sq_frame and sq_frame.mp_world_hip:
                    analyzer.history_data.append(sq_frame)
            frame_idx += 1
            if frame_idx % 500 == 0: 
                print(f"      ... {frame_idx} / {len(gt_arr)} 진행 중")

        cap.release()
        pose_estimator.close()

        # 3. 피처 생성 및 v5 보정
        if not analyzer.history_data: continue
        
        df = extract_v5_features(analyzer, "side" if "Camera18" in v_path.name else "front")
        if df.empty: continue

        # GT 각도 매핑
        gt_angles = []
        for t in range(len(gt_arr)):
            hip = Point3D(*gt_arr[t, h_idx, :3].tolist(), 1.0)
            knee = Point3D(*gt_arr[t, k_idx, :3].tolist(), 1.0)
            ankle = Point3D(*gt_arr[t, a_idx, :3].tolist(), 1.0)
            gt_angles.append(calculate_knee_angle(hip, knee, ankle))
        
        df['gt_angle'] = df['frame_index'].apply(lambda idx: gt_angles[int(idx)] if int(idx) < len(gt_angles) else np.nan)
        df = df.dropna(subset=['gt_angle'])
        
        if df.empty: continue

        # V5 보정 수행
        X = scaler.transform(df[FEATURE_COLS])
        df['v5_residual_pred'] = model.predict(X)
        df['v5_corrected_angle'] = df['mp_knee_angle'] + df['v5_residual_pred']
        
        # 성과 기록
        mae_raw = np.abs(df['gt_angle'] - df['mp_knee_angle']).mean()
        mae_v5 = np.abs(df['gt_angle'] - df['v5_corrected_angle']).mean()
        
        print(f"    ✨ MAE: Raw {mae_raw:.2f}° -> v5 {mae_v5:.2f}° (개선 {mae_raw-mae_v5:.2f}°)")
        results.append({
            "video": v_path.name,
            "mae_raw": mae_raw,
            "mae_v5": mae_v5
        })

        # 중간 저장
        pd.DataFrame(results).to_csv(PROJECT_ROOT / "outputs/ml_correction/v5_paper_final/ex4_evaluation_results.csv", index=False)

    # 최종 요약
    if not results: return

    report_df = pd.DataFrame(results)
    avg_raw = report_df['mae_raw'].mean()
    avg_v5 = report_df['mae_v5'].mean()
    
    print("\n" + "="*50)
    print(f"📊 [Ex4: Abduction] v5 모델 최종 평가 결과")
    print(f"  - 평균 Raw MAE: {avg_raw:.2f}°")
    print(f"  - 평균 v5 MAE:  {avg_v5:.2f}°")
    print(f"  - 전체 개선량:   {avg_raw - avg_v5:.2f}°")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()
