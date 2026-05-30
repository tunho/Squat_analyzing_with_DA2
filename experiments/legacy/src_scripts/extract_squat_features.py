"""
extract_squat_features.py (v5 - Final Robust Version)

9명 피험자의 정면(Camera17) 및 측면(Camera18) 영상을 모두 스캔하여
GT(정답) 데이터가 존재하는 유효 구간의 피처를 추출합니다.

특징:
 - 18개 영상 전수 조사 (9 subjects * 2 views)
 - GT 데이터 범위 핀포인트 스캔 (고속)
 - 피험자/영상별 독립 리소스 관리 (안전)
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squat.analyzer import MPWorldRawSquatAnalyzer
from squat.domain.types import Point3D
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import point_to_np
from squat.pose_estimation import PoseEstimator
from squat.pipeline.frame_builder import build_squat_frame
from squat.pipeline.frame_processor import _extract_world_landmarks

# =========================================================
# 설정
# =========================================================
SUBJECTS = [
    "PM_008", "PM_022", "PM_029", "PM_038", "PM_043",
    "PM_105", "PM_113", "PM_118", "PM_126",
]

# 탐색할 카메라 시점 접미사
VIEW_SUFFIXES = [
    "Camera17-30fps",               # 정면
    "Camera17-30fps_compatible",    # 정면 호환
    "Camera18-30fps-transposed",    # 측면
]

VIDEO_DIR = PROJECT_ROOT / "dataset/raw/REHAB24-6/videos/Ex6"
GT_DIR = PROJECT_ROOT / "dataset/raw/REHAB24-6/3d_joints/Ex6"

GT_HIP_IDX, GT_KNEE_IDX, GT_ANKLE_IDX = 16, 17, 18
OUTPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"


def load_gt_angles(gt_path: Path) -> dict[int, float]:
    """GT npy에서 각도를 추출하여 {idx: angle} 반환."""
    try:
        arr = np.load(gt_path)
        angles = {}
        for t in range(arr.shape[0]):
            hip = Point3D(*arr[t, GT_HIP_IDX, :3].tolist(), 1.0)
            knee = Point3D(*arr[t, GT_KNEE_IDX, :3].tolist(), 1.0)
            ankle = Point3D(*arr[t, GT_ANKLE_IDX, :3].tolist(), 1.0)
            angles[t] = calculate_knee_angle(hip, knee, ankle)
        return angles
    except Exception as e:
        print(f"      [WARN] GT 로드 실패 ({gt_path.name}): {e}")
        return {}


def extract_features_from_analyzer(analyzer: MPWorldRawSquatAnalyzer, video_tag: str) -> pd.DataFrame:
    """분석기 데이터에서 학습용 피처 추출."""
    rows = []
    if not analyzer.history_data:
        return pd.DataFrame()

    # 엉덩이 깊이 정규화를 위한 max/min 산출
    hips_y = [f.mp_world_hip.y for f in analyzer.history_data if f.mp_world_hip]
    if not hips_y: return pd.DataFrame()
    y_min, y_max = min(hips_y), max(hips_y)
    y_range = max(y_max - y_min, 1e-6)

    # 뼈 길이 중앙값 (Scale 정규화용)
    thigh_lens, shank_lens = [], []
    for f in analyzer.history_data:
        if f.mp_world_hip and f.mp_world_knee and f.mp_world_ankle:
            h, k, a = point_to_np(f.mp_world_hip), point_to_np(f.mp_world_knee), point_to_np(f.mp_world_ankle)
            thigh_lens.append(np.linalg.norm(h - k))
            shank_lens.append(np.linalg.norm(a - k))
    
    med_total = np.median(thigh_lens) + np.median(shank_lens) if thigh_lens else 1.0
    view_type = "side" if "Camera18" in video_tag else "front"

    for f in analyzer.history_data:
        if f.mp_world_hip and f.mp_world_knee and f.mp_world_ankle:
            h = point_to_np(f.mp_world_hip)
            k = point_to_np(f.mp_world_knee)
            a = point_to_np(f.mp_world_ankle)
            s = point_to_np(f.shoulder) if f.shoulder else h # fallback to hip if no shoulder
            
            mp_knee_angle = calculate_knee_angle(f.mp_world_hip, f.mp_world_knee, f.mp_world_ankle)
            # Hip angle: shoulder - hip - knee
            mp_hip_angle = calculate_knee_angle(f.shoulder or f.mp_world_hip, f.mp_world_hip, f.mp_world_knee)
            
            rows.append({
                "frame_index": f.frame_index,
                "subject_id": video_tag.split('-')[0],
                "view_type": view_type,
                
                # Angles
                "mp_knee_angle": mp_knee_angle,
                "mp_hip_angle": mp_hip_angle,
                
                # Normalized positions (centered at hip, scaled by bone length)
                "k_x": (k[0] - h[0]) / med_total,
                "k_y": (k[1] - h[1]) / med_total,
                "k_z": (k[2] - h[2]) / med_total,
                "a_x": (a[0] - h[0]) / med_total,
                "a_y": (a[1] - h[1]) / med_total,
                "a_z": (a[2] - h[2]) / med_total,
                "s_x": (s[0] - h[0]) / med_total,
                "s_y": (s[1] - h[1]) / med_total,
                "s_z": (s[2] - h[2]) / med_total,
                
                # Visibility
                "h_vis": f.mp_world_hip.vis,
                "k_vis": f.mp_world_knee.vis,
                "a_vis": f.mp_world_ankle.vis,
                "s_vis": f.shoulder.vis if f.shoulder else 0.0,
                
                # Existing robust features
                "hip_depth_norm": (y_max - f.mp_world_hip.y) / y_range,
                "hip_ankle_dist_norm": np.linalg.norm(h - a) / med_total,
            })
    return pd.DataFrame(rows)


def run_fast_extraction(video_path: Path, gt_angles: dict[int, float], subject_id: str) -> pd.DataFrame:
    """GT 범위 내에서만 MediaPipe를 실행하여 특징 추출."""
    if not gt_angles: return pd.DataFrame()
    
    start_f, end_f = min(gt_angles.keys()), max(gt_angles.keys())
    tag = "side" if "Camera18" in video_path.name else "front"
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return pd.DataFrame()
    
    # 해당 구간으로 점프
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    
    pose_estimator = PoseEstimator(static_image_mode=False)
    analyzer = MPWorldRawSquatAnalyzer()

    print(f"    → {tag} 시점 처리 시작 ({start_f} ~ {end_f} 프레임)")
    
    curr_f = start_f
    while curr_f <= end_f:
        ret, bgr = cap.read()
        if not ret or bgr is None: break
        
        # 30fps 기준 타임스탬프 (MediaPipe Tasks API)
        extracted = pose_estimator.extract_keypoints_only(bgr, frame_timestamp_ms=curr_f * 33)
        landmarks, res_obj = (extracted if isinstance(extracted, tuple) else (extracted, None))
        
        if landmarks:
            world_lms = _extract_world_landmarks(res_obj)
            sq_frame = build_squat_frame(curr_f, "left", landmarks, None, bgr, None, False, False, world_lms)
            if sq_frame and sq_frame.mp_world_hip:
                analyzer.history_data.append(sq_frame)
        
        curr_f += 1
        if curr_f % 500 == 0:
            print(f"      ... {curr_f} / {end_f} 진행 중")
            
    cap.release()
    pose_estimator.close()
    
    df = extract_features_from_analyzer(analyzer, video_path.name)
    if not df.empty:
        df["gt_angle"] = df["frame_index"].map(gt_angles)
        df = df.dropna(subset=["gt_angle"])
    return df


def main():
    Path(OUTPUT_CSV.parent).mkdir(parents=True, exist_ok=True)
    all_df = pd.DataFrame()

    print(f"\n🚀 [9명 피험자 x 전 시점] 고속 통합 추출 시작 (중간 저장 활성)\n")

    for sid in SUBJECTS:
        print(f"  [{sid}] 처리 중...")
        gt_angles = load_gt_angles(GT_DIR / f"{sid}-30fps.npy")
        if not gt_angles: continue
        
        for suffix in VIEW_SUFFIXES:
            v_path = VIDEO_DIR / f"{sid}-{suffix}.mp4"
            if v_path.exists():
                try:
                    df = run_fast_extraction(v_path, gt_angles, sid)
                    if not df.empty:
                        df["subject_id"] = sid
                        all_df = pd.concat([all_df, df], ignore_index=True)
                        # 중간 저장 (데이터 안전성)
                        all_df.to_csv(OUTPUT_CSV, index=False)
                        print(f"    ✅ {v_path.name}: {len(df)} 샘플 성공 (누적: {len(all_df)})")
                except Exception as e:
                    print(f"    ❌ {v_path.name} 실패: {e}")
                    import traceback
                    traceback.print_exc()

    if all_df.empty:
        print("\n[ERROR] 추출된 데이터가 없습니다.")
    else:
        print(f"\n✨ 최종 완료! 총 {len(all_df)} 프레임 데이터 확보 → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
