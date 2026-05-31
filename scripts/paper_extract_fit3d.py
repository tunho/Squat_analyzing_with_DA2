"""Extract paper-compatible features from FiT3D squat videos for external validation.

Usage (from scripts/):
    PYTHONPATH=../src python paper_extract_fit3d.py \
        --dataset ../dataset/raw/fit3d/fit3d_train \
        --output ../experiments/paper/features_fit3d.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squat.domain.types import Point3D
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import point_to_np
from squat.pipeline.frame_builder import build_squat_frame
from squat.pipeline.frame_processor import _extract_world_landmarks
from squat.pose_estimation import PoseEstimator


# FiT3D joints3d_25 uses H36M-based ordering
# Left leg:  hip=4, knee=5, ankle=6
# Right leg: hip=1, knee=2, ankle=3
GT_LEFT_LEG = (4, 5, 6)

SUBJECTS = ["s03", "s04", "s05"]

# All 4 cameras at diagonal positions (no pure front/side/back exists in FiT3D).
# cam_id holds the accurate position label; view_type is the closest mapping for the
# model trained on REHAB24-6 (Camera17=front, Camera18=side). Both front-diagonal
# cameras are treated as "front"; both rear-diagonal cameras are also "front" since
# the training data contains no rear views — the model has never seen these.
CAM_INFO = {
    "50591643": ("left_rear",   "front"),
    "58860488": ("right_rear",  "front"),
    "60457274": ("right_front", "front"),
    "65906101": ("left_front",  "front"),
}

FPS = 50
TIMESTAMP_STEP_MS = 1000 // FPS  # 50fps → 20ms

# (start_sec, end_sec) per subject — exclude warmup/cooldown
TIME_RANGES = {
    "s03": (8.0, 20.0),
    "s04": (6.0, 18.0),
    "s05": (6.0, 17.0),
}


def load_gt_joints(json_path: Path) -> list:
    with open(json_path) as f:
        return json.load(f)["joints3d_25"]


def gt_knee_angle(frame_joints: list, leg_indices: tuple[int, int, int]) -> float:
    hi, ki, ai = leg_indices
    hip = Point3D(*frame_joints[hi][:3], 1.0)
    knee = Point3D(*frame_joints[ki][:3], 1.0)
    ankle = Point3D(*frame_joints[ai][:3], 1.0)
    return calculate_knee_angle(hip, knee, ankle)


def extract_feature_rows(
    frames_data: list[dict],
    gt_joints: list,
    subject_id: str,
    cam_id: str,
    view_type: str,
    calib_frames: int = 0,
) -> list[dict]:
    if not frames_data:
        return []

    hips_y = [f["hip_y"] for f in frames_data]
    y_min, y_max = min(hips_y), max(hips_y)
    y_range = max(y_max - y_min, 1e-6)

    # causal 정규화: 초기 calib_frames 로 med_total 추정. 0이면 전체중앙값(오프라인).
    calib = frames_data[:calib_frames] if calib_frames and calib_frames > 0 else frames_data
    med_total = max(
        float(np.median([f["thigh_len"] for f in calib])
              + np.median([f["shank_len"] for f in calib])),
        1e-6,
    )

    rows = []
    for f in frames_data:
        frame_idx = f["frame_index"]
        if frame_idx >= len(gt_joints):
            continue

        hip: Point3D = f["hip"]
        knee: Point3D = f["knee"]
        ankle: Point3D = f["ankle"]
        shoulder: Point3D | None = f["shoulder"]

        h = point_to_np(hip)
        k = point_to_np(knee)
        a = point_to_np(ankle)
        s = point_to_np(shoulder) if shoulder else h

        shoulder_pt = shoulder or hip
        mp_knee = calculate_knee_angle(hip, knee, ankle)
        mp_hip = calculate_knee_angle(shoulder_pt, hip, knee)
        gt_angle = gt_knee_angle(gt_joints[frame_idx], GT_LEFT_LEG)

        rows.append({
            "frame_index": frame_idx,
            "subject_id": subject_id,
            "cam_id": cam_id,
            "view_type": view_type,
            "mp_knee_angle": mp_knee,
            "mp_hip_angle": mp_hip,
            "k_x": (k[0] - h[0]) / med_total,
            "k_y": (k[1] - h[1]) / med_total,
            "k_z": (k[2] - h[2]) / med_total,
            "a_x": (a[0] - h[0]) / med_total,
            "a_y": (a[1] - h[1]) / med_total,
            "a_z": (a[2] - h[2]) / med_total,
            "s_x": (s[0] - h[0]) / med_total,
            "s_y": (s[1] - h[1]) / med_total,
            "s_z": (s[2] - h[2]) / med_total,
            "h_vis": hip.vis,
            "k_vis": knee.vis,
            "a_vis": ankle.vis,
            "s_vis": shoulder.vis if shoulder else 0.0,
            "hip_depth_norm": (y_max - f["hip_y"]) / y_range,
            "hip_ankle_dist_norm": float(np.linalg.norm(h - a) / med_total),
            "gt_angle": gt_angle,
        })
    return rows


def extract_video(
    pose_estimator: PoseEstimator,
    video_path: Path,
    gt_joints: list,
    subject_id: str,
    cam_id: str,
    view_type: str,
    start_frame: int,
    end_frame: int,
    calib_frames: int = 0,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return pd.DataFrame()

    max_frames = min(len(gt_joints), end_frame)
    frames_data: list[dict] = []
    frame_idx = 0

    try:
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx < start_frame:
                frame_idx += 1
                continue

            extracted = pose_estimator.extract_keypoints_only(
                frame, frame_timestamp_ms=frame_idx * TIMESTAMP_STEP_MS
            )
            landmarks, result = (
                extracted if isinstance(extracted, tuple) else (extracted, None)
            )

            if landmarks:
                world_lms = _extract_world_landmarks(result)
                squat_frame = build_squat_frame(
                    frame_index=frame_idx,
                    side="left",
                    landmarks=landmarks,
                    depth_map=None,
                    frame_bgr=frame,
                    drawn_image=None,
                    store_debug_images=False,
                    store_depth_maps=False,
                    world_landmarks=world_lms,
                )
                if squat_frame.mp_world_hip is not None:
                    hip = squat_frame.mp_world_hip
                    knee = squat_frame.mp_world_knee
                    ankle = squat_frame.mp_world_ankle
                    shoulder = squat_frame.mp_world_shoulder
                    h = point_to_np(hip)
                    k = point_to_np(knee)
                    a = point_to_np(ankle)
                    frames_data.append({
                        "frame_index": frame_idx,
                        "hip": hip,
                        "knee": knee,
                        "ankle": ankle,
                        "shoulder": shoulder,
                        "hip_y": hip.y,
                        "thigh_len": float(np.linalg.norm(h - k)),
                        "shank_len": float(np.linalg.norm(a - k)),
                    })

            frame_idx += 1
    finally:
        cap.release()

    rows = extract_feature_rows(frames_data, gt_joints, subject_id, cam_id, view_type, calib_frames=calib_frames)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract paper features from FiT3D squat videos."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("../dataset/raw/fit3d/fit3d_train"),
        help="Path to fit3d_train directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=SUBJECTS)
    parser.add_argument("--calib-frames", type=int, default=0,
                        help="causal 정규화: 초기 N프레임으로 med_total 추정. 0=시퀀스 전체중앙값(오프라인)")
    args = parser.parse_args()

    all_frames: list[pd.DataFrame] = []

    for subject in args.subjects:
        sub_dir = args.dataset / "train" / subject
        if not sub_dir.exists():
            print(f"Skip {subject}: directory not found")
            continue

        gt_path = sub_dir / "joints3d_25/squat.json"
        if not gt_path.exists():
            print(f"Skip {subject}: GT not found at {gt_path}")
            continue

        gt_joints = load_gt_joints(gt_path)
        subject_id = f"fit3d_{subject}"

        start_sec, end_sec = TIME_RANGES.get(subject, (0.0, len(gt_joints) / FPS))
        start_frame = int(start_sec * FPS)
        end_frame = int(end_sec * FPS)
        print(f"\n{subject}: GT frames {len(gt_joints)}, using {start_frame}-{end_frame} ({start_sec}s-{end_sec}s)")

        for cam_id, (cam_label, view_type) in CAM_INFO.items():
            video_path = sub_dir / f"videos/{cam_id}/squat.mp4"
            if not video_path.exists():
                continue

            pose_estimator = PoseEstimator(static_image_mode=False)
            try:
                df = extract_video(
                    pose_estimator, video_path, gt_joints,
                    subject_id, cam_label, view_type,
                    start_frame, end_frame, calib_frames=args.calib_frames,
                )
            finally:
                pose_estimator.close()

            if not df.empty:
                all_frames.append(df)
                print(f"  {subject}/{cam_id} ({cam_label}): {len(df)} frames")
            else:
                print(f"  {subject}/{cam_id}: no features extracted")

    if not all_frames:
        raise RuntimeError("No features were extracted.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.concat(all_frames, ignore_index=True)
    result.to_csv(args.output, index=False)
    print(f"\nTotal frames: {len(result)}")
    print(f"Subjects: {sorted(result['subject_id'].unique())}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
