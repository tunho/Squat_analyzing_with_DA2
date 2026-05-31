from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from squat.domain.types import Point3D
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import point_to_np
from squat.pipeline.frame_builder import build_squat_frame
from squat.pipeline.frame_processor import _extract_world_landmarks


DEFAULT_VIEW_SUFFIXES = [
    "Camera17-30fps",
    "Camera17-30fps_compatible",
    "Camera18-30fps-transposed",
]


def load_segmentation_table(dataset: Path) -> pd.DataFrame | None:
    path = dataset / "Segmentation.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep=";")


def get_leg_indices(seg_df: pd.DataFrame | None, subject_id: str, exercise: str) -> tuple[int, int, int]:
    if seg_df is None:
        return 16, 17, 18
    try:
        exercise_id = int(exercise.replace("Ex", ""))
        sub = seg_df[(seg_df["video_id"].astype(str).str.startswith(subject_id)) & (seg_df["exercise_id"] == exercise_id)]
        if sub.empty:
            return 16, 17, 18
        subtype = str(sub.iloc[0].get("exercise_subtype", "")).lower()
        if "right" in subtype:
            return 21, 22, 23
    except Exception:
        return 16, 17, 18
    return 16, 17, 18


def load_gt_angles(gt_path: Path, joint_indices: tuple[int, int, int]) -> dict[int, float]:
    arr = np.load(gt_path)
    hip_idx, knee_idx, ankle_idx = joint_indices
    out: dict[int, float] = {}
    for frame_idx in range(arr.shape[0]):
        hip = Point3D(*arr[frame_idx, hip_idx, :3].tolist(), 1.0)
        knee = Point3D(*arr[frame_idx, knee_idx, :3].tolist(), 1.0)
        ankle = Point3D(*arr[frame_idx, ankle_idx, :3].tolist(), 1.0)
        out[frame_idx] = calculate_knee_angle(hip, knee, ankle)
    return out


def extract_feature_rows(frames: list, video_name: str, subject_id: str, med_total: float | None = None) -> list[dict]:
    # med_total override(causal 캘리브레이션 정규화)용. None이면 기존대로 시퀀스 전체 중앙값 사용(하위호환).
    if not frames:
        return []

    hips_y = [frame.mp_world_hip.y for frame in frames if frame.mp_world_hip]
    if not hips_y:
        return []
    y_min = min(hips_y)
    y_max = max(hips_y)
    y_range = max(y_max - y_min, 1e-6)

    thigh_lens = []
    shank_lens = []
    for frame in frames:
        if frame.mp_world_hip and frame.mp_world_knee and frame.mp_world_ankle:
            hip = point_to_np(frame.mp_world_hip)
            knee = point_to_np(frame.mp_world_knee)
            ankle = point_to_np(frame.mp_world_ankle)
            thigh_lens.append(np.linalg.norm(hip - knee))
            shank_lens.append(np.linalg.norm(ankle - knee))
    if not thigh_lens or not shank_lens:
        return []

    if med_total is None:
        med_total = float(np.median(thigh_lens) + np.median(shank_lens))
    med_total = max(med_total, 1e-6)
    view_type = "side" if "Camera18" in video_name else "front"

    rows = []
    for frame in frames:
        if not (frame.mp_world_hip and frame.mp_world_knee and frame.mp_world_ankle):
            continue

        hip = point_to_np(frame.mp_world_hip)
        knee = point_to_np(frame.mp_world_knee)
        ankle = point_to_np(frame.mp_world_ankle)
        shoulder_point = frame.mp_world_shoulder or frame.shoulder
        shoulder = point_to_np(shoulder_point) if shoulder_point else hip

        rows.append(
            {
                "frame_index": frame.frame_index,
                "subject_id": subject_id,
                "view_type": view_type,
                "mp_knee_angle": calculate_knee_angle(frame.mp_world_hip, frame.mp_world_knee, frame.mp_world_ankle),
                "mp_hip_angle": calculate_knee_angle(shoulder_point or frame.mp_world_hip, frame.mp_world_hip, frame.mp_world_knee),
                "k_x": (knee[0] - hip[0]) / med_total,
                "k_y": (knee[1] - hip[1]) / med_total,
                "k_z": (knee[2] - hip[2]) / med_total,
                "a_x": (ankle[0] - hip[0]) / med_total,
                "a_y": (ankle[1] - hip[1]) / med_total,
                "a_z": (ankle[2] - hip[2]) / med_total,
                "s_x": (shoulder[0] - hip[0]) / med_total,
                "s_y": (shoulder[1] - hip[1]) / med_total,
                "s_z": (shoulder[2] - hip[2]) / med_total,
                "h_vis": frame.mp_world_hip.vis,
                "k_vis": frame.mp_world_knee.vis,
                "a_vis": frame.mp_world_ankle.vis,
                "s_vis": shoulder_point.vis if shoulder_point else 0.0,
                "hip_depth_norm": (y_max - frame.mp_world_hip.y) / y_range,
                "hip_ankle_dist_norm": float(np.linalg.norm(hip - ankle) / med_total),
            }
        )
    return rows


def calib_med_total(frames: list, k: int) -> float | None:
    """초기 k프레임의 분절길이로 med_total 추정(causal 정규화). k<=0이면 None."""
    if not k or k <= 0:
        return None
    th, sh = [], []
    for f in frames[:k]:
        if f.mp_world_hip and f.mp_world_knee and f.mp_world_ankle:
            hip = point_to_np(f.mp_world_hip); knee = point_to_np(f.mp_world_knee); ank = point_to_np(f.mp_world_ankle)
            th.append(float(np.linalg.norm(hip - knee))); sh.append(float(np.linalg.norm(ank - knee)))
    if not th or not sh:
        return None
    return max(float(np.median(th) + np.median(sh)), 1e-6)


def extract_video_features(
    pose_estimator,
    video_path: Path,
    gt_angles: dict[int, float],
    subject_id: str,
    timestamp_step_ms: int,
    calib_frames: int = 0,
) -> pd.DataFrame:
    if not gt_angles:
        return pd.DataFrame()

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return pd.DataFrame()

    start_frame = min(gt_angles)
    end_frame = max(gt_angles)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_idx = start_frame
    try:
        while frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            extracted = pose_estimator.extract_keypoints_only(frame, frame_timestamp_ms=frame_idx * timestamp_step_ms)
            landmarks, result = extracted if isinstance(extracted, tuple) else (extracted, None)
            if landmarks:
                world_landmarks = _extract_world_landmarks(result)
                squat_frame = build_squat_frame(
                    frame_index=frame_idx,
                    side="left",
                    landmarks=landmarks,
                    depth_map=None,
                    frame_bgr=frame,
                    drawn_image=None,
                    store_debug_images=False,
                    store_depth_maps=False,
                    world_landmarks=world_landmarks,
                )
                if squat_frame.mp_world_hip is not None:
                    frames.append(squat_frame)

            frame_idx += 1
    finally:
        cap.release()

    df = pd.DataFrame(extract_feature_rows(frames, video_path.name, subject_id, med_total=calib_med_total(frames, calib_frames)))
    if df.empty:
        return df
    df["gt_angle"] = df["frame_index"].map(gt_angles)
    return df.dropna(subset=["gt_angle"]).copy()


def discover_subjects(gt_dir: Path) -> list[str]:
    return sorted(path.name.replace("-30fps.npy", "") for path in gt_dir.glob("*-30fps.npy"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract official paper features from REHAB24-6 videos.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset/raw/REHAB24-6"))
    parser.add_argument("--exercise", default="Ex6")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--view-suffixes", nargs="*", default=DEFAULT_VIEW_SUFFIXES)
    parser.add_argument("--timestamp-step-ms", type=int, default=33)
    parser.add_argument("--calib-frames", type=int, default=0,
                        help="causal 정규화: 초기 N프레임으로 med_total 추정. 0=시퀀스 전체중앙값(오프라인)")
    args = parser.parse_args()

    video_dir = args.dataset / "videos" / args.exercise
    gt_dir = args.dataset / "3d_joints" / args.exercise
    subjects = args.subjects or discover_subjects(gt_dir)
    seg_df = load_segmentation_table(args.dataset)

    from squat.pose_estimation import PoseEstimator

    all_frames = []
    for subject_id in subjects:
        gt_path = gt_dir / f"{subject_id}-30fps.npy"
        if not gt_path.exists():
            continue
        gt_angles = load_gt_angles(gt_path, get_leg_indices(seg_df, subject_id, args.exercise))

        for suffix in args.view_suffixes:
            video_path = video_dir / f"{subject_id}-{suffix}.mp4"
            if not video_path.exists():
                continue
            pose_estimator = PoseEstimator(static_image_mode=False)
            try:
                df = extract_video_features(pose_estimator, video_path, gt_angles, subject_id, args.timestamp_step_ms, calib_frames=args.calib_frames)
            finally:
                pose_estimator.close()
            if not df.empty:
                all_frames.append(df)
                print(f"Extracted {len(df)} rows from {video_path.name}")

    if not all_frames:
        raise RuntimeError("No feature rows were extracted.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_frames, ignore_index=True).to_csv(args.output, index=False)
    print(f"Saved features: {args.output}")


if __name__ == "__main__":
    main()
