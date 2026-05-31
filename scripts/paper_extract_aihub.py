"""AI Hub '크로스핏 동작 데이터'(213) → official paper feature CSV.

REHAB24-6용 paper_extract.py 와 동일한 스키마/피처를 생성한다. 차이점만:
  - GT: 세션별 3d_points.csv (mm, 26관절 by name) 에서 좌측 무릎각 산출
  - view_type: 8캠 고정 리그 → 카메라 인덱스로 매핑 (--side-cameras)
  - 동작 구간: 카메라별 annotation.json 의 start_frame~end_frame (--full-clip 로 전체)

라벨/원천은 AI Hub 구조상 분리돼 있다:
  labels_root/{운동}/.../{CA##}/{세션}/3d_points.csv  (+ cameraN/video/annotation.json)
  source_root/{운동}/.../{CA##}/{세션}/cameraN/video/*.avi

PYTHONPATH=src .venv-paper/bin/python scripts/paper_extract_aihub.py \
  --labels-root dataset/raw/aihub_test/labels \
  --source-root dataset/raw/aihub_test/source \
  --output experiments/paper/features_aihub.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from squat.domain.types import Point3D
from squat.geometry.angles import calculate_knee_angle
from squat.pipeline.frame_builder import build_squat_frame
from squat.pipeline.frame_processor import _extract_world_landmarks

# REHAB 추출기의 피처 계산 로직 재사용 (동일 스키마 보장)
from paper_extract import extract_feature_rows

# AI Hub 3D/2D 키포인트 컬럼명 (좌측 다리)
GT_LEFT_LEG = ("LHip", "LKnee", "LAnkle")
GT_RIGHT_LEG = ("RHip", "RKnee", "RAnkle")
_CAM_RE = re.compile(r"camera(\d+)")


def load_gt_angles(csv_path: Path, leg: tuple[str, str, str]) -> dict[int, float]:
    """3d_points.csv → {frame_index: gt_knee_angle}. frame_index 는 image_filename(N.jpg)."""
    df = pd.read_csv(csv_path)
    hip_j, knee_j, ankle_j = leg
    out: dict[int, float] = {}
    for i, row in df.iterrows():
        fname = str(row.get("image_filename", i))
        m = re.match(r"(\d+)", fname)
        idx = int(m.group(1)) if m else int(i)

        def pt(j: str) -> Point3D:
            return Point3D(float(row[f"{j}_x"]), float(row[f"{j}_y"]), float(row[f"{j}_z"]), 1.0)

        out[idx] = calculate_knee_angle(pt(hip_j), pt(knee_j), pt(ankle_j))
    return out


def read_segment(annotation_path: Path) -> tuple[int, int] | None:
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        ann = data.get("annotations") or []
        if not ann:
            return None
        a = ann[0]
        return int(a["start_frame"]), int(a["end_frame"])
    except Exception:
        return None


def extract_camera(
    pose_estimator,
    avi_path: Path,
    gt_angles: dict[int, float],
    subject_id: str,
    view_type: str,
    camera: str,
    bounds: tuple[int, int] | None,
    timestamp_step_ms: int,
) -> pd.DataFrame:
    import cv2

    if not gt_angles:
        return pd.DataFrame()
    lo = max(min(gt_angles), bounds[0]) if bounds else min(gt_angles)
    hi = min(max(gt_angles), bounds[1]) if bounds else max(gt_angles)

    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return pd.DataFrame()
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)

    frames = []
    frame_idx = lo
    try:
        while frame_idx <= hi:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            extracted = pose_estimator.extract_keypoints_only(
                frame, frame_timestamp_ms=frame_idx * timestamp_step_ms
            )
            landmarks, result = extracted if isinstance(extracted, tuple) else (extracted, None)
            if landmarks:
                squat_frame = build_squat_frame(
                    frame_index=frame_idx,
                    side="left",
                    landmarks=landmarks,
                    depth_map=None,
                    frame_bgr=frame,
                    drawn_image=None,
                    store_debug_images=False,
                    store_depth_maps=False,
                    world_landmarks=_extract_world_landmarks(result),
                )
                if squat_frame.mp_world_hip is not None:
                    frames.append(squat_frame)
            frame_idx += 1
    finally:
        cap.release()

    df = pd.DataFrame(extract_feature_rows(frames, avi_path.name, subject_id))
    if df.empty:
        return df
    df["view_type"] = view_type           # 카메라 인덱스 기반으로 덮어씀
    df["camera"] = camera
    df["gt_angle"] = df["frame_index"].map(gt_angles)
    return df.dropna(subset=["gt_angle"]).copy()


def find_source_session(source_root: Path, rel_session: Path) -> Path | None:
    cand = source_root / rel_session
    return cand if cand.exists() else None


def main() -> None:
    p = argparse.ArgumentParser(description="Extract official paper features from AI Hub 213 CrossFit dataset.")
    p.add_argument("--labels-root", type=Path, required=True, help="02.라벨링데이터 루트 (3d_points.csv 포함)")
    p.add_argument("--source-root", type=Path, required=True, help="01.원천데이터 루트 (cameraN/video/*.avi 포함)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exercise-filter", default=None, help="경로에 포함돼야 하는 문자열 (예: 스쿼트)")
    p.add_argument("--side-cameras", nargs="*", type=int, default=[2, 3], help="side 로 태깅할 카메라 인덱스")
    p.add_argument("--cameras", nargs="*", type=int, default=None, help="처리할 카메라 인덱스 (기본 전체)")
    p.add_argument("--leg", choices=["left", "right"], default="left")
    p.add_argument("--full-clip", action="store_true", help="annotation 구간 무시하고 전체 프레임 사용")
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--timestamp-step-ms", type=int, default=33)
    args = p.parse_args()

    leg = GT_LEFT_LEG if args.leg == "left" else GT_RIGHT_LEG
    side_set = set(args.side_cameras)
    cam_filter = set(args.cameras) if args.cameras is not None else None

    gt_files = sorted(args.labels_root.rglob("3d_points.csv"))
    if args.exercise_filter:
        gt_files = [g for g in gt_files if args.exercise_filter in str(g)]
    if args.max_sessions:
        gt_files = gt_files[: args.max_sessions]
    if not gt_files:
        raise RuntimeError(f"3d_points.csv 를 찾지 못함: {args.labels_root}")

    from squat.pose_estimation import PoseEstimator

    all_df = []
    for gt_path in gt_files:
        session_dir = gt_path.parent
        rel_session = session_dir.relative_to(args.labels_root)
        subject_id = session_dir.parent.name      # {CA##}/{세션}/3d_points.csv → CA##
        gt_angles = load_gt_angles(gt_path, leg)

        src_session = find_source_session(args.source_root, rel_session)
        if src_session is None:
            print(f"[skip] 원천 세션 없음: {rel_session}")
            continue

        for cam_dir in sorted(src_session.glob("camera*")):
            m = _CAM_RE.search(cam_dir.name)
            if not m:
                continue
            cam_idx = int(m.group(1))
            if cam_filter is not None and cam_idx not in cam_filter:
                continue
            avis = list((cam_dir / "video").glob("*.avi"))
            if not avis:
                continue
            view_type = "side" if cam_idx in side_set else "front"
            bounds = None
            if not args.full_clip:
                ann = session_dir / cam_dir.name / "video" / "annotation.json"
                bounds = read_segment(ann)

            pose = PoseEstimator(static_image_mode=False)
            try:
                df = extract_camera(
                    pose, avis[0], gt_angles, subject_id, view_type,
                    cam_dir.name, bounds, args.timestamp_step_ms,
                )
            finally:
                pose.close()
            if not df.empty:
                all_df.append(df)
                print(f"[{subject_id}] {rel_session}/{cam_dir.name} ({view_type}): {len(df)} rows")

    if not all_df:
        raise RuntimeError("추출된 feature row 가 없음.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat(all_df, ignore_index=True)
    out.to_csv(args.output, index=False)
    print(f"Saved features: {args.output}  ({len(out)} rows, {out['subject_id'].nunique()} subjects)")


if __name__ == "__main__":
    main()
