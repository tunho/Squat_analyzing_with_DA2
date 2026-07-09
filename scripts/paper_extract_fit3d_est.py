from __future__ import annotations

"""
FiT3D 데이터셋을 MediaPipe 외 추정기(NLF/RTMW3D/MotionBERT)로 추출 — 다도메인 표1용.

FiT3D 로더(load_gt_joints/extract_feature_rows/CAM_INFO/TIME_RANGES/SUBJECTS/FPS)는
paper_extract_fit3d 에서 그대로 재사용(동일 GT·동일 피처 수식 보장).
추론 부분만 각 추정기 어댑터로 교체. FiT3D는 GT 좌다리 → side='left'.

실행 venv는 추정기별로:
  nlf      → .venv-nlf
  rtmw3d   → .venv-mmpose
  motionbert → .venv-mmpose
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from squat.domain.types import Point3D
from squat.geometry.world_knee_ops import point_to_np

from paper_extract_fit3d import (
    CAM_INFO, FPS, SUBJECTS, TIME_RANGES,
    extract_feature_rows, load_gt_joints,
)


def build_estimator(kind: str, args):
    """추정기 객체 + (side->관절인덱스 dict) + 모드('frame'|'seq') 반환."""
    if kind == "nlf":
        from paper_extract_nlf import NLFEstimator, SMPL24_SIDE_JOINTS
        est = NLFEstimator(args.nlf_model, device=args.device)
        return est, SMPL24_SIDE_JOINTS, "frame"
    if kind == "rtmw3d":
        from paper_extract_rtmw3d import RTMW3DEstimator, COCO_SIDE_JOINTS
        est = RTMW3DEstimator(args.rtmpose3d_dir, args.det_config, args.det_ckpt,
                              args.pose_config, args.pose_ckpt, device=args.device)
        return est, COCO_SIDE_JOINTS, "frame"
    if kind == "motionbert":
        from paper_extract_motionbert import MotionBERTEstimator, H36M_SIDE_JOINTS
        est = MotionBERTEstimator(args.motionbert_dir, args.mb_config, args.mb_ckpt,
                                  args.det_config, args.det_ckpt, args.pose2d_config,
                                  args.pose2d_ckpt, device=args.device)
        return est, H36M_SIDE_JOINTS, "seq"
    raise ValueError(kind)


def _frame_data(joints, idx, frame_idx):
    hip = Point3D(*joints[idx["hip"]].tolist(), 1.0)
    knee = Point3D(*joints[idx["knee"]].tolist(), 1.0)
    ankle = Point3D(*joints[idx["ankle"]].tolist(), 1.0)
    shoulder = Point3D(*joints[idx["shoulder"]].tolist(), 1.0)
    h, k, a = point_to_np(hip), point_to_np(knee), point_to_np(ankle)
    return {"frame_index": frame_idx, "hip": hip, "knee": knee, "ankle": ankle,
            "shoulder": shoulder, "hip_y": hip.y,
            "thigh_len": float(np.linalg.norm(h - k)),
            "shank_len": float(np.linalg.norm(a - k))}


def extract_video_frame(est, side_idx, video_path, gt_joints, subject_id, cam_label,
                        view_type, start_frame, end_frame, calib_frames):
    """per-frame 추정기(NLF/RTMW3D)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return pd.DataFrame()
    max_frames = min(len(gt_joints), end_frame)
    idx = side_idx["left"]
    frames_data, fi = [], 0
    try:
        while fi < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if fi < start_frame:
                fi += 1; continue
            j = est.predict_joints3d(frame)
            if j is not None:
                frames_data.append(_frame_data(j, idx, fi))
            fi += 1
    finally:
        cap.release()
    rows = extract_feature_rows(frames_data, gt_joints, subject_id, cam_label, view_type,
                                calib_frames=calib_frames)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def extract_video_seq(est, side_idx, video_path, gt_joints, subject_id, cam_label,
                      view_type, start_frame, end_frame, calib_frames):
    """시퀀스 추정기(MotionBERT): 2D 수집 → lift → frames_data."""
    from paper_extract_motionbert import coco17_to_h36m17, normalize_screen
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return pd.DataFrame()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = min(len(gt_joints), end_frame)
    fidx, kpts, fi = [], [], 0
    try:
        while fi < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if fi < start_frame:
                fi += 1; continue
            k = est.get_2d(frame)
            if k is not None:
                fidx.append(fi); kpts.append(k)
            fi += 1
    finally:
        cap.release()
    if not kpts:
        return pd.DataFrame()
    h36m = coco17_to_h36m17(np.stack(kpts))
    pose3d = est.lift(normalize_screen(h36m, w, h))
    idx = side_idx["left"]
    frames_data = [_frame_data(pose3d[t], idx, fr) for t, fr in enumerate(fidx)]
    rows = extract_feature_rows(frames_data, gt_joints, subject_id, cam_label, view_type,
                                calib_frames=calib_frames)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(description="FiT3D extraction with NLF/RTMW3D/MotionBERT.")
    p.add_argument("--estimator", required=True, choices=["nlf", "rtmw3d", "motionbert"])
    p.add_argument("--dataset", type=Path, default=Path("../dataset/raw/fit3d/fit3d_train"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--subjects", nargs="*", default=SUBJECTS)
    p.add_argument("--calib-frames", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    # NLF
    p.add_argument("--nlf-model")
    # mmpose (rtmw3d/motionbert 공용 detector)
    p.add_argument("--rtmpose3d-dir"); p.add_argument("--det-config"); p.add_argument("--det-ckpt")
    p.add_argument("--pose-config"); p.add_argument("--pose-ckpt")
    # motionbert
    p.add_argument("--motionbert-dir"); p.add_argument("--mb-config"); p.add_argument("--mb-ckpt")
    p.add_argument("--pose2d-config"); p.add_argument("--pose2d-ckpt")
    args = p.parse_args()

    est, side_idx, mode = build_estimator(args.estimator, args)
    runner = extract_video_seq if mode == "seq" else extract_video_frame

    all_frames = []
    for subject in args.subjects:
        sub_dir = args.dataset / "train" / subject
        gt_path = sub_dir / "joints3d_25/squat.json"
        if not gt_path.exists():
            continue
        gt_joints = load_gt_joints(gt_path)
        subject_id = f"fit3d_{subject}"
        start_sec, end_sec = TIME_RANGES.get(subject, (0.0, len(gt_joints) / FPS))
        start_frame, end_frame = int(start_sec * FPS), int(end_sec * FPS)
        for cam_id, (cam_label, view_type) in CAM_INFO.items():
            video_path = sub_dir / f"videos/{cam_id}/squat.mp4"
            if not video_path.exists():
                continue
            df = runner(est, side_idx, video_path, gt_joints, subject_id, cam_label,
                        view_type, start_frame, end_frame, args.calib_frames)
            if not df.empty:
                all_frames.append(df)
                print(f"  {subject}/{cam_id} ({cam_label}): {len(df)} frames", flush=True)

    if not all_frames:
        raise RuntimeError("No features extracted (FiT3D estimator).")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_frames, ignore_index=True).to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
