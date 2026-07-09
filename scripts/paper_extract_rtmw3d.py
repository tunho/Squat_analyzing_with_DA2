from __future__ import annotations

"""
RTMW3D (MMPose, 2024) 추정기로 feature CSV 추출 — 추정기 비종속성(표1).

paper_extract_nlf.py 와 동일 설계: RTMW3D의 3D 관절(COCO-wholebody)을
SquatFrame.mp_world_* 에 채우고 paper_extract.extract_feature_rows 재사용
→ MediaPipe/NLF와 동일 CSV 스키마 → 이후 LOSO/보정기 그대로.

추론: rtmdet(person) → inference_topdown(rtmw3d) → keypoints[133,3].
COCO-17 body 인덱스: L sh5/hip11/knee13/ankle15, R sh6/hip12/knee14/ankle16.
무릎각은 평행이동·스케일 불변이라 RTMW3D 출력 좌표 그대로 사용.

실행 venv: .venv-mmpose (torch2.1.2+cu118, mmcv/mmdet/mmpose).
PYTHONPATH 에 src + mmpose_repo/projects/rtmpose3d 필요.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from squat.domain.types import Point3D, SquatFrame

from paper_extract import (
    DEFAULT_VIEW_SUFFIXES,
    calib_med_total,
    discover_subjects,
    extract_feature_rows,
    get_leg_indices,
    load_gt_angles,
    load_segmentation_table,
)


COCO_SIDE_JOINTS = {
    "left":  {"hip": 11, "knee": 13, "ankle": 15, "shoulder": 5},
    "right": {"hip": 12, "knee": 14, "ankle": 16, "shoulder": 6},
}


def gt_side(seg_df, subject_id: str, exercise: str) -> str:
    hip_idx, _, _ = get_leg_indices(seg_df, subject_id, exercise)
    return "right" if hip_idx == 21 else "left"


class RTMW3DEstimator:
    def __init__(self, rtmpose3d_dir: str, det_config: str, det_ckpt: str,
                 pose_config: str, pose_ckpt: str, device: str = "cuda:0",
                 det_score_thr: float = 0.3):
        if rtmpose3d_dir not in sys.path:
            sys.path.insert(0, rtmpose3d_dir)
        import rtmpose3d  # noqa: F401  ← 커스텀 head/codec/estimator 등록
        from mmpose.apis import inference_topdown, init_model
        from mmpose.utils import adapt_mmdet_pipeline
        from mmdet.apis import inference_detector, init_detector

        self._inference_topdown = inference_topdown
        self._inference_detector = inference_detector
        self.det_score_thr = det_score_thr

        self.detector = init_detector(det_config, det_ckpt, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self.pose = init_model(pose_config, pose_ckpt, device=device)

    def predict_joints3d(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        det = self._inference_detector(self.detector, frame_bgr)
        pi = det.pred_instances.cpu().numpy()
        keep = (pi.labels == 0) & (pi.scores > self.det_score_thr)
        bboxes = pi.bboxes[keep]
        if len(bboxes) == 0:
            return None
        # 최고 점수 1인 선택
        scores = pi.scores[keep]
        bboxes = bboxes[np.argsort(scores)[::-1][:1]]
        res = self._inference_topdown(self.pose, frame_bgr, bboxes)
        if not res:
            return None
        kp = res[0].pred_instances.keypoints
        if kp.ndim == 4:
            kp = np.squeeze(kp, axis=1)
        return np.asarray(kp[0])  # [133, 3]


def joints_to_squatframe(joints3d: np.ndarray, side: str, frame_index: int) -> SquatFrame | None:
    idx = COCO_SIDE_JOINTS[side]
    try:
        hip = Point3D(*joints3d[idx["hip"]].tolist(), 1.0)
        knee = Point3D(*joints3d[idx["knee"]].tolist(), 1.0)
        ankle = Point3D(*joints3d[idx["ankle"]].tolist(), 1.0)
        shoulder = Point3D(*joints3d[idx["shoulder"]].tolist(), 1.0)
    except (IndexError, TypeError):
        return None
    return SquatFrame(
        frame_index=frame_index, side=side,
        hip=hip, knee=knee, ankle=ankle, shoulder=shoulder,
        mp_world_hip=hip, mp_world_knee=knee, mp_world_ankle=ankle, mp_world_shoulder=shoulder,
    )


def extract_video(estimator, video_path: Path, gt_angles, subject_id, side, calib_frames=0):
    if not gt_angles:
        return pd.DataFrame()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return pd.DataFrame()
    start, end = min(gt_angles), max(gt_angles)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, fi = [], start
    try:
        while fi <= end:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            j = estimator.predict_joints3d(frame)
            if j is not None:
                sf = joints_to_squatframe(j, side, fi)
                if sf is not None:
                    frames.append(sf)
            fi += 1
    finally:
        cap.release()
    df = pd.DataFrame(extract_feature_rows(frames, video_path.name, subject_id,
                                           med_total=calib_med_total(frames, calib_frames)))
    if df.empty:
        return df
    df["gt_angle"] = df["frame_index"].map(gt_angles)
    return df.dropna(subset=["gt_angle"]).copy()


def main():
    p = argparse.ArgumentParser(description="Extract RTMW3D features (same schema as paper_extract.py).")
    p.add_argument("--dataset", type=Path, default=Path("dataset/raw/REHAB24-6"))
    p.add_argument("--exercise", default="Ex6")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rtmpose3d-dir", required=True, help="mmpose_repo/projects/rtmpose3d 경로")
    p.add_argument("--det-config", required=True)
    p.add_argument("--det-ckpt", required=True)
    p.add_argument("--pose-config", required=True)
    p.add_argument("--pose-ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--view-suffixes", nargs="*", default=DEFAULT_VIEW_SUFFIXES)
    p.add_argument("--calib-frames", type=int, default=0)
    args = p.parse_args()

    video_dir = args.dataset / "videos" / args.exercise
    gt_dir = args.dataset / "3d_joints" / args.exercise
    subjects = args.subjects or discover_subjects(gt_dir)
    seg_df = load_segmentation_table(args.dataset)

    estimator = RTMW3DEstimator(args.rtmpose3d_dir, args.det_config, args.det_ckpt,
                                args.pose_config, args.pose_ckpt, device=args.device)

    all_frames = []
    for subject_id in subjects:
        gt_path = gt_dir / f"{subject_id}-30fps.npy"
        if not gt_path.exists():
            continue
        gt_angles = load_gt_angles(gt_path, get_leg_indices(seg_df, subject_id, args.exercise))
        side = gt_side(seg_df, subject_id, args.exercise)
        for suffix in args.view_suffixes:
            video_path = video_dir / f"{subject_id}-{suffix}.mp4"
            if not video_path.exists():
                continue
            df = extract_video(estimator, video_path, gt_angles, subject_id, side,
                               calib_frames=args.calib_frames)
            if not df.empty:
                all_frames.append(df)
                print(f"Extracted {len(df)} rows from {video_path.name} (side={side})", flush=True)

    if not all_frames:
        raise RuntimeError("No feature rows were extracted (RTMW3D).")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_frames, ignore_index=True).to_csv(args.output, index=False)
    print(f"Saved RTMW3D features: {args.output}")


if __name__ == "__main__":
    main()
