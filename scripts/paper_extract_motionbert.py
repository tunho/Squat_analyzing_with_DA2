from __future__ import annotations

"""
MotionBERT (ICCV 2023, 2D→3D lifting) 추정기로 feature CSV 추출 — 추정기 비종속(표1).

다른 어댑터(NLF/RTMW3D)와 달리 MotionBERT는 '시퀀스 lifting'이라 영상 전체를 모아 처리:
  1) mmpose(rtmdet+rtmpose-2d)로 프레임별 COCO-17 2D 키포인트
  2) COCO-17 → H36M-17 변환 + normalize_screen_coordinates
  3) clip_len(243) 단위로 MotionBERT lifting → 3D H36M-17
  4) side별 hip/knee/ankle/shoulder → SquatFrame → extract_feature_rows 재사용(동일 CSV 스키마)

H36M-17 인덱스(halpe2h36m 기준): L hip4/knee5/ankle6/sh11, R hip1/knee2/ankle3/sh14.
무릎각은 평행이동·스케일 불변이라 root-relative 3D 그대로 사용.

실행 venv: .venv-mmpose (torch+mmpose+easydict). PYTHONPATH 에 src + MotionBERT 경로.
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


H36M_SIDE_JOINTS = {
    "left":  {"hip": 4, "knee": 5, "ankle": 6, "shoulder": 11},
    "right": {"hip": 1, "knee": 2, "ankle": 3, "shoulder": 14},
}


def gt_side(seg_df, subject_id, exercise):
    hip_idx, _, _ = get_leg_indices(seg_df, subject_id, exercise)
    return "right" if hip_idx == 21 else "left"


def coco17_to_h36m17(coco: np.ndarray) -> np.ndarray:
    """coco (T,17,3 [x,y,conf]) → h36m (T,17,3). 파생관절은 평균/최소conf."""
    T = coco.shape[0]
    h = np.zeros((T, 17, 3), dtype=np.float32)

    def mid(a, b):
        m = (coco[:, a] + coco[:, b]) * 0.5
        m[:, 2] = np.minimum(coco[:, a, 2], coco[:, b, 2])
        return m

    h[:, 0] = mid(11, 12)              # Hip(root)
    h[:, 1] = coco[:, 12]              # RHip
    h[:, 2] = coco[:, 14]              # RKnee
    h[:, 3] = coco[:, 16]              # RAnkle
    h[:, 4] = coco[:, 11]             # LHip
    h[:, 5] = coco[:, 13]             # LKnee
    h[:, 6] = coco[:, 15]             # LAnkle
    h[:, 8] = mid(5, 6)               # Thorax(neck)
    h[:, 7] = (h[:, 0] + h[:, 8]) * 0.5   # Spine
    h[:, 9] = coco[:, 0]              # Nose
    h[:, 10] = mid(1, 2)             # Head(eyes avg) 근사
    h[:, 11] = coco[:, 5]            # LShoulder
    h[:, 12] = coco[:, 7]            # LElbow
    h[:, 13] = coco[:, 9]            # LWrist
    h[:, 14] = coco[:, 6]            # RShoulder
    h[:, 15] = coco[:, 8]            # RElbow
    h[:, 16] = coco[:, 10]           # RWrist
    return h


def normalize_screen(kp: np.ndarray, w: int, h: int) -> np.ndarray:
    out = kp.copy()
    out[..., 0] = kp[..., 0] / w * 2 - 1
    out[..., 1] = kp[..., 1] / w * 2 - h / w
    return out  # conf 채널은 그대로


class MotionBERTEstimator:
    def __init__(self, motionbert_dir, mb_config, mb_ckpt,
                 det_config, det_ckpt, pose2d_config, pose2d_ckpt,
                 device="cuda:0", clip_len=243, det_score_thr=0.3):
        # --- 2D: mmpose ---
        from mmpose.apis import inference_topdown, init_model
        from mmpose.utils import adapt_mmdet_pipeline
        from mmdet.apis import inference_detector, init_detector
        self._inf_top = inference_topdown
        self._inf_det = inference_detector
        self.det_score_thr = det_score_thr
        self.detector = init_detector(det_config, det_ckpt, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self.pose2d = init_model(pose2d_config, pose2d_ckpt, device=device)
        # --- lifting: MotionBERT ---
        if motionbert_dir not in sys.path:
            sys.path.insert(0, motionbert_dir)
        import torch
        from lib.utils.tools import get_config
        from lib.utils.learning import load_backbone
        self.torch = torch
        self.device = device
        self.clip_len = clip_len
        args = get_config(mb_config)
        model = load_backbone(args)
        ckpt = torch.load(mb_ckpt, map_location="cpu")
        sd = ckpt.get("model_pos", ckpt)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        self.model = model.to(device).eval()

    def get_2d(self, frame_bgr):
        det = self._inf_det(self.detector, frame_bgr)
        pi = det.pred_instances.cpu().numpy()
        keep = (pi.labels == 0) & (pi.scores > self.det_score_thr)
        bboxes = pi.bboxes[keep]
        if len(bboxes) == 0:
            return None
        bboxes = bboxes[np.argsort(pi.scores[keep])[::-1][:1]]
        res = self._inf_top(self.pose2d, frame_bgr, bboxes)
        if not res:
            return None
        kp = res[0].pred_instances.keypoints[0]          # (17,2)
        sc = res[0].pred_instances.keypoint_scores[0]    # (17,)
        return np.concatenate([kp, sc[:, None]], axis=1).astype(np.float32)  # (17,3)

    def lift(self, h36m_seq_norm: np.ndarray) -> np.ndarray:
        """(T,17,3) 정규화 2D → (T,17,3) 3D. clip_len 단위 비중첩 처리(마지막 패딩)."""
        torch = self.torch
        T = h36m_seq_norm.shape[0]
        C = self.clip_len
        outs = []
        for s in range(0, T, C):
            clip = h36m_seq_norm[s:s + C]
            n = len(clip)
            if n < C:  # 패딩(마지막 프레임 반복)
                clip = np.concatenate([clip, np.repeat(clip[-1:], C - n, axis=0)], axis=0)
            x = torch.from_numpy(clip[None]).float().to(self.device)  # (1,C,17,3)
            with torch.inference_mode():
                y = self.model(x)[0].cpu().numpy()  # (C,17,3)
            outs.append(y[:n])
        return np.concatenate(outs, axis=0)  # (T,17,3)


def extract_video(est: MotionBERTEstimator, video_path: Path, gt_angles, subject_id, side, calib_frames=0):
    if not gt_angles:
        return pd.DataFrame()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return pd.DataFrame()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start, end = min(gt_angles), max(gt_angles)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    fidx, kpts2d, fi = [], [], start
    try:
        while fi <= end:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            k = est.get_2d(frame)
            if k is not None:
                fidx.append(fi); kpts2d.append(k)
            fi += 1
    finally:
        cap.release()
    if not kpts2d:
        return pd.DataFrame()

    coco = np.stack(kpts2d)                      # (T,17,3)
    h36m = coco17_to_h36m17(coco)                # (T,17,3)
    h36m_n = normalize_screen(h36m, w, h)
    pose3d = est.lift(h36m_n)                     # (T,17,3)

    idx = H36M_SIDE_JOINTS[side]
    frames = []
    for t, fr in enumerate(fidx):
        j = pose3d[t]
        hip = Point3D(*j[idx["hip"]].tolist(), 1.0)
        knee = Point3D(*j[idx["knee"]].tolist(), 1.0)
        ankle = Point3D(*j[idx["ankle"]].tolist(), 1.0)
        sh = Point3D(*j[idx["shoulder"]].tolist(), 1.0)
        frames.append(SquatFrame(frame_index=fr, side=side, hip=hip, knee=knee, ankle=ankle,
                                 shoulder=sh, mp_world_hip=hip, mp_world_knee=knee,
                                 mp_world_ankle=ankle, mp_world_shoulder=sh))
    df = pd.DataFrame(extract_feature_rows(frames, video_path.name, subject_id,
                                           med_total=calib_med_total(frames, calib_frames)))
    if df.empty:
        return df
    df["gt_angle"] = df["frame_index"].map(gt_angles)
    return df.dropna(subset=["gt_angle"]).copy()


def main():
    p = argparse.ArgumentParser(description="Extract MotionBERT features (same schema as paper_extract.py).")
    p.add_argument("--dataset", type=Path, default=Path("dataset/raw/REHAB24-6"))
    p.add_argument("--exercise", default="Ex6")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--motionbert-dir", required=True)
    p.add_argument("--mb-config", required=True)
    p.add_argument("--mb-ckpt", required=True)
    p.add_argument("--det-config", required=True)
    p.add_argument("--det-ckpt", required=True)
    p.add_argument("--pose2d-config", required=True)
    p.add_argument("--pose2d-ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--view-suffixes", nargs="*", default=DEFAULT_VIEW_SUFFIXES)
    p.add_argument("--calib-frames", type=int, default=0)
    args = p.parse_args()

    video_dir = args.dataset / "videos" / args.exercise
    gt_dir = args.dataset / "3d_joints" / args.exercise
    subjects = args.subjects or discover_subjects(gt_dir)
    seg_df = load_segmentation_table(args.dataset)

    est = MotionBERTEstimator(args.motionbert_dir, args.mb_config, args.mb_ckpt,
                              args.det_config, args.det_ckpt, args.pose2d_config, args.pose2d_ckpt,
                              device=args.device)

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
            df = extract_video(est, video_path, gt_angles, subject_id, side, calib_frames=args.calib_frames)
            if not df.empty:
                all_frames.append(df)
                print(f"Extracted {len(df)} rows from {video_path.name} (side={side})", flush=True)

    if not all_frames:
        raise RuntimeError("No feature rows were extracted (MotionBERT).")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_frames, ignore_index=True).to_csv(args.output, index=False)
    print(f"Saved MotionBERT features: {args.output}")


if __name__ == "__main__":
    main()
