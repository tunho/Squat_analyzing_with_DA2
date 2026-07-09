from __future__ import annotations

"""
AIHub 213 스쿼트를 MediaPipe 외 추정기(NLF/RTMW3D/MotionBERT)로 추출 — 다도메인 표1 마지막 도메인.

aihub_extract_raw_dump 의 로더(load_gt/read_segment) 재사용. 영상은 선별추출본
dataset/raw/aihub_squat/videos_val/스쿼트/.../camera{N}/video/*.avi (카메라당 avis[0]).
라벨 1206세션 중 영상 있는 세션만 처리(교집합). 좌다리(gt_knee_left) 사용 — 타 도메인과 일치.

출력 = 동일 CSV 스키마(extract_feature_rows). 실행 venv: nlf→.venv-nlf, rtmw3d/motionbert→.venv-mmpose.
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from squat.domain.types import Point3D, SquatFrame
from paper_extract import extract_feature_rows, calib_med_total
from aihub_extract_raw_dump import load_gt, _CAM_RE
from paper_extract_fit3d_est import build_estimator

SIDE_SET = {2, 3}  # side cameras


def _sf(joints, idx, fi):
    hip = Point3D(*joints[idx["hip"]].tolist(), 1.0)
    knee = Point3D(*joints[idx["knee"]].tolist(), 1.0)
    ankle = Point3D(*joints[idx["ankle"]].tolist(), 1.0)
    sh = Point3D(*joints[idx["shoulder"]].tolist(), 1.0)
    return SquatFrame(frame_index=fi, side="left", hip=hip, knee=knee, ankle=ankle,
                      shoulder=sh, mp_world_hip=hip, mp_world_knee=knee,
                      mp_world_ankle=ankle, mp_world_shoulder=sh)


def process_camera(est, side_idx, mode, avi_path, gt, idx):
    """카메라 1개(avis[0]) → SquatFrame 리스트 + gt(frame→left angle)."""
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return [], {}
    frames, gtmap = [], {}
    if mode == "seq":
        from paper_extract_motionbert import coco17_to_h36m17, normalize_screen
        kpts, fis = [], []
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            k = est.get_2d(frame)
            if k is not None and fi in gt:
                kpts.append(k); fis.append(fi)
            fi += 1
        cap.release()
        if not kpts:
            return [], {}
        pose3d = est.lift(normalize_screen(coco17_to_h36m17(np.stack(kpts)), w, h))
        for t, fi in enumerate(fis):
            frames.append(_sf(pose3d[t], idx, fi)); gtmap[fi] = gt[fi][0]  # left
    else:
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if fi in gt:
                j = est.predict_joints3d(frame)
                if j is not None:
                    frames.append(_sf(j, idx, fi)); gtmap[fi] = gt[fi][0]  # left
            fi += 1
        cap.release()
    return frames, gtmap


def main():
    p = argparse.ArgumentParser(description="AIHub extraction with NLF/RTMW3D/MotionBERT.")
    p.add_argument("--estimator", required=True, choices=["nlf", "rtmw3d", "motionbert"])
    p.add_argument("--labels-root", type=Path, default=Path("../dataset/raw/aihub_squat/labels"))
    p.add_argument("--videos-root", type=Path, default=Path("../dataset/raw/aihub_squat/videos_val"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calib-frames", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nlf-model")
    p.add_argument("--rtmpose3d-dir"); p.add_argument("--det-config"); p.add_argument("--det-ckpt")
    p.add_argument("--pose-config"); p.add_argument("--pose-ckpt")
    p.add_argument("--motionbert-dir"); p.add_argument("--mb-config"); p.add_argument("--mb-ckpt")
    p.add_argument("--pose2d-config"); p.add_argument("--pose2d-ckpt")
    args = p.parse_args()

    est, side_idx, mode = build_estimator(args.estimator, args)
    idx = side_idx["left"]

    gt_files = sorted(args.labels_root.rglob("3d_points.csv"))
    gt_files = [g for g in gt_files if "스쿼트" in str(g)]

    all_df, done = [], 0
    for gt_path in gt_files:
        rel = gt_path.parent.relative_to(args.labels_root)
        vdir = args.videos_root / rel
        if not vdir.exists():
            continue  # 이 세션 영상 없음
        gt = load_gt(gt_path)
        subject_id = gt_path.parent.parent.name
        for cam_dir in sorted(vdir.glob("camera*")):
            m = _CAM_RE.search(cam_dir.name)
            if not m:
                continue
            cam_idx = int(m.group(1))
            avis = sorted((cam_dir / "video").glob("*.avi"))
            if not avis:
                continue
            frames, gtmap = process_camera(est, side_idx, mode, avis[0], gt, idx)
            if not frames:
                continue
            rows = extract_feature_rows(frames, cam_dir.name, subject_id,
                                        med_total=calib_med_total(frames, args.calib_frames))
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["view_type"] = "side" if cam_idx in SIDE_SET else "front"
            df["camera"] = cam_dir.name
            df["gt_angle"] = df["frame_index"].map(gtmap)
            df = df.dropna(subset=["gt_angle"])
            if not df.empty:
                all_df.append(df); done += 1
                print(f"[{done}] {rel}/{cam_dir.name}: {len(df)}행", flush=True)

    if not all_df:
        raise RuntimeError("추출 결과 없음 (AIHub estimator)")
    out = pd.concat(all_df, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\n저장: {args.output} ({len(out):,}행, {out.subject_id.nunique()}명, {done} 카메라세션)", flush=True)


if __name__ == "__main__":
    main()
