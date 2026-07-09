from __future__ import annotations

"""
SUMediPose 데이터셋을 MediaPipe 외 추정기(NLF/RTMW3D/MotionBERT)로 추출 — 다도메인 표1용.

SUMediPose 로더(zip 안 jpg + Vicon GT) 구조는 sumedipose_extract 재사용.
추론 부분만 추정기로 교체. GT 좌다리(LHJC/LKJC/LANK) → side='left'.
estimator 구성은 paper_extract_fit3d_est.build_estimator 재사용.

실행 venv: nlf→.venv-nlf, rtmw3d/motionbert→.venv-mmpose.
"""

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from squat.domain.types import Point3D
from paper_extract import extract_feature_rows, calib_med_total
from sumedipose_extract import vicon_knee_angle, GT_JOINTS, _F
from paper_extract_fit3d_est import build_estimator


def _make_F(joints, idx, i):
    f = _F(); f.frame_index = i
    f.mp_world_hip = Point3D(*joints[idx["hip"]].tolist(), 1.0)
    f.mp_world_knee = Point3D(*joints[idx["knee"]].tolist(), 1.0)
    f.mp_world_ankle = Point3D(*joints[idx["ankle"]].tolist(), 1.0)
    f.mp_world_shoulder = Point3D(*joints[idx["shoulder"]].tolist(), 1.0)
    f.shoulder = None
    return f


def process_take_est(est, side_idx, mode, frames_zip, take_json, subject, camera, take, calib_frames):
    tmp = Path(tempfile.mkdtemp(prefix="sume_est_"))
    try:
        with zipfile.ZipFile(frames_zip) as z:
            z.extractall(tmp)
        recs = [fr for fr in take_json if "path" in fr and "xyz" in fr]
        recs.sort(key=lambda fr: fr.get("frame_num", 0))
        ids = recs[0]["point_ids"] if recs else None
        idx = side_idx["left"]
        sframes, gt = [], {}

        if mode == "seq":  # MotionBERT
            from paper_extract_motionbert import coco17_to_h36m17, normalize_screen
            imgs, gts, ii = [], [], []
            for i, fr in enumerate(recs):
                jpg = tmp / Path(fr["path"]).name
                if not jpg.exists():
                    continue
                img = cv2.imread(str(jpg))
                if img is None:
                    continue
                k = est.get_2d(img)
                if k is None:
                    continue
                imgs.append(k); gts.append(vicon_knee_angle(fr["xyz"], ids)); ii.append(i)
                _h, _w = img.shape[:2]
            if not imgs:
                return None
            h36m = coco17_to_h36m17(np.stack(imgs))
            pose3d = est.lift(normalize_screen(h36m, _w, _h))
            for t, i in enumerate(ii):
                sframes.append(_make_F(pose3d[t], idx, i)); gt[i] = gts[t]
        else:  # per-frame: NLF / RTMW3D
            for i, fr in enumerate(recs):
                jpg = tmp / Path(fr["path"]).name
                if not jpg.exists():
                    continue
                img = cv2.imread(str(jpg))
                if img is None:
                    continue
                j = est.predict_joints3d(img)
                if j is None:
                    continue
                sframes.append(_make_F(j, idx, i)); gt[i] = vicon_knee_angle(fr["xyz"], ids)

        if not sframes:
            return None
        med = calib_med_total(sframes, calib_frames)
        rows = extract_feature_rows(sframes, f"{camera}", str(subject), med_total=med)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["view_type"] = "vicon"; df["camera"] = camera
        df["gt_angle"] = df["frame_index"].map(gt)
        df["ex_type"] = "squat"; df["take"] = take
        return df.dropna(subset=["gt_angle"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description="SUMediPose extraction with NLF/RTMW3D/MotionBERT.")
    p.add_argument("--estimator", required=True, choices=["nlf", "rtmw3d", "motionbert"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cameras", nargs="*", default=[f"C{i}" for i in range(1, 7)])
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--calib-frames", type=int, default=15)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nlf-model")
    p.add_argument("--rtmpose3d-dir"); p.add_argument("--det-config"); p.add_argument("--det-ckpt")
    p.add_argument("--pose-config"); p.add_argument("--pose-ckpt")
    p.add_argument("--motionbert-dir"); p.add_argument("--mb-config"); p.add_argument("--mb-ckpt")
    p.add_argument("--pose2d-config"); p.add_argument("--pose2d-ckpt")
    args = p.parse_args()

    est, side_idx, mode = build_estimator(args.estimator, args)

    all_df, done = [], 0
    for cam in args.cameras:
        ezdir = args.root / "internal_data" / cam
        if not ezdir.exists():
            continue
        for sz in sorted(ezdir.glob("S*.zip")):
            subject = sz.stem
            if args.subjects and subject not in args.subjects:
                continue
            with zipfile.ZipFile(sz) as z:
                jsons = [n for n in z.namelist() if re.match(rf"^{subject}A3D\d+\.json$", n)]
                for jn in sorted(jsons):
                    take = re.search(r"A3D(\d+)", jn).group(1)
                    fzip = args.root / "frames" / cam / subject / f"{cam}{subject}A3D{take}.zip"
                    if not fzip.exists():
                        continue
                    tj = json.loads(z.read(jn))
                    df = process_take_est(est, side_idx, mode, fzip, tj, subject, cam, take, args.calib_frames)
                    if df is not None and not df.empty:
                        all_df.append(df); done += 1
                        print(f"[{done}] {cam}/{subject}/A3D{take}: {len(df)}행", flush=True)
    if not all_df:
        raise RuntimeError("추출 결과 없음 (SUMediPose estimator)")
    out = pd.concat(all_df, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\n저장: {args.output} ({len(out):,}행, {out['subject_id'].nunique()}명, {done} takes)", flush=True)


if __name__ == "__main__":
    main()
