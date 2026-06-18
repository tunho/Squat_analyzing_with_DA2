"""raw 덤프(parquet) → 학습용 피처 CSV. 영상 재추출 없이 오프라인 파생.

검증된 paper_extract.extract_feature_rows 를 그대로 재사용(동일 수식 보장).
입력만 MediaPipe 대신 덤프의 world 랜드마크(L: hip=23,knee=25,ankle=27,shoulder=11)로 공급.
calib-frames 로 med_total 정규화 방식 선택(15=causal/기본, 0=offline 시퀀스중앙값).
출력 스키마 = features_aihub_squat*.csv 와 동일 → paper_train_loso 그대로 사용.

사용:
  PYTHONPATH=../src ../.venv-paper/bin/python aihub_features_from_dump.py \
    --dump-dir ../experiments/paper/aihub_raw_dump \
    --output ../experiments/paper/features_aihub_squat_causal.csv --calib-frames 15
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import pandas as pd

from squat.domain.types import Point3D
from paper_extract import extract_feature_rows, calib_med_total

LHIP, LKNEE, LANKLE, LSHO = 23, 25, 27, 11


class _F:
    __slots__ = ("frame_index", "mp_world_hip", "mp_world_knee",
                 "mp_world_ankle", "mp_world_shoulder", "shoulder")


def _pt(r, j):
    return Point3D(float(getattr(r, f"w{j}_x")), float(getattr(r, f"w{j}_y")),
                   float(getattr(r, f"w{j}_z")), float(getattr(r, f"w{j}_v")))


def build_frames(cam_df: pd.DataFrame) -> list:
    frames = []
    for r in cam_df.sort_values("frame_index").itertuples(index=False):
        f = _F()
        f.frame_index = int(r.frame_index)
        f.mp_world_hip = _pt(r, LHIP)
        f.mp_world_knee = _pt(r, LKNEE)
        f.mp_world_ankle = _pt(r, LANKLE)
        f.mp_world_shoulder = _pt(r, LSHO)
        f.shoulder = None
        frames.append(f)
    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calib-frames", type=int, default=15)
    p.add_argument("--frame-step", type=int, default=1,
                   help="프레임 다운샘플(예: 3=30fps→10fps). 인접프레임 중복 제거, 학습 가속.")
    args = p.parse_args()

    files = sorted(glob.glob(str(args.dump_dir / "*.parquet")))
    if not files:
        raise RuntimeError(f"parquet 없음: {args.dump_dir}")

    all_rows = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        subject_id = str(df["subject_id"].iloc[0])
        for cam_idx, g in df.groupby("cam_idx"):
            g = g.sort_values("frame_index")
            if args.frame_step > 1:
                g = g.iloc[:: args.frame_step]
            frames = build_frames(g)
            if not frames:
                continue
            med = calib_med_total(frames, args.calib_frames)  # None → offline
            rows = extract_feature_rows(frames, f"cam{cam_idx}", subject_id, med_total=med)
            if not rows:
                continue
            out = pd.DataFrame(rows)
            out["view_type"] = g["view_type"].iloc[0]
            out["camera"] = g["camera"].iloc[0]
            gt = dict(zip(g["frame_index"], g["gt_knee_left"]))
            out["gt_angle"] = out["frame_index"].map(gt)
            out["ex_type"] = g["ex_type"].iloc[0]
            all_rows.append(out.dropna(subset=["gt_angle"]))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} 세션 처리")

    final = pd.concat(all_rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    print(f"저장: {args.output}  ({len(final):,}행, {final['subject_id'].nunique()} subject, "
          f"calib={args.calib_frames})")


if __name__ == "__main__":
    main()
