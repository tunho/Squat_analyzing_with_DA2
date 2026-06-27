"""AI Hub 213 CrossFit → '두터운' raw 랜드마크 덤프 (재추출 불필요용 마스터 아티팩트).

기존 paper_extract_aihub.py 는 축소·정규화된 피처만 저장한다(좌측 hip/knee/ankle+shoulder,
med_total 정규화 박제). 그래서 다른 관절·다른 정규화·새 피처가 필요하면 영상 재추출이 필요했다.

이 스크립트는 영상을 한 번만 돌려 '모든 것'을 비정규화 raw 로 저장한다:
  - MediaPipe world 33관절 (x,y,z,visibility)  → w{ID}_{x,y,z,v}
  - MediaPipe image 33관절 (x,y,z,visibility)  → p{ID}_{x,y,z,v}
  - GT 무릎각 좌/우 (3d_points.csv 에서 계산)
  - 메타: subject, camera, cam_idx, view_type, frame_index, ex_type/error_type/level,
          ann_start, ann_end (annotation 구간 → 오프라인 재절단용), full-clip 전 프레임

여기서 어떤 피처셋·정규화든 오프라인 파생 가능 → 영상 영구 재방문 불필요(추정기 교체 제외).
세션별 parquet 1개 (재개 가능: 이미 있으면 skip). 학습 측은 디렉토리를 pyarrow dataset 으로 읽거나 concat.

사용:
  PYTHONPATH=../src ../.venv-paper/bin/python aihub_extract_raw_dump.py \
    --labels-root ../dataset/raw/aihub_squat/labels \
    --source-root <임시 영상 루트> \
    --output-dir ../experiments/paper/aihub_raw_dump \
    --exercise-filter 스쿼트
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# PoseEstimator(MediaPipe)는 main()에서 lazy import — 추정기 venv에서 load_gt 등 재사용 위함.
_CAM_RE = re.compile(r"camera(\d+)")
GT_LEFT = ("LHip", "LKnee", "LAnkle")
GT_RIGHT = ("RHip", "RKnee", "RAnkle")
NUM_LM = 33  # MediaPipe Pose


def _world_landmarks(result):
    """result.pose_world_landmarks[0] → 33개 dict. 검출 실패 시 None."""
    if result is None or not getattr(result, "pose_world_landmarks", None):
        return None
    out = []
    for lm in result.pose_world_landmarks[0]:
        vis = getattr(lm, "visibility", None)
        pres = getattr(lm, "presence", None)
        out.append((float(lm.x), float(lm.y), float(lm.z),
                    float(vis if vis is not None else (pres or 0.0))))
    return out


def _angle(a, b, c) -> float:
    """관절각 at b (deg), a-b-c. 각 점은 (x,y,z)."""
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < 1e-9 or nbc < 1e-9:
        return float("nan")
    cosv = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return math.degrees(math.acos(cosv))


def load_gt(csv_path: Path) -> dict[int, tuple[float, float]]:
    """3d_points.csv → {frame_idx: (gt_knee_left, gt_knee_right)}."""
    df = pd.read_csv(csv_path)

    def pt(row, j):
        return (float(row[f"{j}_x"]), float(row[f"{j}_y"]), float(row[f"{j}_z"]))

    out: dict[int, tuple[float, float]] = {}
    for i, row in df.iterrows():
        m = re.match(r"(\d+)", str(row.get("image_filename", i)))
        idx = int(m.group(1)) if m else int(i)
        gl = _angle(pt(row, GT_LEFT[0]), pt(row, GT_LEFT[1]), pt(row, GT_LEFT[2]))
        gr = _angle(pt(row, GT_RIGHT[0]), pt(row, GT_RIGHT[1]), pt(row, GT_RIGHT[2]))
        out[idx] = (gl, gr)
    return out


def read_segment(ann_path: Path):
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        ann = data.get("annotations") or []
        if ann:
            return int(ann[0].get("start_frame", -1)), int(ann[0].get("end_frame", -1))
    except Exception:
        pass
    return -1, -1


def dump_camera(pose, avi_path: Path, gt: dict, meta: dict, step_ms: int) -> list[dict]:
    import cv2

    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return []
    rows = []
    fidx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            lm, result = pose.extract_keypoints_only(frame, frame_timestamp_ms=fidx * step_ms)
            world = _world_landmarks(result)
            if lm and world:
                gl, gr = gt.get(fidx, (float("nan"), float("nan")))
                row = dict(meta)
                row["frame_index"] = fidx
                row["gt_knee_left"] = gl
                row["gt_knee_right"] = gr
                for j in range(NUM_LM):
                    wx, wy, wz, wv = world[j]
                    row[f"w{j}_x"], row[f"w{j}_y"], row[f"w{j}_z"], row[f"w{j}_v"] = wx, wy, wz, wv
                    p = lm[j]
                    row[f"p{j}_x"], row[f"p{j}_y"], row[f"p{j}_z"], row[f"p{j}_v"] = (
                        p["x"], p["y"], p["z"], p["visibility"])
                rows.append(row)
            fidx += 1
    finally:
        cap.release()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="AI Hub 213 → raw landmark 마스터 덤프 (세션별 parquet).")
    p.add_argument("--labels-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--exercise-filter", default=None)
    p.add_argument("--side-cameras", nargs="*", type=int, default=[2, 3])
    p.add_argument("--cameras", nargs="*", type=int, default=None)
    p.add_argument("--timestamp-step-ms", type=int, default=33)
    p.add_argument("--max-sessions", type=int, default=None)
    args = p.parse_args()

    side_set = set(args.side_cameras)
    cam_filter = set(args.cameras) if args.cameras is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(args.labels_root.rglob("3d_points.csv"))
    if args.exercise_filter:
        gt_files = [g for g in gt_files if args.exercise_filter in str(g)]
    if args.max_sessions:
        gt_files = gt_files[: args.max_sessions]
    if not gt_files:
        raise RuntimeError(f"3d_points.csv 없음: {args.labels_root}")

    done = skipped = 0
    for gt_path in gt_files:
        session_dir = gt_path.parent
        rel = session_dir.relative_to(args.labels_root)
        out_path = args.output_dir / (str(rel).replace("/", "__") + ".parquet")
        if out_path.exists():
            skipped += 1
            continue

        src_session = args.source_root / rel
        if not src_session.exists():
            continue  # 이 배치에 영상 없음 → 조용히 skip

        parts = rel.parts  # 스쿼트/{변형}/{오류}/{난이도}/{subject}/{세션}
        base_meta = {
            "subject_id": session_dir.parent.name,
            "session": str(rel),
            "ex_type": parts[1] if len(parts) > 1 else "",
            "error_type": parts[2] if len(parts) > 2 else "",
            "level": parts[3] if len(parts) > 3 else "",
        }
        gt = load_gt(gt_path)

        session_rows = []
        for cam_dir in sorted(src_session.glob("camera*")):
            m = _CAM_RE.search(cam_dir.name)
            if not m:
                continue
            cam_idx = int(m.group(1))
            if cam_filter is not None and cam_idx not in cam_filter:
                continue
            avis = sorted((cam_dir / "video").glob("*.avi"))
            if not avis:
                continue
            ann_start, ann_end = read_segment(session_dir / cam_dir.name / "video" / "annotation.json")
            meta = dict(base_meta)
            meta.update({
                "camera": cam_dir.name,
                "cam_idx": cam_idx,
                "view_type": "side" if cam_idx in side_set else "front",
                "ann_start": ann_start,
                "ann_end": ann_end,
            })
            from squat.pose_estimation import PoseEstimator  # lazy
            pose = PoseEstimator(static_image_mode=False)
            try:
                rows = dump_camera(pose, avis[0], gt, meta, args.timestamp_step_ms)
            finally:
                pose.close()
            session_rows.extend(rows)
            print(f"  {rel}/{cam_dir.name} (cam{cam_idx},{meta['view_type']}): {len(rows)} rows")

        if session_rows:
            pd.DataFrame(session_rows).to_parquet(out_path, index=False)
            done += 1
            print(f"[saved] {out_path.name}  ({len(session_rows)} rows)")

    print(f"\n완료: {done} 세션 덤프, {skipped} 세션 이미존재(skip).")


if __name__ == "__main__":
    main()
