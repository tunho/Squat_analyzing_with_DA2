from __future__ import annotations

"""
NLF (Neural Localizer Fields, NeurIPS 2024) 추정기로 feature CSV 추출.

기존 MediaPipe 추출(paper_extract.py)과 '동일한 CSV 포맷'을 만들어,
이후 LOSO/보정기 파이프라인을 그대로 재사용한다(추정기 비종속 실험, 표1).
→ 여기서는 MediaPipe 대신 NLF의 3D 관절을 SquatFrame.mp_world_* 에 채우고
   paper_extract.extract_feature_rows 를 재사용한다.

핵심 설계:
  - NLF 호출부는 NLFEstimator 한 곳에만 격리(설치/가중치 의존성 분리).
  - 무릎각은 평행이동·스케일 불변이라 NLF의 카메라좌표(meter) 그대로 OK.
    extract_feature_rows 가 hip 기준 상대화 + med_total 정규화를 수행.
  - 좌/우 다리는 GT 인덱스(get_leg_indices)와 동일 규칙으로 결정.

실행 전제: torch + nlf(github isarandi/nlf) 설치 + 가중치(.torchscript). GPU 권장.
"""

import argparse
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


# SMPL-24 스켈레톤 관절 인덱스 (NLF 기본 출력 스켈레톤 중 하나).
# 다른 skeleton(h36m_17 등) 쓰면 여기만 교체.
SMPL24_SIDE_JOINTS = {
    "left":  {"hip": 1, "knee": 4, "ankle": 7, "shoulder": 16},
    "right": {"hip": 2, "knee": 5, "ankle": 8, "shoulder": 17},
}


def gt_side(seg_df, subject_id: str, exercise: str) -> str:
    """GT 다리 인덱스로 좌/우 판정(get_leg_indices 와 동일 규칙)."""
    hip_idx, _, _ = get_leg_indices(seg_df, subject_id, exercise)
    return "right" if hip_idx == 21 else "left"


class NLFEstimator:
    """NLF TorchScript 모델 래퍼. predict_joints3d(frame_bgr) → (J,3) meter 또는 None.

    ※ NLF inference API는 설치한 nlf 버전에 맞춰 _infer() 한 줄만 조정하면 됨.
       (공식 repo: github.com/isarandi/nlf — TorchScript multi-person 추론)
    """

    def __init__(self, model_path: str, device: str = "cuda", skeleton: str = "smpl_24"):
        import torch
        import torchvision  # noqa: F401  ← TorchScript 모델의 C++ op(nms 등) 등록에 필수

        self.torch = torch
        self.device = device
        self.skeleton = skeleton
        self.model = torch.jit.load(model_path, map_location=device).to(device).eval()

    def _infer(self, rgb_uint8_hwc: np.ndarray):
        """단일 이미지 → NLF 추론. 반환: pred['joints3d'] = list(이미지별), 각 [num_people, 24, 3] mm.

        실측 확인(demo.ipynb·실행 검증):
            pred = model.detect_smpl_batched(batch)   # batch: [B,3,H,W] uint8 cuda
            pred['joints3d'][0]  # → [num_people, 24, 3] (SMPL 24관절, mm)
        """
        torch = self.torch
        chw = torch.from_numpy(rgb_uint8_hwc).permute(2, 0, 1).contiguous()
        batch = chw.unsqueeze(0).to(self.device)  # [1,3,H,W] uint8
        with torch.inference_mode():
            pred = self.model.detect_smpl_batched(batch)
        return pred["joints3d"]

    def predict_joints3d(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        j = self._infer(rgb)
        if not j or len(j) == 0:
            return None
        per_image = j[0]  # [num_people, 24, 3]
        if per_image is None or per_image.shape[0] == 0:  # 검출 없음
            return None
        arr = per_image[0]  # 첫 번째 사람 [24,3] (스쿼트 영상=1인)
        arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
        # 단위(mm)는 무릎각·정규화에 무관(hip상대/med_total 정규화로 상쇄). 그대로 사용.
        return arr


def joints_to_squatframe(joints3d: np.ndarray, side: str, frame_index: int) -> SquatFrame | None:
    """NLF 3D 관절(J,3) → SquatFrame(mp_world_* 채움). 무릎각 계산에 필요한 4관절만."""
    idx = SMPL24_SIDE_JOINTS[side]
    try:
        hip = Point3D(*joints3d[idx["hip"]].tolist(), 1.0)
        knee = Point3D(*joints3d[idx["knee"]].tolist(), 1.0)
        ankle = Point3D(*joints3d[idx["ankle"]].tolist(), 1.0)
        shoulder = Point3D(*joints3d[idx["shoulder"]].tolist(), 1.0)
    except (IndexError, TypeError):
        return None
    return SquatFrame(
        frame_index=frame_index,
        side=side,
        hip=hip, knee=knee, ankle=ankle, shoulder=shoulder,
        mp_world_hip=hip, mp_world_knee=knee, mp_world_ankle=ankle, mp_world_shoulder=shoulder,
    )


def extract_video_features_nlf(
    estimator: NLFEstimator,
    video_path: Path,
    gt_angles: dict[int, float],
    subject_id: str,
    side: str,
    calib_frames: int = 0,
) -> pd.DataFrame:
    if not gt_angles:
        return pd.DataFrame()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return pd.DataFrame()

    start_frame = min(gt_angles)
    end_frame = max(gt_angles)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[SquatFrame] = []
    frame_idx = start_frame
    try:
        while frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            joints = estimator.predict_joints3d(frame)
            if joints is not None:
                sf = joints_to_squatframe(joints, side, frame_idx)
                if sf is not None:
                    frames.append(sf)
            frame_idx += 1
    finally:
        cap.release()

    df = pd.DataFrame(
        extract_feature_rows(frames, video_path.name, subject_id,
                             med_total=calib_med_total(frames, calib_frames))
    )
    if df.empty:
        return df
    df["gt_angle"] = df["frame_index"].map(gt_angles)
    return df.dropna(subset=["gt_angle"]).copy()


def main() -> None:
    p = argparse.ArgumentParser(description="Extract NLF features (same CSV schema as paper_extract.py).")
    p.add_argument("--dataset", type=Path, default=Path("dataset/raw/REHAB24-6"))
    p.add_argument("--exercise", default="Ex6")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=str, required=True, help="NLF TorchScript 모델 경로(.torchscript)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skeleton", default="smpl_24")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--view-suffixes", nargs="*", default=DEFAULT_VIEW_SUFFIXES)
    p.add_argument("--calib-frames", type=int, default=0)
    args = p.parse_args()

    video_dir = args.dataset / "videos" / args.exercise
    gt_dir = args.dataset / "3d_joints" / args.exercise
    subjects = args.subjects or discover_subjects(gt_dir)
    seg_df = load_segmentation_table(args.dataset)

    estimator = NLFEstimator(args.model, device=args.device, skeleton=args.skeleton)

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
            df = extract_video_features_nlf(estimator, video_path, gt_angles, subject_id, side,
                                            calib_frames=args.calib_frames)
            if not df.empty:
                all_frames.append(df)
                print(f"Extracted {len(df)} rows from {video_path.name} (side={side})", flush=True)

    if not all_frames:
        raise RuntimeError("No feature rows were extracted (NLF).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_frames, ignore_index=True).to_csv(args.output, index=False)
    print(f"Saved NLF features: {args.output}")


if __name__ == "__main__":
    main()
