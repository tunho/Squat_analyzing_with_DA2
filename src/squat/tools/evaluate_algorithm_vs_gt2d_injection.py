from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np


_THIS_FILE = Path(__file__).resolve()
_CANDIDATE_PATHS = [
    _THIS_FILE.parent,
    _THIS_FILE.parent.parent,
    _THIS_FILE.parent / "src",
    _THIS_FILE.parent.parent / "src",
]
for _p in _CANDIDATE_PATHS:
    if _p.exists():
        _s = str(_p)
        if _s not in sys.path:
            sys.path.insert(0, _s)

try:
    from squat.analyzer.knee_relative import RelativeModelSquatAnalyzer
    from squat.calibration.relative_model import RelativeDepthModel
    from squat.calibration.relative_models import get_relative_model
    from squat.domain.types import Point3D, SquatFrame
    from squat.geometry.angles import calculate_knee_angle
except ImportError:
    from analyzer.knee_relative import RelativeModelSquatAnalyzer
    from calibration.relative_model import RelativeDepthModel
    from calibration.relative_models import get_relative_model
    from domain.types import Point3D, SquatFrame
    from geometry.angles import calculate_knee_angle


@dataclass(slots=True)
class FrameAlignedData:
    frame_index: int
    gt_hip: np.ndarray
    gt_knee: np.ndarray
    gt_ankle: np.ndarray
    gt_angle: float


@dataclass(slots=True)
class EvalSample:
    frame_index: int
    gt_angle: float
    pred_angle: float
    gt2d_injected_angle: float
    mp_world_angle: float


@dataclass(slots=True)
class EvalSummary:
    algorithm: str
    aligned_frames: int
    pred_mae: float
    gt2d_injected_mae: float
    mp_world_mae: float
    mp_world_aligned_frames: int
    angle_plot_path: str
    error_plot_path: str


class AlgorithmAdapter(Protocol):
    name: str

    def create_analyzer(self) -> Any:
        ...

    def finalize_pred_angles(self, analyzer: Any) -> list[float]:
        ...

    def compute_gt2d_injected_angles(
        self,
        analyzer: Any,
        aligned_gt: list[FrameAlignedData],
    ) -> list[float]:
        ...


def load_gt_joints(gt_npy_path: str) -> np.ndarray:
    gt_joints = np.load(gt_npy_path)
    if gt_joints.ndim != 3:
        raise ValueError(
            f"GT npy must be 3D: expected (frames, joints, dims), got {gt_joints.shape}."
        )
    if gt_joints.shape[2] < 3:
        raise ValueError(
            f"GT last dimension must contain at least x,y,z. Got shape {gt_joints.shape}."
        )
    return gt_joints


def validate_joint_indices(
    gt_joints: np.ndarray,
    hip_idx: int,
    knee_idx: int,
    ankle_idx: int,
) -> None:
    joint_count = int(gt_joints.shape[1])
    valid_range = f"0~{joint_count - 1}"

    for name, idx in (
        ("hip_idx", hip_idx),
        ("knee_idx", knee_idx),
        ("ankle_idx", ankle_idx),
    ):
        if idx < 0 or idx >= joint_count:
            raise IndexError(
                f"{name}={idx} is out of range for GT joint_count={joint_count}. "
                f"Valid index range is {valid_range}."
            )


def calc_2d_len(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def calc_gt_angle(gt_hip: np.ndarray, gt_knee: np.ndarray, gt_ankle: np.ndarray) -> float:
    return calculate_knee_angle(
        Point3D(float(gt_hip[0]), float(gt_hip[1]), float(gt_hip[2])),
        Point3D(float(gt_knee[0]), float(gt_knee[1]), float(gt_knee[2])),
        Point3D(float(gt_ankle[0]), float(gt_ankle[1]), float(gt_ankle[2])),
    )


def build_aligned_gt(
    history_data: list[Any],
    gt_joints: np.ndarray,
    hip_idx: int,
    knee_idx: int,
    ankle_idx: int,
) -> list[FrameAlignedData]:
    aligned: list[FrameAlignedData] = []
    num_frames = int(gt_joints.shape[0])

    for frame in history_data:
        frame_idx = int(frame.frame_index)
        if frame_idx < 0 or frame_idx >= num_frames:
            continue

        gt_hip = gt_joints[frame_idx, hip_idx]
        gt_knee = gt_joints[frame_idx, knee_idx]
        gt_ankle = gt_joints[frame_idx, ankle_idx]

        if (
            np.isnan(gt_hip[:3]).any()
            or np.isnan(gt_knee[:3]).any()
            or np.isnan(gt_ankle[:3]).any()
        ):
            continue

        aligned.append(
            FrameAlignedData(
                frame_index=frame_idx,
                gt_hip=gt_hip,
                gt_knee=gt_knee,
                gt_ankle=gt_ankle,
                gt_angle=calc_gt_angle(gt_hip, gt_knee, gt_ankle),
            )
        )

    return aligned


def compute_mp_world_angles(history_data: list[SquatFrame]) -> dict[int, float]:
    mp_angles: dict[int, float] = {}
    for frame in history_data:
        if (
            frame.mp_world_hip is None
            or frame.mp_world_knee is None
            or frame.mp_world_ankle is None
        ):
            continue
        mp_angles[int(frame.frame_index)] = calculate_knee_angle(
            frame.mp_world_hip,
            frame.mp_world_knee,
            frame.mp_world_ankle,
        )
    return mp_angles


def mae(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Cannot compute MAE on empty arrays.")
    return float(np.mean(np.abs(a - b)))


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=float)
    if window <= 1:
        return values.astype(float, copy=True)

    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = float(np.mean(values[start : i + 1]))
    return out


def save_angle_plot(
    samples: list[EvalSample],
    output_png: Path,
    title: str,
) -> None:
    frame_indices = np.array([s.frame_index for s in samples], dtype=int)
    gt_angles = np.array([s.gt_angle for s in samples], dtype=float)
    pred_angles = np.array([s.pred_angle for s in samples], dtype=float)
    gt2d_angles = np.array([s.gt2d_injected_angle for s in samples], dtype=float)
    mp_world_angles = np.array([s.mp_world_angle for s in samples], dtype=float)

    pred_mae = mae(pred_angles, gt_angles)
    gt2d_mae = mae(gt2d_angles, gt_angles)
    mp_world_mae = mae(mp_world_angles, gt_angles)

    plt.figure(figsize=(14, 5))
    plt.plot(frame_indices, gt_angles, label="GT angle")
    plt.plot(frame_indices, pred_angles, label=f"Algorithm (MAE={pred_mae:.3f})")
    plt.plot(frame_indices, gt2d_angles, label=f"Algorithm + GT 2D (MAE={gt2d_mae:.3f})")
    plt.plot(frame_indices, mp_world_angles, label=f"MediaPipe world 3D (MAE={mp_world_mae:.3f})")
    plt.xlabel("Frame index")
    plt.ylabel("Knee angle (deg)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()


def save_error_plot(
    samples: list[EvalSample],
    output_png: Path,
    title: str,
    rolling_window: int,
) -> None:
    frame_indices = np.array([s.frame_index for s in samples], dtype=int)
    gt_angles = np.array([s.gt_angle for s in samples], dtype=float)
    pred_angles = np.array([s.pred_angle for s in samples], dtype=float)
    gt2d_angles = np.array([s.gt2d_injected_angle for s in samples], dtype=float)
    mp_world_angles = np.array([s.mp_world_angle for s in samples], dtype=float)

    pred_abs_err = np.abs(pred_angles - gt_angles)
    gt2d_abs_err = np.abs(gt2d_angles - gt_angles)
    mp_world_abs_err = np.abs(mp_world_angles - gt_angles)
    pred_roll = rolling_mean(pred_abs_err, rolling_window)
    gt2d_roll = rolling_mean(gt2d_abs_err, rolling_window)
    mp_world_roll = rolling_mean(mp_world_abs_err, rolling_window)

    plt.figure(figsize=(14, 5))
    plt.plot(frame_indices, pred_abs_err, label="Abs error: algorithm", alpha=0.45)
    plt.plot(frame_indices, gt2d_abs_err, label="Abs error: algorithm + GT 2D", alpha=0.45)
    plt.plot(frame_indices, mp_world_abs_err, label="Abs error: MediaPipe world 3D", alpha=0.45)
    plt.plot(frame_indices, pred_roll, label=f"Rolling MAE algorithm (w={rolling_window})", linewidth=2)
    plt.plot(frame_indices, gt2d_roll, label=f"Rolling MAE algorithm + GT 2D (w={rolling_window})", linewidth=2)
    plt.plot(frame_indices, mp_world_roll, label=f"Rolling MAE MediaPipe world 3D (w={rolling_window})", linewidth=2)
    plt.xlabel("Frame index")
    plt.ylabel("Absolute error (deg)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()


class RelativeModelAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model = get_relative_model(name)

    def create_analyzer(self) -> Any:
        return RelativeModelSquatAnalyzer(model=self.name)

    def finalize_pred_angles(self, analyzer: Any) -> list[float]:
        _, pred_angles, params = analyzer.finalize_analysis()
        if params is None:
            raise RuntimeError(f"{self.name} returned no calibration params.")
        return pred_angles

    def compute_gt2d_injected_angles(
        self,
        analyzer: Any,
        aligned_gt: list[FrameAlignedData],
    ) -> list[float]:
        frame_map: dict[int, SquatFrame] = {
            int(frame.frame_index): frame for frame in analyzer.history_data
        }

        filtered_frames: list[SquatFrame] = []
        for item in aligned_gt:
            frame = frame_map.get(item.frame_index)
            if frame is None:
                continue

            frame.thigh_len_2d = calc_2d_len(item.gt_hip, item.gt_knee)
            frame.shank_len_2d = calc_2d_len(item.gt_knee, item.gt_ankle)
            frame.hip.x = float(item.gt_hip[0])
            frame.hip.y = float(item.gt_hip[1])
            frame.knee.x = float(item.gt_knee[0])
            frame.knee.y = float(item.gt_knee[1])
            frame.ankle.x = float(item.gt_ankle[0])
            frame.ankle.y = float(item.gt_ankle[1])
            filtered_frames.append(frame)

        if not filtered_frames:
            return []

        params = self.model.fit(filtered_frames)
        if params is None:
            return []

        angles: list[float] = []
        for frame in filtered_frames:
            hip, knee, ankle = self.model.build_points(frame, params)
            angles.append(calculate_knee_angle(hip, knee, ankle))
        return angles


def get_adapter(name: str) -> AlgorithmAdapter:
    return RelativeModelAdapter(name)


def evaluate(
    algorithm: str,
    video_path: str,
    gt_npy_path: str,
    hip_idx: int,
    knee_idx: int,
    ankle_idx: int,
    output_dir: str,
    force_side: str | None = None,
    max_frames: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    rolling_window: int = 15,
) -> EvalSummary:
    gt_joints = load_gt_joints(gt_npy_path)
    validate_joint_indices(gt_joints, hip_idx, knee_idx, ankle_idx)

    adapter = get_adapter(algorithm)
    analyzer = adapter.create_analyzer()

    analyzer.process_video(
        video_path=video_path,
        output_path=None,
        max_frames=max_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        force_side=force_side,
        save_video=False,
        store_debug_images=False,
        store_depth_maps=False,
    )

    if not analyzer.history_data:
        raise RuntimeError("No valid frames were accumulated by analyzer.history_data.")

    pred_angles = adapter.finalize_pred_angles(analyzer)
    aligned_gt = build_aligned_gt(
        history_data=analyzer.history_data,
        gt_joints=gt_joints,
        hip_idx=hip_idx,
        knee_idx=knee_idx,
        ankle_idx=ankle_idx,
    )
    if not aligned_gt:
        raise RuntimeError("No GT-aligned frames found after filtering frame range / NaNs.")

    gt2d_injected_angles = adapter.compute_gt2d_injected_angles(analyzer, aligned_gt)
    if not gt2d_injected_angles:
        raise RuntimeError("Failed to compute GT 2D injected angles.")

    mp_world_by_frame = compute_mp_world_angles(analyzer.history_data)
    if not mp_world_by_frame:
        raise RuntimeError("Failed to compute MediaPipe world 3D baseline angles.")

    pred_by_frame = {
        int(frame.frame_index): float(angle)
        for frame, angle in zip(analyzer.history_data, pred_angles)
    }
    gt_by_frame = {item.frame_index: float(item.gt_angle) for item in aligned_gt}
    gt2d_by_frame = {
        item.frame_index: float(angle)
        for item, angle in zip(aligned_gt, gt2d_injected_angles)
    }

    common_frames = sorted(set(pred_by_frame) & set(gt_by_frame) & set(gt2d_by_frame) & set(mp_world_by_frame))
    if not common_frames:
        raise RuntimeError("No common frames across pred / gt / gt2d-injected / MediaPipe world 3D series.")

    samples: list[EvalSample] = [
        EvalSample(
            frame_index=idx,
            gt_angle=gt_by_frame[idx],
            pred_angle=pred_by_frame[idx],
            gt2d_injected_angle=gt2d_by_frame[idx],
            mp_world_angle=mp_world_by_frame[idx],
        )
        for idx in common_frames
    ]

    gt_arr = np.array([s.gt_angle for s in samples], dtype=float)
    pred_arr = np.array([s.pred_angle for s in samples], dtype=float)
    gt2d_arr = np.array([s.gt2d_injected_angle for s in samples], dtype=float)
    mp_world_arr = np.array([s.mp_world_angle for s in samples], dtype=float)

    pred_mae = mae(pred_arr, gt_arr)
    gt2d_mae = mae(gt2d_arr, gt_arr)
    mp_world_mae = mae(mp_world_arr, gt_arr)

    output_dir_path = Path(output_dir)
    angle_plot_path = output_dir_path / f"{algorithm}_angle_comparison.png"
    error_plot_path = output_dir_path / f"{algorithm}_error_comparison.png"
    summary_json_path = output_dir_path / f"{algorithm}_summary.json"

    save_angle_plot(
        samples=samples,
        output_png=angle_plot_path,
        title=f"{algorithm}: angle comparison",
    )
    save_error_plot(
        samples=samples,
        output_png=error_plot_path,
        title=f"{algorithm}: absolute error / rolling MAE",
        rolling_window=rolling_window,
    )

    summary = EvalSummary(
        algorithm=algorithm,
        aligned_frames=len(samples),
        pred_mae=pred_mae,
        gt2d_injected_mae=gt2d_mae,
        mp_world_mae=mp_world_mae,
        mp_world_aligned_frames=len(samples),
        angle_plot_path=str(angle_plot_path),
        error_plot_path=str(error_plot_path),
    )

    output_dir_path.mkdir(parents=True, exist_ok=True)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare: (1) current algorithm angle vs GT angle and "
            "(2) algorithm with GT 2D injection vs GT angle."
        )
    )
    parser.add_argument("--algorithm", default="relative_linear")
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt-npy", required=True)
    parser.add_argument("--hip-idx", type=int, required=True)
    parser.add_argument("--knee-idx", type=int, required=True)
    parser.add_argument("--ankle-idx", type=int, required=True)
    parser.add_argument("--force-side", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--rolling-window", type=int, default=15)
    parser.add_argument("--output-dir", default="outputs/eval_algorithm_vs_gt2d")
    args = parser.parse_args()

    summary = evaluate(
        algorithm=args.algorithm,
        video_path=args.video,
        gt_npy_path=args.gt_npy,
        hip_idx=args.hip_idx,
        knee_idx=args.knee_idx,
        ankle_idx=args.ankle_idx,
        output_dir=args.output_dir,
        force_side=args.force_side,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        rolling_window=args.rolling_window,
    )

    print("=" * 80)
    print(f"Algorithm                 : {summary.algorithm}")
    print(f"Aligned frames            : {summary.aligned_frames}")
    print(f"Algorithm MAE             : {summary.pred_mae:.6f} deg")
    print(f"Algorithm + GT 2D MAE     : {summary.gt2d_injected_mae:.6f} deg")
    print(f"MediaPipe world 3D MAE    : {summary.mp_world_mae:.6f} deg")
    print(f"MP world aligned frames   : {summary.mp_world_aligned_frames}")
    print(f"Angle plot                : {summary.angle_plot_path}")
    print(f"Error plot                : {summary.error_plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()