from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    from squat.tools.evaluate_algorithm_vs_gt2d_injection import (
        RelativeModelAdapter,
        build_aligned_gt,
        compute_mp_world_angles,
        load_gt_joints,
        mae,
        rolling_mean,
        validate_joint_indices,
    )
except ImportError:
    from tools.evaluate_algorithm_vs_gt2d_injection import (
        RelativeModelAdapter,
        build_aligned_gt,
        compute_mp_world_angles,
        load_gt_joints,
        mae,
        rolling_mean,
        validate_joint_indices,
    )


@dataclass(slots=True)
class SeriesResult:
    algorithm: str
    frame_indices: list[int]
    gt_angles: list[float]
    pred_angles: list[float]
    gt2d_angles: list[float]
    mp_world_angles: list[float]
    pred_mae: float
    gt2d_mae: float
    mp_world_mae: float
    aligned_frames: int


@dataclass(slots=True)
class DualComparisonSummary:
    primary_algorithm: str
    secondary_algorithm: str
    aligned_frames: int
    primary_mae: float
    secondary_mae: float
    primary_gt2d_mae: float
    secondary_gt2d_mae: float
    mp_world_mae: float
    angle_plot_path: str
    error_plot_path: str
    summary_json_path: str


def evaluate_series(
    algorithm: str,
    video_path: str,
    gt_npy_path: str,
    hip_idx: int,
    knee_idx: int,
    ankle_idx: int,
    force_side: str | None = None,
    max_frames: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    calibration_overrides: dict[str, float] | None = None,
) -> SeriesResult:
    gt_joints = load_gt_joints(gt_npy_path)
    validate_joint_indices(gt_joints, hip_idx, knee_idx, ankle_idx)

    adapter = RelativeModelAdapter(algorithm, calibration_overrides=calibration_overrides)
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
        raise RuntimeError(f"{algorithm}: no valid frames were accumulated.")

    pred_angles, _pred_params = adapter.finalize_pred_angles(analyzer)
    aligned_gt = build_aligned_gt(
        history_data=analyzer.history_data,
        gt_joints=gt_joints,
        hip_idx=hip_idx,
        knee_idx=knee_idx,
        ankle_idx=ankle_idx,
    )
    if not aligned_gt:
        raise RuntimeError(f"{algorithm}: no GT-aligned frames found.")

    gt2d_injected_angles, _gt2d_params = adapter.compute_gt2d_injected_angles(analyzer, aligned_gt)
    if not gt2d_injected_angles:
        raise RuntimeError(f"{algorithm}: failed to compute GT 2D injected angles.")

    mp_world_by_frame = compute_mp_world_angles(analyzer.history_data)
    if not mp_world_by_frame:
        raise RuntimeError(f"{algorithm}: failed to compute MediaPipe world 3D baseline.")

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
        raise RuntimeError(f"{algorithm}: no common frames across all series.")

    gt_arr = np.array([gt_by_frame[idx] for idx in common_frames], dtype=float)
    pred_arr = np.array([pred_by_frame[idx] for idx in common_frames], dtype=float)
    gt2d_arr = np.array([gt2d_by_frame[idx] for idx in common_frames], dtype=float)
    mp_world_arr = np.array([mp_world_by_frame[idx] for idx in common_frames], dtype=float)

    return SeriesResult(
        algorithm=algorithm,
        frame_indices=list(common_frames),
        gt_angles=gt_arr.tolist(),
        pred_angles=pred_arr.tolist(),
        gt2d_angles=gt2d_arr.tolist(),
        mp_world_angles=mp_world_arr.tolist(),
        pred_mae=mae(pred_arr, gt_arr),
        gt2d_mae=mae(gt2d_arr, gt_arr),
        mp_world_mae=mae(mp_world_arr, gt_arr),
        aligned_frames=len(common_frames),
    )


def intersect_series(primary: SeriesResult, secondary: SeriesResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common_frames = sorted(set(primary.frame_indices) & set(secondary.frame_indices))
    if not common_frames:
        raise RuntimeError("No common frames between primary and secondary algorithm results.")

    p_idx = {frame: i for i, frame in enumerate(primary.frame_indices)}
    s_idx = {frame: i for i, frame in enumerate(secondary.frame_indices)}

    frames = np.array(common_frames, dtype=int)
    gt = np.array([primary.gt_angles[p_idx[f]] for f in common_frames], dtype=float)
    mp_world = np.array([primary.mp_world_angles[p_idx[f]] for f in common_frames], dtype=float)
    primary_pred = np.array([primary.pred_angles[p_idx[f]] for f in common_frames], dtype=float)
    secondary_pred = np.array([secondary.pred_angles[s_idx[f]] for f in common_frames], dtype=float)
    primary_gt2d = np.array([primary.gt2d_angles[p_idx[f]] for f in common_frames], dtype=float)
    secondary_gt2d = np.array([secondary.gt2d_angles[s_idx[f]] for f in common_frames], dtype=float)
    return frames, gt, mp_world, primary_pred, secondary_pred, primary_gt2d, secondary_gt2d


def save_dual_angle_plot(
    frames: np.ndarray,
    gt: np.ndarray,
    mp_world: np.ndarray,
    primary_pred: np.ndarray,
    secondary_pred: np.ndarray,
    output_png: Path,
    title: str,
    primary_label: str,
    secondary_label: str,
) -> None:
    primary_mae = mae(primary_pred, gt)
    secondary_mae = mae(secondary_pred, gt)
    mp_world_mae = mae(mp_world, gt)

    plt.figure(figsize=(15, 6))
    plt.plot(frames, gt, label="GT angle")
    plt.plot(frames, primary_pred, label=f"{primary_label} (MAE={primary_mae:.3f})")
    plt.plot(frames, secondary_pred, label=f"{secondary_label} (MAE={secondary_mae:.3f})")
    plt.plot(frames, mp_world, label=f"MediaPipe world 3D raw (MAE={mp_world_mae:.3f})", alpha=0.8)
    plt.xlabel("Frame index")
    plt.ylabel("Knee angle (deg)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()


def save_dual_error_plot(
    frames: np.ndarray,
    gt: np.ndarray,
    mp_world: np.ndarray,
    primary_pred: np.ndarray,
    secondary_pred: np.ndarray,
    output_png: Path,
    title: str,
    rolling_window: int,
    primary_label: str,
    secondary_label: str,
) -> None:
    primary_abs = np.abs(primary_pred - gt)
    secondary_abs = np.abs(secondary_pred - gt)
    mp_world_abs = np.abs(mp_world - gt)

    primary_roll = rolling_mean(primary_abs, rolling_window)
    secondary_roll = rolling_mean(secondary_abs, rolling_window)
    mp_world_roll = rolling_mean(mp_world_abs, rolling_window)

    plt.figure(figsize=(15, 6))
    plt.plot(frames, primary_abs, label=f"Abs error: {primary_label}", alpha=0.35)
    plt.plot(frames, secondary_abs, label=f"Abs error: {secondary_label}", alpha=0.35)
    plt.plot(frames, mp_world_abs, label="Abs error: MediaPipe world 3D raw", alpha=0.35)
    plt.plot(frames, primary_roll, label=f"Rolling MAE {primary_label} (w={rolling_window})", linewidth=2)
    plt.plot(frames, secondary_roll, label=f"Rolling MAE {secondary_label} (w={rolling_window})", linewidth=2)
    plt.plot(frames, mp_world_roll, label=f"Rolling MAE MediaPipe world 3D raw (w={rolling_window})", linewidth=2)
    plt.xlabel("Frame index")
    plt.ylabel("Absolute error (deg)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()


def compare_algorithms(
    primary_algorithm: str,
    secondary_algorithm: str,
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
    calibration_overrides: dict[str, float] | None = None,
) -> DualComparisonSummary:
    primary = evaluate_series(
        algorithm=primary_algorithm,
        video_path=video_path,
        gt_npy_path=gt_npy_path,
        hip_idx=hip_idx,
        knee_idx=knee_idx,
        ankle_idx=ankle_idx,
        force_side=force_side,
        max_frames=max_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        calibration_overrides=calibration_overrides,
    )
    secondary = evaluate_series(
        algorithm=secondary_algorithm,
        video_path=video_path,
        gt_npy_path=gt_npy_path,
        hip_idx=hip_idx,
        knee_idx=knee_idx,
        ankle_idx=ankle_idx,
        force_side=force_side,
        max_frames=max_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        calibration_overrides=calibration_overrides,
    )

    frames, gt, mp_world, primary_pred, secondary_pred, primary_gt2d, secondary_gt2d = intersect_series(primary, secondary)

    output_dir_path = Path(output_dir)
    stem = f"{primary_algorithm}_vs_{secondary_algorithm}"
    angle_plot_path = output_dir_path / f"{stem}_angle_comparison.png"
    error_plot_path = output_dir_path / f"{stem}_error_comparison.png"
    summary_json_path = output_dir_path / f"{stem}_summary.json"

    save_dual_angle_plot(
        frames=frames,
        gt=gt,
        mp_world=mp_world,
        primary_pred=primary_pred,
        secondary_pred=secondary_pred,
        output_png=angle_plot_path,
        title=f"{primary_algorithm} vs {secondary_algorithm}: angle comparison",
        primary_label=primary_algorithm,
        secondary_label=secondary_algorithm,
    )
    save_dual_error_plot(
        frames=frames,
        gt=gt,
        mp_world=mp_world,
        primary_pred=primary_pred,
        secondary_pred=secondary_pred,
        output_png=error_plot_path,
        title=f"{primary_algorithm} vs {secondary_algorithm}: absolute error / rolling MAE",
        rolling_window=rolling_window,
        primary_label=primary_algorithm,
        secondary_label=secondary_algorithm,
    )

    summary = DualComparisonSummary(
        primary_algorithm=primary_algorithm,
        secondary_algorithm=secondary_algorithm,
        aligned_frames=len(frames),
        primary_mae=mae(primary_pred, gt),
        secondary_mae=mae(secondary_pred, gt),
        primary_gt2d_mae=mae(primary_gt2d, gt),
        secondary_gt2d_mae=mae(secondary_gt2d, gt),
        mp_world_mae=mae(mp_world, gt),
        angle_plot_path=str(angle_plot_path),
        error_plot_path=str(error_plot_path),
        summary_json_path=str(summary_json_path),
    )

    output_dir_path.mkdir(parents=True, exist_ok=True)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run two algorithms on the same video/GT pair and draw both curves on the same plots. "
            "Useful for direct comparison such as MediaPipe 3D + correction vs MediaPipe 2D + DA2 + correction."
        )
    )
    parser.add_argument("--primary-algorithm", default="mp_world_add")
    parser.add_argument("--secondary-algorithm", default="relative_add")
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
    parser.add_argument("--a", type=float, default=None)
    parser.add_argument("--b", type=float, default=None)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--target-leg-len", type=float, default=None)
    parser.add_argument("--min-visibility", type=float, default=None)
    parser.add_argument("--top-percent", type=float, default=None)
    parser.add_argument("--trim-low", type=float, default=None)
    parser.add_argument("--trim-high", type=float, default=None)
    parser.add_argument("--output-dir", default="outputs/dual_algorithm_comparison")
    args = parser.parse_args()

    overrides = {
        "min_visibility": args.min_visibility,
        "top_percent": args.top_percent,
        "trim_percentile_low": args.trim_low,
        "trim_percentile_high": args.trim_high,
    }

    summary = compare_algorithms(
        primary_algorithm=args.primary_algorithm,
        secondary_algorithm=args.secondary_algorithm,
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
        calibration_overrides=overrides,
    )

    print("=" * 80)
    print(f"Primary algorithm          : {summary.primary_algorithm}")
    print(f"Secondary algorithm        : {summary.secondary_algorithm}")
    print(f"Aligned frames             : {summary.aligned_frames}")
    print(f"Primary MAE                : {summary.primary_mae:.6f} deg")
    print(f"Secondary MAE              : {summary.secondary_mae:.6f} deg")
    print(f"Primary + GT 2D MAE        : {summary.primary_gt2d_mae:.6f} deg")
    print(f"Secondary + GT 2D MAE      : {summary.secondary_gt2d_mae:.6f} deg")
    print(f"MediaPipe world 3D raw MAE : {summary.mp_world_mae:.6f} deg")
    print(f"Angle plot                 : {summary.angle_plot_path}")
    print(f"Error plot                 : {summary.error_plot_path}")
    print(f"Summary JSON               : {summary.summary_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()