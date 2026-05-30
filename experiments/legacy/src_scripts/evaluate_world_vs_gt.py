from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from squat.analyzer import (
    MPWorldLenStableSquatAnalyzer,
    MPWorldOutlierCorrectedSquatAnalyzer,
    MPWorldRawSquatAnalyzer,
    MPWorldSmoothSquatAnalyzer,
    MPWorldVisibilityFilteredSquatAnalyzer,
    MPWorldSelectiveRepairSquatAnalyzer,
    MPWorldSelectiveBurstRepairSquatAnalyzer,
    MPWorldSelectiveHingeRepairSquatAnalyzer,
    MPWorldSelectiveDescentRepairAnalyzer,
)
from squat.domain.types import Point3D, SquatFrame
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import angle_between_unit_vectors, point_to_np, safe_unit


PHASE_ORDER = ["top", "descent", "bottom", "ascent"]
DESCENT_SUBPHASE_ORDER = ["descent_early", "descent_late"]


def build_analyzer(name: str):
    if name == "mp_world_raw":
        return MPWorldRawSquatAnalyzer()
    if name == "mp_world_smooth":
        return MPWorldSmoothSquatAnalyzer(smooth_alpha=0.2)
    if name == "mp_world_len_stable":
        return MPWorldLenStableSquatAnalyzer()
    if name == "mp_world_visibility_filtered":
        return MPWorldVisibilityFilteredSquatAnalyzer()
    if name == "mp_world_outlier_corrected":
        return MPWorldOutlierCorrectedSquatAnalyzer()
    if name == "mp_world_selective_repair":
        return MPWorldSelectiveRepairSquatAnalyzer()
    if name == "mp_world_selective_burst_repair":
        return MPWorldSelectiveBurstRepairSquatAnalyzer()
    if name == "mp_world_selective_hinge_repair":
        return MPWorldSelectiveHingeRepairSquatAnalyzer()
    if name == "mp_world_selective_descent_repair":
        return MPWorldSelectiveDescentRepairAnalyzer()
    raise ValueError(f"Unknown algorithm: {name}")



def load_gt_table(
    gt_path: str,
    frame_col: str,
    angle_col: str,
    gt_format: str,
    gt_hip_idx: int | None = None,
    gt_knee_idx: int | None = None,
    gt_ankle_idx: int | None = None,
) -> pd.DataFrame:
    path = Path(gt_path)

    if gt_format == "table":
        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported GT table format: {suffix}")

        if frame_col not in df.columns:
            raise ValueError(f"GT frame column '{frame_col}' not found. Available: {list(df.columns)}")
        if angle_col not in df.columns:
            raise ValueError(f"GT angle column '{angle_col}' not found. Available: {list(df.columns)}")

        df = df[[frame_col, angle_col]].copy()
        df = df.rename(columns={frame_col: "frame_index", angle_col: "gt_angle"})
        df["frame_index"] = df["frame_index"].astype(int)
        df["gt_angle"] = df["gt_angle"].astype(float)
        df = df.sort_values("frame_index").reset_index(drop=True)
        return df

    if gt_format == "npy_3d":
        if gt_hip_idx is None or gt_knee_idx is None or gt_ankle_idx is None:
            raise ValueError("For gt_format='npy_3d', you must provide --gt-hip-idx --gt-knee-idx --gt-ankle-idx")

        arr = np.load(path)

        if arr.ndim != 3:
            raise ValueError(f"Expected GT npy shape (T, J, C), got {arr.shape}")
        if arr.shape[2] < 3:
            raise ValueError(f"Expected last dim >= 3 for xyz, got {arr.shape}")

        xyz = arr[:, :, :3]
        num_frames = xyz.shape[0]
        gt_rows: list[dict] = []

        for t in range(num_frames):
            hip_xyz = xyz[t, gt_hip_idx]
            knee_xyz = xyz[t, gt_knee_idx]
            ankle_xyz = xyz[t, gt_ankle_idx]

            hip = Point3D(float(hip_xyz[0]), float(hip_xyz[1]), float(hip_xyz[2]), 1.0)
            knee = Point3D(float(knee_xyz[0]), float(knee_xyz[1]), float(knee_xyz[2]), 1.0)
            ankle = Point3D(float(ankle_xyz[0]), float(ankle_xyz[1]), float(ankle_xyz[2]), 1.0)

            gt_rows.append({
                "frame_index": t,
                "gt_angle": calculate_knee_angle(hip, knee, ankle),
            })

        return pd.DataFrame(gt_rows)

    raise ValueError(f"Unsupported gt_format: {gt_format}")



def build_prediction_df(analyzer, final_angles: list[float]) -> pd.DataFrame:
    valid_frames = []
    for frame in analyzer.history_data:
        angle = analyzer.compute_frame_angle(frame)
        if angle is not None:
            valid_frames.append(frame.frame_index)

    if len(valid_frames) != len(final_angles):
        raise RuntimeError(
            f"Frame count mismatch: valid_frames={len(valid_frames)} vs final_angles={len(final_angles)}"
        )

    return pd.DataFrame({
        "frame_index": valid_frames,
        "pred_angle": final_angles,
    })



def compute_metrics(aligned_df: pd.DataFrame) -> dict:
    err = aligned_df["abs_error"].to_numpy(dtype=float)
    sq = aligned_df["sq_error"].to_numpy(dtype=float)
    return {
        "aligned_frames": int(len(aligned_df)),
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(sq))),
        "max_error": float(np.max(err)),
        "median_abs_error": float(np.median(err)),
        "p95_abs_error": float(np.percentile(err, 95)),
        "mean_signed_error": float(np.mean(aligned_df["error"].to_numpy(dtype=float))),
    }



def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float).copy()
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)



def gradient_by_frame(frame_index: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=float)
    if len(values) == 2:
        dt = max(float(frame_index[1] - frame_index[0]), 1.0)
        g = np.array([(values[1] - values[0]) / dt, (values[1] - values[0]) / dt], dtype=float)
        return g
    return np.gradient(values.astype(float), frame_index.astype(float))



def label_gt_phases(
    gt_df: pd.DataFrame,
    smooth_window: int,
    top_band_ratio: float,
    bottom_band_ratio: float,
    motion_eps_ratio: float,
) -> pd.DataFrame:
    out = gt_df.copy().sort_values("frame_index").reset_index(drop=True)
    gt_angle = out["gt_angle"].to_numpy(dtype=float)
    gt_angle_smooth = rolling_mean(gt_angle, smooth_window)

    p5 = float(np.percentile(gt_angle_smooth, 5))
    p95 = float(np.percentile(gt_angle_smooth, 95))
    robust_range = max(p95 - p5, 1e-6)

    normalized_depth = (p95 - gt_angle_smooth) / robust_range
    normalized_depth = np.clip(normalized_depth, 0.0, 1.0)

    gt_vel = gradient_by_frame(out["frame_index"].to_numpy(dtype=float), gt_angle_smooth)
    motion_eps = motion_eps_ratio * robust_range

    phases: list[str] = []
    last_motion_phase = "descent"
    for depth, vel in zip(normalized_depth, gt_vel):
        if depth <= top_band_ratio:
            phase = "top"
        elif depth >= bottom_band_ratio:
            phase = "bottom"
        else:
            if vel < -motion_eps:
                phase = "descent"
                last_motion_phase = phase
            elif vel > motion_eps:
                phase = "ascent"
                last_motion_phase = phase
            else:
                phase = last_motion_phase
        phases.append(phase)

    depth_mid = 0.5 * (top_band_ratio + bottom_band_ratio)
    subphase: list[str] = []
    for phase, depth in zip(phases, normalized_depth):
        if phase == "descent":
            subphase.append("descent_early" if depth < depth_mid else "descent_late")
        else:
            subphase.append("")

    out["gt_angle_smooth"] = gt_angle_smooth
    out["gt_angle_velocity"] = gt_vel
    out["gt_angle_acceleration"] = gradient_by_frame(out["frame_index"].to_numpy(dtype=float), gt_vel)
    out["normalized_depth"] = normalized_depth
    out["gt_phase"] = phases
    out["descent_subphase"] = subphase
    out["gt_phase_depth_mid"] = depth_mid
    out["gt_phase_motion_eps"] = motion_eps
    return out



def _world_triplet(frame: SquatFrame) -> tuple[Point3D, Point3D, Point3D] | None:
    if frame.mp_world_hip is None or frame.mp_world_knee is None or frame.mp_world_ankle is None:
        return None
    return frame.mp_world_hip, frame.mp_world_knee, frame.mp_world_ankle



def build_world_feature_df(analyzer) -> pd.DataFrame:
    rows: list[dict] = []
    prev_thigh_u: np.ndarray | None = None
    prev_shank_u: np.ndarray | None = None

    for frame in analyzer.history_data:
        triplet = _world_triplet(frame)
        if triplet is None:
            continue
        hip, knee, ankle = triplet
        h = point_to_np(hip)
        k = point_to_np(knee)
        a = point_to_np(ankle)

        thigh_vec = h - k
        shank_vec = a - k
        thigh_len = float(np.linalg.norm(thigh_vec))
        shank_len = float(np.linalg.norm(shank_vec))
        thigh_u = safe_unit(thigh_vec)
        shank_u = safe_unit(shank_vec)

        min_visibility = float(min(hip.vis, knee.vis, ankle.vis))

        thigh_dir_change = np.nan
        shank_dir_change = np.nan
        if prev_thigh_u is not None:
            thigh_dir_change = float(angle_between_unit_vectors(prev_thigh_u, thigh_u))
        if prev_shank_u is not None:
            shank_dir_change = float(angle_between_unit_vectors(prev_shank_u, shank_u))

        rows.append({
            "frame_index": int(frame.frame_index),
            "min_visibility": min_visibility,
            "thigh_len": thigh_len,
            "shank_len": shank_len,
            "thigh_dir_change": thigh_dir_change,
            "shank_dir_change": shank_dir_change,
            "max_dir_change": np.nan if (np.isnan(thigh_dir_change) and np.isnan(shank_dir_change)) else float(np.nanmax([thigh_dir_change, shank_dir_change])),
        })

        prev_thigh_u = thigh_u
        prev_shank_u = shank_u

    feat_df = pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)
    if feat_df.empty:
        return feat_df

    thigh_med = float(np.median(feat_df["thigh_len"].to_numpy(dtype=float)))
    shank_med = float(np.median(feat_df["shank_len"].to_numpy(dtype=float)))
    feat_df["thigh_len_dev"] = (feat_df["thigh_len"] - thigh_med).abs()
    feat_df["shank_len_dev"] = (feat_df["shank_len"] - shank_med).abs()
    return feat_df



def summarize_subset(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "label": label,
            "frames": 0,
            "ratio": 0.0,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_signed_error": np.nan,
            "median_abs_error": np.nan,
            "p95_abs_error": np.nan,
        }
    err = df["error"].to_numpy(dtype=float)
    abs_err = np.abs(err)
    return {
        "label": label,
        "frames": int(len(df)),
        "ratio": float(len(df) / max(len(df.attrs.get("parent_df", df)), 1)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean_signed_error": float(np.mean(err)),
        "median_abs_error": float(np.median(abs_err)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
    }



def summarize_velocity_residual(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "label": label,
            "frames": 0,
            "mean_velocity_residual": np.nan,
            "std_velocity_residual": np.nan,
            "median_velocity_residual": np.nan,
            "mean_abs_velocity_residual": np.nan,
            "rmse_velocity_residual": np.nan,
            "p95_abs_velocity_residual": np.nan,
            "max_abs_velocity_residual": np.nan,
            "fraction_pred_more_negative": np.nan,
            "fraction_pred_less_negative": np.nan,
        }
    resid = df["velocity_residual"].to_numpy(dtype=float)
    abs_resid = np.abs(resid)
    return {
        "label": label,
        "frames": int(len(df)),
        "mean_velocity_residual": float(np.mean(resid)),
        "std_velocity_residual": float(np.std(resid)),
        "median_velocity_residual": float(np.median(resid)),
        "mean_abs_velocity_residual": float(np.mean(abs_resid)),
        "rmse_velocity_residual": float(np.sqrt(np.mean(resid ** 2))),
        "p95_abs_velocity_residual": float(np.percentile(abs_resid, 95)),
        "max_abs_velocity_residual": float(np.max(abs_resid)),
        "fraction_pred_more_negative": float(np.mean(df["pred_angle_velocity"].to_numpy(dtype=float) < df["gt_angle_velocity"].to_numpy(dtype=float))),
        "fraction_pred_less_negative": float(np.mean(df["pred_angle_velocity"].to_numpy(dtype=float) > df["gt_angle_velocity"].to_numpy(dtype=float))),
    }



def build_phase_metrics_df(aligned_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase in PHASE_ORDER:
        subset = aligned_df[aligned_df["gt_phase"] == phase].copy()
        subset.attrs["parent_df"] = aligned_df
        rows.append(summarize_subset(subset, phase))
    return pd.DataFrame(rows)



def build_descent_subphase_metrics_df(aligned_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    descent_df = aligned_df[aligned_df["gt_phase"] == "descent"].copy()
    for subphase in DESCENT_SUBPHASE_ORDER:
        subset = descent_df[descent_df["descent_subphase"] == subphase].copy()
        subset.attrs["parent_df"] = descent_df
        rows.append(summarize_subset(subset, subphase))
    return pd.DataFrame(rows)



def correlation_rows(df: pd.DataFrame, label: str, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        sub = df[[col, "abs_error", "error"]].dropna()
        if len(sub) < 3:
            continue
        rows.append({
            "subset": label,
            "feature": col,
            "pearson_abs_error": float(sub[col].corr(sub["abs_error"])),
            "pearson_signed_error": float(sub[col].corr(sub["error"])),
            "n": int(len(sub)),
        })
    return pd.DataFrame(rows)



def save_angle_plot(aligned_df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(aligned_df["frame_index"], aligned_df["gt_angle"], label="GT angle")
    plt.plot(aligned_df["frame_index"], aligned_df["pred_angle"], label="Pred angle")
    plt.xlabel("Frame")
    plt.ylabel("Knee angle (deg)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def save_error_plot(aligned_df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(aligned_df["frame_index"], aligned_df["abs_error"], label="Absolute error")
    plt.xlabel("Frame")
    plt.ylabel("Absolute error (deg)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def save_phase_label_plot(aligned_df: pd.DataFrame, out_path: Path, title: str) -> None:
    phase_colors = {
        "top": "tab:blue",
        "descent": "tab:orange",
        "bottom": "tab:red",
        "ascent": "tab:green",
    }
    plt.figure(figsize=(16, 7))
    plt.plot(aligned_df["frame_index"], aligned_df["gt_angle"], color="black", linewidth=1.2, label="GT angle")
    plt.plot(aligned_df["frame_index"], aligned_df["pred_angle"], color="0.6", linewidth=1.0, label="Pred angle")
    for phase in PHASE_ORDER:
        mask = aligned_df["gt_phase"] == phase
        if mask.any():
            plt.scatter(
                aligned_df.loc[mask, "frame_index"],
                aligned_df.loc[mask, "gt_angle"],
                s=10,
                color=phase_colors[phase],
                label=f"GT phase={phase}",
            )
    plt.xlabel("Frame")
    plt.ylabel("Knee angle (deg)")
    plt.title(title)
    plt.legend(loc="lower left", ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def save_descent_subphase_overlay(aligned_df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(16, 7))
    plt.plot(aligned_df["frame_index"], aligned_df["gt_angle"], label="GT angle")
    plt.plot(aligned_df["frame_index"], aligned_df["pred_angle"], label="Pred angle")
    early = aligned_df[aligned_df["descent_subphase"] == "descent_early"]
    late = aligned_df[aligned_df["descent_subphase"] == "descent_late"]
    if not early.empty:
        plt.scatter(early["frame_index"], early["gt_angle"], s=10, label="descent_early")
    if not late.empty:
        plt.scatter(late["frame_index"], late["gt_angle"], s=10, label="descent_late")
    plt.xlabel("Frame")
    plt.ylabel("Knee angle (deg)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def save_velocity_vs_frame(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(16, 7))
    plt.plot(df["frame_index"], df["gt_angle_velocity"], label="GT angle velocity")
    plt.plot(df["frame_index"], df["pred_angle_velocity"], label="Pred angle velocity")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Frame")
    plt.ylabel("Velocity (deg/frame)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def save_velocity_scatter(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 8))
    plt.scatter(df["gt_angle_velocity"], df["pred_angle_velocity"], s=18, alpha=0.7)
    plt.xlabel("gt_angle_velocity")
    plt.ylabel("pred_angle_velocity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def save_scatter(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str) -> None:
    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        return
    plt.figure(figsize=(10, 8))
    plt.scatter(sub[x_col], sub[y_col], s=18, alpha=0.7)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def save_residual_histogram(df: pd.DataFrame, out_path: Path, title: str) -> None:
    sub = df[["velocity_residual"]].dropna()
    if sub.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(sub["velocity_residual"], bins=40)
    plt.xlabel("velocity_residual (pred - gt)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def print_summary_table(title: str, df: pd.DataFrame, label_col: str) -> None:
    print("\n" + title)
    for _, row in df.iterrows():
        print(
            f"{str(row[label_col]):>12}: frames={int(row['frames']):4d} "
            f"ratio={float(row['ratio']):.3f} mae={float(row['mae']):.4f} "
            f"rmse={float(row['rmse']):.4f} mean_signed_error={float(row['mean_signed_error']):.4f}"
        )



def print_velocity_residual_summary(title: str, stats: dict) -> None:
    print("\n" + title)
    print(f"      frames: {stats['frames']}")
    print(f"   mean_resid: {stats['mean_velocity_residual']:.6f}")
    print(f"    std_resid: {stats['std_velocity_residual']:.6f}")
    print(f" median_resid: {stats['median_velocity_residual']:.6f}")
    print(f"mean_abs_resid: {stats['mean_abs_velocity_residual']:.6f}")
    print(f"   rmse_resid: {stats['rmse_velocity_residual']:.6f}")
    print(f"p95_abs_resid: {stats['p95_abs_velocity_residual']:.6f}")
    print(f"max_abs_resid: {stats['max_abs_velocity_residual']:.6f}")
    print(f"fraction_pred_more_negative: {stats['fraction_pred_more_negative']:.6f}")
    print(f"fraction_pred_less_negative: {stats['fraction_pred_less_negative']:.6f}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt", required=True, help="GT file path (.csv / .xlsx / .json / .npy)")
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=[
            "mp_world_raw",
            "mp_world_smooth",
            "mp_world_len_stable",
            "mp_world_visibility_filtered",
            "mp_world_outlier_corrected",
            "mp_world_selective_repair",
            "mp_world_selective_burst_repair",
            "mp_world_selective_hinge_repair",
            "mp_world_selective_descent_repair",
        ],
    )
    parser.add_argument("--gt-frame-col", default="frame_index")
    parser.add_argument("--gt-angle-col", default="gt_angle")
    parser.add_argument("--output-dir", default="outputs/world_eval")
    parser.add_argument("--force-side", choices=["left", "right"], default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--store-debug-images", action="store_true")
    parser.add_argument("--store-depth-maps", action="store_true")
    parser.add_argument(
        "--gt-format",
        choices=["table", "npy_3d"],
        default="table",
        help="GT format: table(csv/xlsx/json with angle column) or npy_3d(joint xyz sequence)",
    )
    parser.add_argument("--gt-hip-idx", type=int, default=None)
    parser.add_argument("--gt-knee-idx", type=int, default=None)
    parser.add_argument("--gt-ankle-idx", type=int, default=None)
    parser.add_argument("--phase-smooth-window", type=int, default=5)
    parser.add_argument("--phase-top-band-ratio", type=float, default=0.15)
    parser.add_argument("--phase-bottom-band-ratio", type=float, default=0.85)
    parser.add_argument("--phase-motion-eps-ratio", type=float, default=0.01)
    parser.add_argument("--top-k-error-frames", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.algorithm
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = build_analyzer(args.algorithm)
    output_video_path = str(output_dir / f"{args.algorithm}_overlay.mp4") if args.save_video else None

    video_info = analyzer.process_video(
        video_path=args.video,
        output_path=output_video_path,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        force_side=args.force_side,
        save_video=args.save_video,
        store_debug_images=args.store_debug_images,
        store_depth_maps=args.store_depth_maps,
    )

    final_count, final_angles, extra = analyzer.finalize_analysis()

    gt_df = load_gt_table(
        gt_path=args.gt,
        frame_col=args.gt_frame_col,
        angle_col=args.gt_angle_col,
        gt_format=args.gt_format,
        gt_hip_idx=args.gt_hip_idx,
        gt_knee_idx=args.gt_knee_idx,
        gt_ankle_idx=args.gt_ankle_idx,
    )
    gt_df = label_gt_phases(
        gt_df,
        smooth_window=args.phase_smooth_window,
        top_band_ratio=args.phase_top_band_ratio,
        bottom_band_ratio=args.phase_bottom_band_ratio,
        motion_eps_ratio=args.phase_motion_eps_ratio,
    )

    pred_df = build_prediction_df(analyzer, final_angles)
    feature_df = build_world_feature_df(analyzer)

    aligned_df = pd.merge(gt_df, pred_df, on="frame_index", how="inner")
    if not feature_df.empty:
        aligned_df = pd.merge(aligned_df, feature_df, on="frame_index", how="left")
    if aligned_df.empty:
        raise RuntimeError("No aligned frames between GT and prediction.")

    aligned_df = aligned_df.sort_values("frame_index").reset_index(drop=True)
    aligned_df["error"] = aligned_df["pred_angle"] - aligned_df["gt_angle"]
    aligned_df["abs_error"] = aligned_df["error"].abs()
    aligned_df["sq_error"] = aligned_df["error"] ** 2
    aligned_df["pred_angle_velocity"] = gradient_by_frame(aligned_df["frame_index"].to_numpy(dtype=float), aligned_df["pred_angle"].to_numpy(dtype=float))
    aligned_df["pred_angle_acceleration"] = gradient_by_frame(aligned_df["frame_index"].to_numpy(dtype=float), aligned_df["pred_angle_velocity"].to_numpy(dtype=float))
    aligned_df["velocity_residual"] = aligned_df["pred_angle_velocity"] - aligned_df["gt_angle_velocity"]
    aligned_df["abs_velocity_residual"] = aligned_df["velocity_residual"].abs()

    metrics = compute_metrics(aligned_df)
    phase_metrics_df = build_phase_metrics_df(aligned_df)
    descent_subphase_metrics_df = build_descent_subphase_metrics_df(aligned_df)

    feature_cols = [
        "gt_angle_velocity",
        "pred_angle_velocity",
        "gt_angle_acceleration",
        "pred_angle_acceleration",
        "velocity_residual",
        "abs_velocity_residual",
        "normalized_depth",
        "min_visibility",
        "thigh_len",
        "shank_len",
        "thigh_len_dev",
        "shank_len_dev",
        "thigh_dir_change",
        "shank_dir_change",
        "max_dir_change",
    ]
    corr_all = correlation_rows(aligned_df, "all", feature_cols)
    corr_descent = correlation_rows(aligned_df[aligned_df["gt_phase"] == "descent"], "descent", feature_cols)
    corr_descent_late = correlation_rows(aligned_df[aligned_df["descent_subphase"] == "descent_late"], "descent_late", feature_cols)
    feature_corr_df = pd.concat([corr_all, corr_descent, corr_descent_late], ignore_index=True)

    descent_df = aligned_df[aligned_df["gt_phase"] == "descent"].copy()
    descent_late_df = aligned_df[aligned_df["descent_subphase"] == "descent_late"].copy()
    descent_top_error_df = descent_df.nlargest(args.top_k_error_frames, "abs_error")
    descent_late_top_error_df = descent_late_df.nlargest(args.top_k_error_frames, "abs_error")

    velocity_residual_all = summarize_velocity_residual(aligned_df, "all")
    velocity_residual_descent = summarize_velocity_residual(descent_df, "descent")
    velocity_residual_descent_late = summarize_velocity_residual(descent_late_df, "descent_late")
    velocity_residual_df = pd.DataFrame([
        velocity_residual_all,
        velocity_residual_descent,
        velocity_residual_descent_late,
    ])

    pred_df.to_csv(output_dir / "pred_angles.csv", index=False, encoding="utf-8-sig")
    aligned_df.to_csv(output_dir / "aligned_angles_with_phase_features.csv", index=False, encoding="utf-8-sig")
    phase_metrics_df.to_csv(output_dir / "phase_metrics.csv", index=False, encoding="utf-8-sig")
    descent_subphase_metrics_df.to_csv(output_dir / "descent_subphase_metrics.csv", index=False, encoding="utf-8-sig")
    feature_corr_df.to_csv(output_dir / "feature_error_correlation.csv", index=False, encoding="utf-8-sig")
    descent_top_error_df.to_csv(output_dir / "descent_top_abs_error_frames.csv", index=False, encoding="utf-8-sig")
    descent_late_top_error_df.to_csv(output_dir / "descent_late_top_abs_error_frames.csv", index=False, encoding="utf-8-sig")
    velocity_residual_df.to_csv(output_dir / "velocity_residual_summary.csv", index=False, encoding="utf-8-sig")

    metrics_payload = {
        "algorithm": args.algorithm,
        "video": args.video,
        "gt": args.gt,
        "side": video_info["side"],
        "num_processed_frames": video_info["num_processed_frames"],
        "process_video_count": video_info["count"],
        "final_count": final_count,
        "num_final_angles": len(final_angles),
        "metrics": metrics,
        "phase_metrics": phase_metrics_df.to_dict(orient="records"),
        "descent_subphase_metrics": descent_subphase_metrics_df.to_dict(orient="records"),
        "velocity_residual_summary": velocity_residual_df.to_dict(orient="records"),
        "phase_params": {
            "phase_smooth_window": args.phase_smooth_window,
            "phase_top_band_ratio": args.phase_top_band_ratio,
            "phase_bottom_band_ratio": args.phase_bottom_band_ratio,
            "phase_motion_eps_ratio": args.phase_motion_eps_ratio,
        },
        "extra": extra,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    save_angle_plot(aligned_df, output_dir / "angle_vs_gt.png", title=f"{args.algorithm} vs GT angle")
    save_error_plot(aligned_df, output_dir / "framewise_abs_error.png", title=f"{args.algorithm} frame-wise absolute error")
    save_phase_label_plot(aligned_df, output_dir / "phase_labeled_gt.png", title=f"{args.algorithm}: GT phase labeling (top/descent/bottom/ascent)")
    save_descent_subphase_overlay(aligned_df, output_dir / "descent_subphase_overlay.png", title=f"{args.algorithm}: descent early vs late")
    save_velocity_vs_frame(aligned_df, output_dir / "velocity_vs_frame.png", title=f"{args.algorithm}: GT vs Pred angle velocity")
    save_velocity_vs_frame(descent_late_df, output_dir / "descent_late_velocity_vs_frame.png", title=f"{args.algorithm}: descent_late GT vs Pred velocity")
    save_velocity_scatter(descent_late_df, output_dir / "descent_late_gt_vs_pred_velocity_scatter.png", title=f"{args.algorithm}: descent_late GT vs Pred velocity")
    save_scatter(descent_late_df, "min_visibility", "abs_error", output_dir / "descent_late_error_vs_visibility.png", title=f"{args.algorithm}: descent_late abs error vs visibility")
    save_scatter(descent_late_df, "max_dir_change", "abs_error", output_dir / "descent_late_error_vs_dir_change.png", title=f"{args.algorithm}: descent_late abs error vs dir change")
    save_scatter(descent_late_df, "frame_index", "velocity_residual", output_dir / "descent_late_velocity_residual_vs_frame.png", title=f"{args.algorithm}: descent_late velocity residual vs frame")
    save_residual_histogram(descent_late_df, output_dir / "descent_late_velocity_residual_hist.png", title=f"{args.algorithm}: descent_late velocity residual histogram")

    print("=" * 80)
    print(f"Algorithm          : {args.algorithm}")
    print(f"Video              : {args.video}")
    print(f"GT                 : {args.gt}")
    print(f"Side               : {video_info['side']}")
    print(f"Processed frames   : {video_info['num_processed_frames']}")
    print(f"Final angle frames : {len(final_angles)}")
    print(f"Aligned frames     : {metrics['aligned_frames']}")
    print(f"MAE                : {metrics['mae']:.6f}")
    print(f"RMSE               : {metrics['rmse']:.6f}")
    print(f"Median Abs Error   : {metrics['median_abs_error']:.6f}")
    print(f"P95 Abs Error      : {metrics['p95_abs_error']:.6f}")
    print(f"Max Error          : {metrics['max_error']:.6f}")
    print(f"Output dir         : {output_dir}")

    print_summary_table("gt_phase summary", phase_metrics_df, "label")
    print_summary_table("descent_subphase summary", descent_subphase_metrics_df, "label")
    print_velocity_residual_summary("velocity residual summary (all)", velocity_residual_all)
    print_velocity_residual_summary("velocity residual summary (descent)", velocity_residual_descent)
    print_velocity_residual_summary("velocity residual summary (descent_late)", velocity_residual_descent_late)


if __name__ == "__main__":
    main()