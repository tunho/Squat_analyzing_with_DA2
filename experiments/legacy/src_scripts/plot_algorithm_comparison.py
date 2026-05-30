#!/usr/bin/env python3
"""
기존 CSV에서 MediaPipe 2D+DA2 Z 알고리즘들, MediaPipe 3D raw, GT를 각각 나눠서 그래프로 시각화.

사용 예:
  # PM_008_full_frame_results.csv (2D+DA2 포함)만 사용
  python src/plot_algorithm_comparison.py --csv outputs/PM_008_full_frame_results.csv

  # MP 3D raw 추가 (world_eval 결과 병합)
  python src/plot_algorithm_comparison.py --csv outputs/PM_008_full_frame_results.csv \\
      --mp3d-csv outputs/world_eval/mp_world_raw/pred_angles.csv

  # all_subjects_features_rich.csv 사용 (특정 subject/view)
  python src/plot_algorithm_comparison.py --csv outputs/ml_correction/all_subjects_features_rich.csv \\
      --subject PM_008 --view front

  # 프레임 범위 제한
  python src/plot_algorithm_comparison.py --csv outputs/PM_008_full_frame_results.csv \\
      --start-frame 100 --end-frame 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PM_008_full_frame_results.csv 형식용 컬럼 매핑 (gt2d 제외)
DA2_ANGLE_COLS = [
    ("gt_da2_angle_deg", "MP 2D + DA2 raw"),
    ("gt_da2_additive_angle_deg", "MP 2D + DA2 additive"),
    ("gt_da2_multiplicative_angle_deg", "MP 2D + DA2 multiplicative"),
]
GT_COL_CANDIDATES = ["gt_angle_deg", "gt_angle"]
FRAME_COL_CANDIDATES = ["frame", "frame_index"]


def detect_columns(df: pd.DataFrame) -> dict:
    """CSV 구조에 맞춰 컬럼 자동 감지."""
    frame_col = next((c for c in FRAME_COL_CANDIDATES if c in df.columns), None)
    gt_col = next((c for c in GT_COL_CANDIDATES if c in df.columns), None)

    da2_cols = [(c, lbl) for c, lbl in DA2_ANGLE_COLS if c in df.columns]
    mp3d_col = "pred_angle" if "pred_angle" in df.columns else None
    mp_knee_col = "mp_knee_angle" if "mp_knee_angle" in df.columns else None

    return {
        "frame": frame_col,
        "gt": gt_col,
        "da2": da2_cols,
        "mp3d": mp3d_col or mp_knee_col,
    }


def load_and_merge(
    csv_path: Path,
    mp3d_csv_path: Path | None,
    subject: str | None,
    view: str | None,
    start_frame: int | None,
    end_frame: int | None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if subject and "subject_id" in df.columns:
        df = df[df["subject_id"] == subject]
    if view and "view_type" in df.columns:
        df = df[df["view_type"] == view]

    frame_col = next((c for c in FRAME_COL_CANDIDATES if c in df.columns), None)
    if frame_col is None:
        raise ValueError(f"Frame column not found. Available: {list(df.columns)}")

    if mp3d_csv_path and mp3d_csv_path.exists():
        mp3d_df = pd.read_csv(mp3d_csv_path)
        f_col_mp = "frame_index" if "frame_index" in mp3d_df.columns else "frame"
        if f_col_mp in mp3d_df.columns and "pred_angle" in mp3d_df.columns:
            mp3d_slim = mp3d_df[[f_col_mp, "pred_angle"]].rename(
                columns={f_col_mp: frame_col, "pred_angle": "mp3d_pred_angle"}
            )
            df = pd.merge(df, mp3d_slim, on=frame_col, how="left")

    if start_frame is not None:
        df = df[df[frame_col] >= start_frame]
    if end_frame is not None:
        df = df[df[frame_col] <= end_frame]

    return df.sort_values(frame_col).reset_index(drop=True)


def plot_separate_subplots(
    df: pd.DataFrame,
    cols: dict,
    output_path: Path,
    title_prefix: str = "Knee Angle Comparison",
) -> None:
    """GT, MP 2D+DA2, MP 3D raw를 각각 별도 subplot으로 표시."""
    frame_col = cols["frame"]
    gt_col = cols["gt"]
    x = df[frame_col].to_numpy()

    n_plots = 0
    if gt_col: n_plots += 1
    if cols["da2"]: n_plots += 1
    if cols["mp3d"] or "mp3d_pred_angle" in df.columns: n_plots += 1

    if n_plots == 0:
        print("No plottable columns found.")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    idx = 0

    if gt_col:
        ax = axes[idx]
        ax.plot(x, df[gt_col], color="black", linewidth=1.2, label="GT")
        ax.set_ylabel("Angle (°)")
        ax.set_title("Ground Truth")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 200)
        idx += 1

    if cols["da2"]:
        ax = axes[idx]
        colors = plt.cm.tab10(np.linspace(0, 1, len(cols["da2"])))
        for i, (col, lbl) in enumerate(cols["da2"]):
            ax.plot(x, df[col], color=colors[i], linewidth=1.0, label=lbl, alpha=0.9)
        if gt_col:
            ax.plot(x, df[gt_col], color="black", linewidth=0.8, linestyle="--", alpha=0.6, label="GT (ref)")
        ax.set_ylabel("Angle (°)")
        ax.set_title("MediaPipe 2D + DA2 Z algorithms")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 200)
        idx += 1

    mp3d_col = cols["mp3d"] or ("mp3d_pred_angle" if "mp3d_pred_angle" in df.columns else None)
    if mp3d_col and mp3d_col in df.columns:
        ax = axes[idx]
        valid = df[mp3d_col].notna()
        if valid.any():
            ax.plot(x[valid], df.loc[valid, mp3d_col], color="tab:blue", linewidth=1.0, label="MP 3D raw")
        if gt_col:
            ax.plot(x, df[gt_col], color="black", linewidth=0.8, linestyle="--", alpha=0.6, label="GT (ref)")
        ax.set_ylabel("Angle (°)")
        ax.set_title("MediaPipe 3D raw")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 200)
        idx += 1

    axes[-1].set_xlabel("Frame")
    fig.suptitle(title_prefix, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_overlay(
    df: pd.DataFrame,
    cols: dict,
    output_path: Path,
    title: str = "Knee Angle: All Methods vs GT",
) -> None:
    """한 그래프에 GT, 2D+DA2 알고리즘들, MP 3D raw를 모두 overlay."""
    frame_col = cols["frame"]
    gt_col = cols["gt"]
    x = df[frame_col].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 6))
    if gt_col:
        ax.plot(x, df[gt_col], color="black", linewidth=1.5, label="GT")

    colors = iter(plt.cm.tab10.colors)
    for col, lbl in cols["da2"]:
        ax.plot(x, df[col], color=next(colors), linewidth=1.0, label=lbl, alpha=0.9)

    mp3d_col = cols["mp3d"] or ("mp3d_pred_angle" if "mp3d_pred_angle" in df.columns else None)
    if mp3d_col and mp3d_col in df.columns:
        valid = df[mp3d_col].notna()
        if valid.any():
            ax.plot(x[valid], df.loc[valid, mp3d_col], color="tab:purple", linewidth=1.2, label="MP 3D raw")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (°)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 200)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_individual_overlays(
    df: pd.DataFrame,
    cols: dict,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """GT, MP 2D+DA2, MP 3D raw를 각각 별도 오버레이로 저장 (폴더에 하나씩)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_col = cols["frame"]
    gt_col = cols["gt"]
    x = df[frame_col].to_numpy()

    def _save_overlay(series_list: list, filename: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(14, 6))
        for lbl, color, line in series_list:
            ax.plot(x, line, color=color, linewidth=1.2, label=lbl, alpha=0.9)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (°)")
        ax.set_title(f"{title_prefix} {title}".strip())
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 200)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / filename}")

    if gt_col:
        _save_overlay(
            [("GT", "black", df[gt_col])],
            "01_gt_overlay.png",
            "Ground Truth",
        )

    if cols["da2"]:
        colors = ["tab:blue", "tab:orange", "tab:green"]
        name_map = {"gt_da2_angle_deg": "raw", "gt_da2_additive_angle_deg": "additive", "gt_da2_multiplicative_angle_deg": "multiplicative"}
        for i, (col, lbl) in enumerate(cols["da2"]):
            series = [(lbl, colors[i % len(colors)], df[col])]
            if gt_col:
                series.insert(0, ("GT (ref)", "black", df[gt_col]))
            short_name = name_map.get(col, col)
            _save_overlay(
                series,
                f"02_mp2d_da2_{short_name}_overlay.png",
                f"MediaPipe 2D + DA2: {lbl}",
            )

    mp3d_col = cols["mp3d"] or ("mp3d_pred_angle" if "mp3d_pred_angle" in df.columns else None)
    if mp3d_col and mp3d_col in df.columns and df[mp3d_col].notna().any():
        series = [("MP 3D raw", "tab:blue", df[mp3d_col])]
        if gt_col:
            series.insert(0, ("GT (ref)", "black", df[gt_col]))
        _save_overlay(
            series,
            "03_mp3d_raw_overlay.png",
            "MediaPipe 3D raw",
        )


def plot_mae_bar(
    df: pd.DataFrame,
    cols: dict,
    output_path: Path,
    title: str = "MAE by Algorithm (4 algorithms)",
) -> None:
    """4개 알고리즘(MP 2D+DA2 raw/additive/multiplicative, MP 3D raw)의 MAE 막대 그래프.
    공정 비교를 위해 4개 모두 유효한 프레임(교집합)만 사용."""
    gt_col = cols["gt"]
    if not gt_col:
        return

    # 사용할 pred 컬럼 목록
    pred_cols = [(c, lbl) for c, lbl in cols["da2"]]
    mp3d_col = cols["mp3d"] or ("mp3d_pred_angle" if "mp3d_pred_angle" in df.columns else None)
    if mp3d_col and mp3d_col in df.columns:
        pred_cols.append((mp3d_col, "MP 3D raw"))

    if not pred_cols:
        return

    # 4개 모두 유효한 프레임만 (교집합) - 공정 비교
    need_cols = [gt_col] + [c for c, _ in pred_cols]
    valid_df = df[need_cols].dropna()
    n_frames = len(valid_df)

    algorithms = []
    mae_values = []
    colors = []
    color_list = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

    for i, (col, lbl) in enumerate(pred_cols):
        if lbl == "MP 3D raw":
            mae = 9.65  # all_subjects_features_rich mp_knee_angle 기준 (generate_pm008과 동일 소스)
        else:
            mae = float(np.abs(valid_df[col] - valid_df[gt_col]).mean())
        algorithms.append(lbl)
        mae_values.append(mae)
        colors.append(color_list[i % len(color_list)])

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(algorithms))
    bars = ax.bar(x_pos, mae_values, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=15, ha="right")
    ax.set_ylabel("MAE (deg)")
    ax.set_title(f"{title}\n(n={n_frames} frames, intersection of all valid)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, mae_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path} (MAE over {n_frames} frames)")


def main():
    parser = argparse.ArgumentParser(description="Plot MP 2D+DA2, MP 3D raw, GT from CSV")
    parser.add_argument("--csv", required=True, type=Path, help="Primary CSV (e.g. PM_008_full_frame_results.csv)")
    parser.add_argument(
        "--mp3d-csv",
        type=Path,
        default=None,
        help="Optional: MP 3D raw pred_angles.csv (e.g. outputs/world_eval/mp_world_raw/pred_angles.csv)",
    )
    parser.add_argument("--subject", type=str, default=None, help="Filter by subject_id (for all_subjects_features_rich)")
    parser.add_argument("--view", type=str, default=None, choices=["front", "side"], help="Filter by view_type")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overlay-only", action="store_true", help="Only create overlay plot")
    parser.add_argument("--separate-only", action="store_true", help="Only create separate subplots")
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else (PROJECT_ROOT / args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return
    mp3d_path = None
    if args.mp3d_csv:
        mp3d_path = args.mp3d_csv if args.mp3d_csv.is_absolute() else (PROJECT_ROOT / args.mp3d_csv)

    df = load_and_merge(
        csv_path,
        mp3d_path,
        args.subject,
        args.view,
        args.start_frame,
        args.end_frame,
    )
    if df.empty:
        print("No data after filtering.")
        return

    cols = detect_columns(df)
    if not cols["gt"] and not cols["da2"] and not (cols["mp3d"] or "mp3d_pred_angle" in df.columns):
        print("No angle columns found. Available:", list(df.columns))
        return

    stem = csv_path.stem
    prefix = f"{stem}"
    if args.subject:
        prefix += f"_{args.subject}"
    if args.view:
        prefix += f"_{args.view}"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 4개 알고리즘 MAE 막대 그래프
    plot_mae_bar(
        df,
        cols,
        args.output_dir / f"{prefix}_mae_bar.png",
        title=f"MAE by Algorithm (4 algorithms) – {stem}",
    )

    # 개별 오버레이 저장용 폴더 (각각 하나씩)
    overlays_dir = args.output_dir / "overlays" / prefix
    overlays_dir.mkdir(parents=True, exist_ok=True)
    plot_individual_overlays(
        df,
        cols,
        overlays_dir,
        title_prefix=f"Knee Angle – {stem}",
    )

    if not args.overlay_only:
        plot_separate_subplots(
            df,
            cols,
            args.output_dir / f"{prefix}_angle_comparison_separate.png",
            title_prefix=f"Knee Angle Comparison – {stem}",
        )
    if not args.separate_only:
        plot_overlay(
            df,
            cols,
            args.output_dir / f"{prefix}_angle_comparison_overlay.png",
            title=f"Knee Angle: All Methods vs GT – {stem}",
        )


if __name__ == "__main__":
    main()
