from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_train_loso import summarize_frame_set


def summarize_regions(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for label, group in [
        (f"gt_angle < {threshold:g}", df[df["gt_angle"] < threshold]),
        (f"gt_angle >= {threshold:g}", df[df["gt_angle"] >= threshold]),
    ]:
        if group.empty:
            continue
        row = summarize_frame_set(group, label)
        row["region"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def feature_stats(subject_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "h_vis",
        "k_vis",
        "a_vis",
        "min_leg_vis",
        "visibility_drop",
        "thigh_len_dev",
        "shank_len_dev",
        "thigh_dir_change",
        "shank_dir_change",
        "abs_error_raw",
        "abs_error_corrected",
    ]
    rows = []
    for col in cols:
        rows.append({
            "feature": col,
            "subject_mean": float(subject_df[col].mean()),
            "subject_std": float(subject_df[col].std()),
            "others_mean": float(other_df[col].mean()),
            "others_std": float(other_df[col].std()),
        })
    return pd.DataFrame(rows)


def save_timeseries_plot(df: pd.DataFrame, output: Path, subject: str) -> None:
    sample = df.sort_values("frame_index")
    if len(sample) > 1200:
        sample = sample.iloc[:1200]
    plt.figure(figsize=(12, 5))
    plt.plot(sample["frame_index"], sample["gt_angle"], label="GT", linewidth=1.4, color="black")
    plt.plot(sample["frame_index"], sample["mp_knee_angle"], label="Raw", linewidth=1.0, alpha=0.8)
    plt.plot(sample["frame_index"], sample["corrected_angle"], label="Corrected", linewidth=1.0, alpha=0.8)
    plt.xlabel("Frame")
    plt.ylabel("Knee angle")
    plt.title(f"{subject}: GT vs Raw vs Corrected")
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze weak subject-level residual correction cases.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--subject", default="PM_038")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deep-flexion-threshold", type=float, default=110.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions)
    df["abs_error_raw"] = (df["gt_angle"] - df["mp_knee_angle"]).abs()
    df["abs_error_corrected"] = (df["gt_angle"] - df["corrected_angle"]).abs()
    subject_df = df[df["subject_id"] == args.subject].copy()
    other_df = df[df["subject_id"] != args.subject].copy()
    if subject_df.empty:
        raise RuntimeError(f"No rows found for subject {args.subject}")

    view_rows = []
    for view, group in subject_df.groupby("view_type"):
        row = summarize_frame_set(group, view)
        row["subject_id"] = args.subject
        row["view_type"] = view
        view_rows.append(row)
    pd.DataFrame(view_rows).to_csv(args.output / "metrics_by_view.csv", index=False)

    summarize_regions(subject_df, args.deep_flexion_threshold).to_csv(
        args.output / "metrics_by_angle_region.csv",
        index=False,
    )
    feature_stats(subject_df, other_df).to_csv(args.output / "feature_stats_vs_others.csv", index=False)
    subject_df.sort_values(["view_type", "frame_index"]).to_csv(args.output / "subject_timeseries.csv", index=False)

    # Prefer front if present because it is usually the more distorted view; fallback to all rows.
    plot_df = subject_df[subject_df["view_type"] == "front"]
    if plot_df.empty:
        plot_df = subject_df
    save_timeseries_plot(plot_df, args.output / f"{args.subject}_gt_raw_corrected.png", args.subject)
    print(f"Saved failure analysis: {args.output}")


if __name__ == "__main__":
    main()
