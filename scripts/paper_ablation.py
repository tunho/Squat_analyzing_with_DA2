from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_common import FEATURE_SETS, improvement_pct
from paper_train_loso import prepare_v6_data, run_loso, summarize_frame_set


DEFAULT_FEATURE_SETS = [
    "angle_only",
    "lower_body_current",
    "temporal_angle",
    "coordinate_lag_delta",
    "segment_stability",
    "visibility",
    "view",
]


def mean_deep_metrics(pred_df: pd.DataFrame, threshold: float) -> dict:
    deep_df = pred_df[pred_df["gt_angle"] < threshold].copy()
    if deep_df.empty:
        return {
            "deep_frames": 0,
            "deep_mae_raw": float("nan"),
            "deep_mae_corrected": float("nan"),
            "deep_improvement_pct": float("nan"),
        }
    rows = []
    for _, group in deep_df.groupby("subject_id"):
        rows.append(summarize_frame_set(group, "deep"))
    deep_metrics = pd.DataFrame(rows)
    raw = float(deep_metrics["mae_raw"].mean())
    corrected = float(deep_metrics["mae_corrected"].mean())
    return {
        "deep_frames": int(len(deep_df)),
        "deep_mae_raw": raw,
        "deep_mae_corrected": corrected,
        "deep_improvement_pct": improvement_pct(raw, corrected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature ablation for the paper residual model.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-sets", nargs="*", default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--deep-flexion-threshold", type=float, default=110.0)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.features)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for feature_set in args.feature_sets:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {feature_set}")
        feature_cols = FEATURE_SETS[feature_set]
        df = prepare_v6_data(raw_df, feature_set=feature_set)
        loso_df, pred_df = run_loso(
            df=df,
            feature_cols=feature_cols,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )

        out_dir = args.output / feature_set
        out_dir.mkdir(parents=True, exist_ok=True)
        loso_df.to_csv(out_dir / "loso_metrics.csv", index=False)
        pred_df.to_csv(out_dir / "loso_predictions.csv", index=False)

        raw = float(loso_df["mae_raw"].mean())
        corrected = float(loso_df["mae_corrected"].mean())
        row = {
            "feature_set": feature_set,
            "num_features": len(feature_cols),
            "subjects": int(loso_df["subject_id"].nunique()),
            "frames": int(len(pred_df)),
            "mae_raw": raw,
            "mae_corrected": corrected,
            "rmse_raw": float(loso_df["rmse_raw"].mean()),
            "rmse_corrected": float(loso_df["rmse_corrected"].mean()),
            "improvement_pct": improvement_pct(raw, corrected),
        }
        row.update(mean_deep_metrics(pred_df, args.deep_flexion_threshold))
        summary_rows.append(row)
        print(f"{feature_set}: {raw:.3f} -> {corrected:.3f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary.csv", index=False)
    print(f"Saved ablation summary: {args.output / 'summary.csv'}")


if __name__ == "__main__":
    main()
