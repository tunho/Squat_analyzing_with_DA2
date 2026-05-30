from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_common import FEATURE_SETS, load_model_bundle, write_json
from paper_train_loso import prepare_v6_data, predict_corrected, summarize_frame_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained v6 residual model on a feature CSV.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deep-flexion-threshold", type=float, default=110.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    bundle = load_model_bundle(args.model)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_set = bundle.get("feature_set", "v6")
    feature_cols = bundle.get("features", FEATURE_SETS[feature_set])

    df = prepare_v6_data(pd.read_csv(args.features), feature_set=feature_set)
    pred_df = predict_corrected(model, scaler, df, feature_cols)
    pred_df.to_csv(args.output / "predictions.csv", index=False)

    rows = []
    for label, group in pred_df.groupby(["subject_id", "view_type"], dropna=False):
        subject_id, view_type = label
        row = summarize_frame_set(group, f"{subject_id}_{view_type}")
        row["subject_id"] = subject_id
        row["view_type"] = view_type
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(args.output / "metrics.csv", index=False)

    deep_df = pred_df[pred_df["gt_angle"] < args.deep_flexion_threshold].copy()
    deep_rows = []
    if not deep_df.empty:
        for label, group in deep_df.groupby(["subject_id", "view_type"], dropna=False):
            subject_id, view_type = label
            row = summarize_frame_set(group, f"{subject_id}_{view_type}")
            row["subject_id"] = subject_id
            row["view_type"] = view_type
            deep_rows.append(row)
    pd.DataFrame(deep_rows).to_csv(args.output / "deep_flexion_metrics.csv", index=False)

    summary = {
        "model": str(args.model),
        "features": str(args.features),
        "overall": summarize_frame_set(pred_df, "all"),
        "deep_flexion_threshold": float(args.deep_flexion_threshold),
        "deep_flexion_frames": int(len(deep_df)),
    }
    if not deep_df.empty:
        summary["deep_flexion"] = summarize_frame_set(deep_df, "deep_flexion")
    write_json(args.output / "summary_metrics.json", summary)

    print(f"Saved evaluation metrics: {args.output / 'metrics.csv'}")


if __name__ == "__main__":
    main()
