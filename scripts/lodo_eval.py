"""Leave-One-Dataset-Out (LODO) cross-validation.

Train on 3 datasets, test on the entire held-out 4th dataset — measures
generalization to a *completely unseen domain* (vs pooled LOSO, where the
target domain's other subjects are already in training).

Uses the same protocol as the main pooled experiment:
ExtraTrees, enhanced_safe (39 feat), n_estimators=300, max_depth=15.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from paper_common import FEATURE_SETS
from paper_train_loso import (
    prepare_v6_data,
    fit_model,
    predict_corrected,
    summarize_frame_set,
)


def dataset_of(subject_id: str) -> str:
    s = str(subject_id)
    if s.startswith("SUME"):
        return "SUMediPose"
    if s.startswith("fit3d"):
        return "FiT3D"
    if s.startswith("PM"):
        return "REHAB"
    if s[:2] in ("CA", "CB", "CI"):
        return "AIHub"
    return "UNKNOWN"


def main() -> None:
    p = argparse.ArgumentParser(description="Leave-One-Dataset-Out cross-validation.")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--feature-set", default="enhanced_safe", choices=sorted(FEATURE_SETS))
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=15)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--deep-flexion-threshold", type=float, default=110.0)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    feature_cols = FEATURE_SETS[args.feature_set]

    df = prepare_v6_data(pd.read_csv(args.features), feature_set=args.feature_set)
    df["dataset"] = df["subject_id"].map(dataset_of)

    datasets = ["AIHub", "SUMediPose", "REHAB", "FiT3D"]
    fold_rows = []
    all_preds = []

    for held in datasets:
        train_df = df[df["dataset"] != held]
        test_df = df[df["dataset"] == held]
        if train_df.empty or test_df.empty:
            continue
        print(f"[LODO] hold out {held}: train {len(train_df)} frames "
              f"({train_df['subject_id'].nunique()} subj) -> test {len(test_df)} frames "
              f"({test_df['subject_id'].nunique()} subj)", flush=True)

        model, scaler = fit_model(train_df, feature_cols, args.n_estimators,
                                  args.max_depth, args.random_state)
        pred = predict_corrected(model, scaler, test_df, feature_cols)
        all_preds.append(pred)

        # pooled (frame-weighted) over the held-out dataset
        overall = summarize_frame_set(pred, held)
        # per-subject mean improvement within the held-out dataset
        subj = [summarize_frame_set(g, str(sid)) for sid, g in pred.groupby("subject_id")]
        subj_imp = sum(r["improvement_pct"] for r in subj) / len(subj)
        n_improved = sum(1 for r in subj if r["improvement_pct"] > 0)
        # deep flexion
        deep = pred[pred["gt_angle"] < args.deep_flexion_threshold]
        deep_imp = summarize_frame_set(deep, held)["improvement_pct"] if len(deep) else float("nan")

        row = {
            "held_out": held,
            "n_subjects": test_df["subject_id"].nunique(),
            "frames": overall["frames"],
            "mae_raw": overall["mae_raw"],
            "mae_corrected": overall["mae_corrected"],
            "improvement_pooled_pct": overall["improvement_pct"],
            "improvement_subjmean_pct": subj_imp,
            "subjects_improved": f"{n_improved}/{len(subj)}",
            "deep_improvement_pct": deep_imp,
        }
        fold_rows.append(row)
        print(f"        -> raw {overall['mae_raw']:.2f}° corr {overall['mae_corrected']:.2f}° "
              f"| pooled {overall['improvement_pct']:+.1f}% | subjmean {subj_imp:+.1f}% "
              f"| improved {n_improved}/{len(subj)} | deep {deep_imp:+.1f}%", flush=True)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(args.output / "lodo_metrics.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(args.output / "lodo_predictions.csv", index=False)

    summary = {
        "protocol": "leave-one-dataset-out",
        "feature_set": args.feature_set,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "folds": fold_rows,
        "mean_improvement_pooled_pct": float(folds["improvement_pooled_pct"].mean()),
        "mean_improvement_subjmean_pct": float(folds["improvement_subjmean_pct"].mean()),
    }
    (args.output / "summary_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSaved:", args.output / "lodo_metrics.csv")
    print(folds.to_string(index=False))


if __name__ == "__main__":
    main()
