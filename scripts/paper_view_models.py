from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_common import FEATURE_SETS, improvement_pct
from paper_train_loso import fit_model, prepare_v6_data, predict_corrected, run_loso, summarize_frame_set


def summarize_by_view(pred_df: pd.DataFrame, model_name: str, threshold: float) -> list[dict]:
    rows = []
    for view, group in pred_df.groupby("view_type"):
        row = summarize_frame_set(group, view)
        row["model"] = model_name
        row["view_type"] = view
        deep = group[group["gt_angle"] < threshold]
        if not deep.empty:
            deep_row = summarize_frame_set(deep, f"{view}_deep")
            row["deep_frames"] = deep_row["frames"]
            row["deep_mae_raw"] = deep_row["mae_raw"]
            row["deep_mae_corrected"] = deep_row["mae_corrected"]
            row["deep_improvement_pct"] = deep_row["improvement_pct"]
        rows.append(row)
    overall = summarize_frame_set(pred_df, "overall")
    overall["model"] = model_name
    overall["view_type"] = "overall"
    rows.append(overall)
    return rows


def run_view_only_loso(df: pd.DataFrame, feature_cols: list[str], view: str, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    view_df = df[df["view_type"] == view].copy()
    return run_loso(view_df, feature_cols, args.n_estimators, args.max_depth, args.random_state)


def run_two_expert_loso(df: pd.DataFrame, feature_cols: list[str], args) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = []
    for subject in sorted(df["subject_id"].unique()):
        subject_preds = []
        for view in sorted(df["view_type"].unique()):
            train_df = df[(df["subject_id"] != subject) & (df["view_type"] == view)]
            test_df = df[(df["subject_id"] == subject) & (df["view_type"] == view)]
            if train_df.empty or test_df.empty:
                continue
            model, scaler = fit_model(train_df, feature_cols, args.n_estimators, args.max_depth, args.random_state)
            subject_preds.append(predict_corrected(model, scaler, test_df, feature_cols))
        if not subject_preds:
            continue
        pred_df = pd.concat(subject_preds, ignore_index=True)
        row = summarize_frame_set(pred_df, str(subject))
        row["subject_id"] = str(subject)
        rows.append(row)
        predictions.append(pred_df)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def save_model_outputs(output: Path, model_name: str, loso_df: pd.DataFrame, pred_df: pd.DataFrame, threshold: float) -> list[dict]:
    out_dir = output / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    loso_df.to_csv(out_dir / "loso_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "loso_predictions.csv", index=False)
    return summarize_by_view(pred_df, model_name, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare unified and view-specific residual models.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="enhanced_safe")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--deep-flexion-threshold", type=float, default=110.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(args.features)
    df = prepare_v6_data(raw_df, feature_set=args.feature_set)
    feature_cols = FEATURE_SETS[args.feature_set]

    summary_rows = []

    loso_df, pred_df = run_loso(df, feature_cols, args.n_estimators, args.max_depth, args.random_state)
    summary_rows.extend(save_model_outputs(args.output, "unified", loso_df, pred_df, args.deep_flexion_threshold))

    for view in ["front", "side"]:
        loso_df, pred_df = run_view_only_loso(df, feature_cols, view, args)
        summary_rows.extend(save_model_outputs(args.output, f"{view}_only", loso_df, pred_df, args.deep_flexion_threshold))

    loso_df, pred_df = run_two_expert_loso(df, feature_cols, args)
    summary_rows.extend(save_model_outputs(args.output, "two_expert", loso_df, pred_df, args.deep_flexion_threshold))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary.csv", index=False)

    compact = []
    for model, group in summary[summary["view_type"] == "overall"].groupby("model"):
        raw = float(group["mae_raw"].iloc[0])
        corrected = float(group["mae_corrected"].iloc[0])
        compact.append({
            "model": model,
            "mae_raw": raw,
            "mae_corrected": corrected,
            "improvement_pct": improvement_pct(raw, corrected),
        })
    pd.DataFrame(compact).to_csv(args.output / "model_summary.csv", index=False)
    print(f"Saved view model summary: {args.output / 'summary.csv'}")


if __name__ == "__main__":
    main()
