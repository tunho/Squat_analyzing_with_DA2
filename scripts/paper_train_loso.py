from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from paper_common import (
    FEATURE_SETS,
    improvement_pct,
    mae,
    rmse,
    save_model_bundle,
    write_json,
)


def _angle_between_rows(prev: pd.DataFrame, curr: pd.DataFrame, prefix: str) -> pd.Series:
    prev_vec = prev[[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]].to_numpy(dtype=float)
    curr_vec = curr[[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]].to_numpy(dtype=float)
    dot = np.sum(prev_vec * curr_vec, axis=1)
    prev_norm = np.linalg.norm(prev_vec, axis=1)
    curr_norm = np.linalg.norm(curr_vec, axis=1)
    denom = np.maximum(prev_norm * curr_norm, 1e-8)
    cosine = np.clip(dot / denom, -1.0, 1.0)
    return pd.Series(np.degrees(np.arccos(cosine)), index=curr.index)


def prepare_v6_data(df: pd.DataFrame, feature_set: str = "v6") -> pd.DataFrame:
    required = {"subject_id", "view_type", "frame_index", "gt_angle", "mp_knee_angle"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Available: {sorted(FEATURE_SETS)}")

    df = df.sort_values(["subject_id", "view_type", "frame_index"]).copy()

    df["thigh_len"] = np.sqrt(df["k_x"] ** 2 + df["k_y"] ** 2 + df["k_z"] ** 2)
    df["shank_len"] = np.sqrt(
        (df["a_x"] - df["k_x"]) ** 2
        + (df["a_y"] - df["k_y"]) ** 2
        + (df["a_z"] - df["k_z"]) ** 2
    )
    subject_ratio = df.groupby("subject_id")[["thigh_len", "shank_len"]].mean()
    subject_ratio["leg_ratio"] = subject_ratio["thigh_len"] / (subject_ratio["shank_len"] + 1e-6)
    df["leg_ratio"] = df["subject_id"].map(subject_ratio["leg_ratio"])

    group_cols = ["subject_id", "view_type"]
    df["angle_velocity"] = df.groupby(group_cols)["mp_knee_angle"].diff().fillna(0.0)
    df["angle_vel_smooth"] = (
        df.groupby(group_cols)["angle_velocity"]
        .transform(lambda x: x.rolling(5, center=False, min_periods=1).mean())
        .fillna(0.0)
    )
    df["angle_acceleration"] = df.groupby(group_cols)["angle_velocity"].diff().fillna(0.0)
    df["knee_lag1"] = df.groupby(group_cols)["mp_knee_angle"].shift(1)
    df["knee_lag2"] = df.groupby(group_cols)["mp_knee_angle"].shift(2)
    df["view_is_side"] = (df["view_type"] == "side").astype(float)

    for col in ["k_x", "k_y", "k_z", "a_x", "a_y", "a_z"]:
        df[f"{col}_lag1"] = df.groupby(group_cols)[col].shift(1)
        joint, axis = col.split("_")
        df[f"{joint}_d{axis}"] = df[col] - df[f"{col}_lag1"]

    df["min_leg_vis"] = df[["h_vis", "k_vis", "a_vis"]].min(axis=1)
    df["mean_leg_vis"] = df[["h_vis", "k_vis", "a_vis"]].mean(axis=1)
    df["visibility_drop"] = df.groupby(group_cols)["min_leg_vis"].diff().fillna(0.0)
    df["thigh_len_dev"] = df["thigh_len"] - df.groupby("subject_id")["thigh_len"].transform("median")
    df["shank_len_dev"] = df["shank_len"] - df.groupby("subject_id")["shank_len"].transform("median")

    prev = df.groupby(group_cols, group_keys=False).shift(1)
    df["thigh_dir_change"] = _angle_between_rows(prev.rename(columns={"k_x": "k_x", "k_y": "k_y", "k_z": "k_z"}), df, "k").fillna(0.0)
    prev_shank = pd.DataFrame(index=df.index)
    curr_shank = pd.DataFrame(index=df.index)
    for axis in ["x", "y", "z"]:
        prev_shank[f"s_{axis}"] = prev[f"a_{axis}"] - prev[f"k_{axis}"]
        curr_shank[f"s_{axis}"] = df[f"a_{axis}"] - df[f"k_{axis}"]
    df["shank_dir_change"] = _angle_between_rows(prev_shank, curr_shank, "s").fillna(0.0)
    df["residual"] = df["gt_angle"] - df["mp_knee_angle"]

    feature_cols = FEATURE_SETS[feature_set]
    return df.dropna(subset=feature_cols + ["gt_angle", "residual"]).copy()


def fit_model(train_df: pd.DataFrame, feature_cols: list[str], n_estimators: int, max_depth: int, random_state: int):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols])
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, train_df["residual"])
    return model, scaler


def predict_corrected(model, scaler, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    pred_residual = model.predict(scaler.transform(out[feature_cols]))
    out["predicted_residual"] = pred_residual
    out["corrected_angle"] = out["mp_knee_angle"].to_numpy(dtype=float) + pred_residual
    return out


def summarize_frame_set(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    raw_mae = mae(df["gt_angle"], df["mp_knee_angle"])
    corrected_mae = mae(df["gt_angle"], df["corrected_angle"])
    raw_rmse = rmse(df["gt_angle"], df["mp_knee_angle"])
    corrected_rmse = rmse(df["gt_angle"], df["corrected_angle"])
    return {
        "set": label,
        "frames": int(len(df)),
        "mae_raw": raw_mae,
        "mae_corrected": corrected_mae,
        "rmse_raw": raw_rmse,
        "rmse_corrected": corrected_rmse,
        "improvement_pct": improvement_pct(raw_mae, corrected_mae),
    }


def run_loso(df: pd.DataFrame, feature_cols: list[str], n_estimators: int, max_depth: int, random_state: int):
    subjects = sorted(df["subject_id"].unique())
    rows = []
    predictions = []

    for subject in subjects:
        train_df = df[df["subject_id"] != subject]
        test_df = df[df["subject_id"] == subject]
        if train_df.empty or test_df.empty:
            continue

        model, scaler = fit_model(train_df, feature_cols, n_estimators, max_depth, random_state)
        pred_df = predict_corrected(model, scaler, test_df, feature_cols)
        summary = summarize_frame_set(pred_df, str(subject))
        summary["subject_id"] = str(subject)
        rows.append(summary)
        predictions.append(pred_df)

    if not rows:
        raise RuntimeError("No LOSO folds were produced.")

    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the official paper v6 residual LOSO model.")
    parser.add_argument("--features", required=True, type=Path, help="Feature CSV with GT and MediaPipe angles.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for paper LOSO artifacts.")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--final-n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--deep-flexion-threshold", type=float, default=110.0)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="v6")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(args.features)
    feature_cols = FEATURE_SETS[args.feature_set]
    df = prepare_v6_data(raw_df, feature_set=args.feature_set)

    loso_df, pred_df = run_loso(df, feature_cols, args.n_estimators, args.max_depth, args.random_state)

    loso_df.to_csv(args.output / "loso_metrics.csv", index=False)
    pred_df.to_csv(args.output / "loso_predictions.csv", index=False)

    deep_df = pred_df[pred_df["gt_angle"] < args.deep_flexion_threshold].copy()
    deep_rows = []
    if not deep_df.empty:
        for subject, group in deep_df.groupby("subject_id"):
            row = summarize_frame_set(group, str(subject))
            row["subject_id"] = str(subject)
            deep_rows.append(row)
    pd.DataFrame(deep_rows).to_csv(args.output / "deep_flexion_metrics.csv", index=False)

    total = summarize_frame_set(pred_df, "all")
    summary = {
        "method": "v6_single_residual",
        "feature_set": args.feature_set,
        "target": "gt_angle - mp_knee_angle",
        "prediction": "mp_knee_angle + predicted_residual",
        "features": feature_cols,
        "num_subjects": int(loso_df["subject_id"].nunique()),
        "num_frames": int(len(pred_df)),
        "loso_mean_mae_raw": float(loso_df["mae_raw"].mean()),
        "loso_mean_mae_corrected": float(loso_df["mae_corrected"].mean()),
        "loso_mean_rmse_raw": float(loso_df["rmse_raw"].mean()),
        "loso_mean_rmse_corrected": float(loso_df["rmse_corrected"].mean()),
        "loso_improvement_pct": improvement_pct(
            float(loso_df["mae_raw"].mean()),
            float(loso_df["mae_corrected"].mean()),
        ),
        "pooled_metrics": total,
        "deep_flexion_threshold": float(args.deep_flexion_threshold),
    }
    write_json(args.output / "summary_metrics.json", summary)

    final_model, final_scaler = fit_model(df, feature_cols, args.final_n_estimators, args.max_depth, args.random_state)
    save_model_bundle(
        args.output / "v6_single_residual_model.pkl",
        {
            "model": final_model,
            "scaler": final_scaler,
            "features": feature_cols,
            "target": "gt_angle - mp_knee_angle",
            "method": "v6_single_residual",
            "feature_set": args.feature_set,
            "created_from": str(args.features),
            "n_estimators": args.final_n_estimators,
            "max_depth": args.max_depth,
            "random_state": args.random_state,
        },
    )

    print(f"Saved LOSO metrics: {args.output / 'loso_metrics.csv'}")
    print(f"Saved summary: {args.output / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()
