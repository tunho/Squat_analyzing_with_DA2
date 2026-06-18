"""Causal (real-time / calib-15) LOSO on a strided-3000-capped protocol,
matching the offline (offstd) sampling so offline-vs-causal is apples-to-apples.

The online-safe variant uses enhanced_causal (34 feat: offline-only features
hip_depth_norm/thigh_len_dev/shank_len_dev/leg_ratio/view_is_side removed,
calib-15 normalization baked into the CSV at extraction).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_common import FEATURE_SETS, improvement_pct
from paper_train_loso import prepare_v6_data, run_loso, summarize_frame_set


def strided_cap(df: pd.DataFrame, cap: int = 3000) -> pd.DataFrame:
    """Per subject, strided-subsample to <=cap frames preserving temporal order
    (same rule as the offline offstd protocol)."""
    seqcols = [c for c in ["subject_id", "view_type", "camera"] if c in df.columns]
    df = df.sort_values(seqcols + ["frame_index"]).reset_index(drop=True)
    keep = []
    for _, g in df.groupby("subject_id", sort=False):
        n = len(g)
        if n <= cap:
            keep.append(g)
        else:
            step = int(np.ceil(n / cap))
            keep.append(g.iloc[::step].iloc[:cap])
    return pd.concat(keep, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=15)
    ap.add_argument("--deep-flexion-threshold", type=float, default=110.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    feature_cols = FEATURE_SETS["enhanced_causal"]
    datasets = {
        "AIHub": "../experiments/paper/features_aihub_squat_causal.csv",
        "SUMediPose": "../experiments/paper/features_sumedipose_causal.csv",
        "REHAB": "../experiments/paper/features_ex6_causal.csv",
        "FiT3D": "../experiments/paper/features_fit3d_causal.csv",
    }
    offline_ref = {"AIHub": 44.3, "SUMediPose": 42.6, "REHAB": 34.0, "FiT3D": 41.0}

    rows = []
    for name, csv in datasets.items():
        raw = pd.read_csv(csv)
        capped = strided_cap(raw, args.cap)
        df = prepare_v6_data(capped, feature_set="enhanced_causal")
        print(f"\n=== {name} causal: {raw.shape[0]} -> capped {len(capped)} "
              f"({capped['subject_id'].nunique()} subj, {len(capped)//capped['subject_id'].nunique()}/subj) ===", flush=True)
        loso_df, pred_df = run_loso(df, feature_cols, args.n_estimators, args.max_depth, 42)

        rmae = float(loso_df["mae_raw"].mean())
        cmae = float(loso_df["mae_corrected"].mean())
        imp = improvement_pct(rmae, cmae)
        deep = pred_df[pred_df["gt_angle"] < args.deep_flexion_threshold]
        drows = [summarize_frame_set(g, "d") for _, g in deep.groupby("subject_id")]
        dimp = improvement_pct(float(np.mean([r["mae_raw"] for r in drows])),
                               float(np.mean([r["mae_corrected"] for r in drows])))
        row = {"dataset": name, "subjects": int(loso_df["subject_id"].nunique()),
               "frames": int(len(pred_df)), "mae_raw": round(rmae, 2),
               "mae_corrected": round(cmae, 2), "causal_pct": round(imp, 1),
               "offline_pct": offline_ref[name], "diff_pp": round(imp - offline_ref[name], 1),
               "causal_deep_pct": round(dimp, 1)}
        rows.append(row)
        print(f"  causal +{imp:.1f}% (offline +{offline_ref[name]}%, diff {imp-offline_ref[name]:+.1f}pp) | deep +{dimp:.1f}%", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.output / "causal_vs_offline.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print("\n=== Causal(real-time) vs Offline — LOSO 개선율 ===")
    print(out.to_string(index=False))
    print(f"\nmean causal {out.causal_pct.mean():.1f}% vs offline {out.offline_pct.mean():.1f}% "
          f"(mean diff {out.diff_pp.mean():+.1f}pp)")


if __name__ == "__main__":
    main()
