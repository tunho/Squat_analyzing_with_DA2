"""Few-shot domain recovery curve.

For each held-out target domain D, train on the other 3 datasets PLUS k
randomly-chosen subjects from D, and test on the REMAINING D subjects
(never in training -> no leakage). Sweeps k = 0,1,2,3.

k=0 == leave-one-dataset-out (zero-shot domain). Increasing k shows how
little target supervision is needed. Repeats over random subject draws
give mean +/- std (the uncertainty band for the figure).

Same protocol as the main experiments: enhanced_safe(39), ExtraTrees,
n_estimators=300, max_depth=15.
"""
from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from paper_common import FEATURE_SETS
from paper_train_loso import prepare_v6_data, fit_model, predict_corrected, summarize_frame_set
from lodo_eval import dataset_of


def eval_on(model, scaler, test_df, feature_cols, deep_thr):
    pred = predict_corrected(model, scaler, test_df, feature_cols)
    overall = summarize_frame_set(pred, "test")
    subj = [summarize_frame_set(g, str(s))["improvement_pct"] for s, g in pred.groupby("subject_id")]
    deep = pred[pred["gt_angle"] < deep_thr]
    deep_imp = summarize_frame_set(deep, "deep")["improvement_pct"] if len(deep) else float("nan")
    return {
        "pooled": overall["improvement_pct"],
        "subjmean": float(np.mean(subj)),
        "n_improved": int(sum(1 for x in subj if x > 0)),
        "n_subj": len(subj),
        "deep": deep_imp,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Few-shot domain recovery curve.")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--feature-set", default="enhanced_safe", choices=sorted(FEATURE_SETS))
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=15)
    p.add_argument("--k-values", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--deep-flexion-threshold", type=float, default=110.0)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    feature_cols = FEATURE_SETS[args.feature_set]
    df = prepare_v6_data(pd.read_csv(args.features), feature_set=args.feature_set)
    df["dataset"] = df["subject_id"].map(dataset_of)

    datasets = ["AIHub", "SUMediPose", "REHAB", "FiT3D"]
    rows = []

    for held in datasets:
        base_train = df[df["dataset"] != held]
        target = df[df["dataset"] == held]
        subs = sorted(target["subject_id"].unique())
        n = len(subs)
        print(f"\n=== held-out {held}: {n} subjects ===", flush=True)

        for k in args.k_values:
            if k >= n:   # need at least 1 test subject
                continue
            if k == 0:
                draws = [()]                       # zero-shot, deterministic
            else:
                all_combos = list(combinations(subs, k))
                if len(all_combos) <= args.repeats:
                    draws = all_combos             # enumerate all if few
                else:
                    rng = random.Random(args.random_state + k)
                    draws = [tuple(rng.sample(subs, k)) for _ in range(args.repeats)]

            res = []
            for chosen in draws:
                chosen = set(chosen)
                test_df = target[~target["subject_id"].isin(chosen)]
                train_df = pd.concat([base_train, target[target["subject_id"].isin(chosen)]])
                model, scaler = fit_model(train_df, feature_cols, args.n_estimators,
                                          args.max_depth, args.random_state)
                res.append(eval_on(model, scaler, test_df, feature_cols, args.deep_flexion_threshold))

            pooled = np.array([r["pooled"] for r in res])
            deep = np.array([r["deep"] for r in res])
            subjm = np.array([r["subjmean"] for r in res])
            row = {
                "held_out": held, "k": k, "n_draws": len(draws),
                "n_test_subj": res[0]["n_subj"],
                "pooled_mean": float(pooled.mean()), "pooled_std": float(pooled.std()),
                "subjmean_mean": float(subjm.mean()),
                "deep_mean": float(np.nanmean(deep)), "deep_std": float(np.nanstd(deep)),
            }
            rows.append(row)
            print(f"  k={k} (draws={len(draws)}): pooled {row['pooled_mean']:+.1f}±{row['pooled_std']:.1f}% "
                  f"| deep {row['deep_mean']:+.1f}±{row['deep_std']:.1f}%", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.output / "fewshot_curve.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print("\nSaved:", args.output / "fewshot_curve.csv")
    print(out.to_string(index=False))

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for held in datasets:
            sub = out[out["held_out"] == held].sort_values("k")
            ax.errorbar(sub["k"], sub["pooled_mean"], yerr=sub["pooled_std"],
                        marker="o", capsize=3, label=held)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("# target-domain subjects in training (k)")
        ax.set_ylabel("MAE improvement on held-out test subjects (%)")
        ax.set_title("Few-shot domain recovery (k=0 is zero-shot / LODO)")
        ax.set_xticks(sorted(out["k"].unique()))
        ax.legend()
        fig.tight_layout()
        figpath = args.output / "recovery.png"
        fig.savefig(figpath, dpi=150)
        print("Saved figure:", figpath)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
