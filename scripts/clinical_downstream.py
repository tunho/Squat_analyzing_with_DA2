"""Downstream clinical metrics — converts per-frame knee-angle MAE into
squat-assessment decisions.

Per squat sequence (subject x view x camera), from GT / raw-MP / corrected:
  - Depth  = min knee angle (deepest point of the squat)
  - ROM    = max - min knee angle (range of motion)
  - Depth classification at parallel (<=90 deg) and other clinical cutoffs.

Shows whether the residual correction improves the actual clinical readout,
not just the frame-level angle error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def dataset_of(s: str) -> str:
    s = str(s)
    if s.startswith("SUME"): return "SUMediPose"
    if s.startswith("fit3d"): return "FiT3D"
    if s.startswith("PM"): return "REHAB"
    return "AIHub"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path, help="loso_predictions.csv")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[90.0, 100.0, 110.0])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    p = pd.read_csv(args.pred)
    seqcols = [c for c in ["subject_id", "view_type", "camera"] if c in p.columns]
    p["dataset"] = p["subject_id"].map(dataset_of)

    # per-sequence depth (min) and ROM (max-min) for each angle source
    def agg(g):
        out = {"dataset": g["dataset"].iloc[0]}
        for src, col in [("gt", "gt_angle"), ("raw", "mp_knee_angle"), ("cor", "corrected_angle")]:
            out[f"depth_{src}"] = g[col].min()
            out[f"rom_{src}"] = g[col].max() - g[col].min()
        return pd.Series(out)

    seq = p.groupby(seqcols).apply(agg, include_groups=False).reset_index()
    seq.to_csv(args.output / "per_sequence.csv", index=False)

    def mae(a, b): return float(np.mean(np.abs(a - b)))

    rows = []
    for scope, sub in [("전체", seq)] + [(d, g) for d, g in seq.groupby("dataset")]:
        rec = {"scope": scope, "n_seq": len(sub)}
        # depth (min-angle) error
        rec["depth_mae_raw"] = round(mae(sub.depth_gt, sub.depth_raw), 2)
        rec["depth_mae_cor"] = round(mae(sub.depth_gt, sub.depth_cor), 2)
        rec["depth_impr_pct"] = round((1 - rec["depth_mae_cor"] / rec["depth_mae_raw"]) * 100, 1)
        # ROM error
        rec["rom_mae_raw"] = round(mae(sub.rom_gt, sub.rom_raw), 2)
        rec["rom_mae_cor"] = round(mae(sub.rom_gt, sub.rom_cor), 2)
        rec["rom_impr_pct"] = round((1 - rec["rom_mae_cor"] / rec["rom_mae_raw"]) * 100, 1)
        rows.append(rec)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "depth_rom_metrics.csv", index=False)

    # depth classification (pass = reached at/below cutoff) accuracy
    cls_rows = []
    for cut in args.cutoffs:
        gt_pass = seq.depth_gt <= cut
        for src in ["raw", "cor"]:
            pred_pass = seq[f"depth_{src}"] <= cut
            acc = float((pred_pass == gt_pass).mean())
            # also error rate among true reaches/misses
            cls_rows.append({"cutoff": cut, "source": src, "accuracy": round(acc, 3),
                             "pass_rate_gt": round(float(gt_pass.mean()), 3),
                             "pass_rate_pred": round(float(pred_pass.mean()), 3)})
    cls = pd.DataFrame(cls_rows)
    cls.to_csv(args.output / "depth_classification.csv", index=False)

    (args.output / "summary.json").write_text(json.dumps(
        {"depth_rom": rows, "classification": cls_rows}, indent=2, ensure_ascii=False))

    print("=== 깊이(최소각)·ROM 오차: raw → corrected ===")
    print(metrics.to_string(index=False))
    print("\n=== 깊이 분류 정확도 (pass = min각 <= cutoff) ===")
    for cut in args.cutoffs:
        c = cls[cls.cutoff == cut]
        r = c[c.source == "raw"].iloc[0]; co = c[c.source == "cor"].iloc[0]
        print(f"  cutoff {cut:.0f}°: raw {r.accuracy:.3f} → corrected {co.accuracy:.3f}  "
              f"(GT pass율 {r.pass_rate_gt:.2f})")


if __name__ == "__main__":
    main()
