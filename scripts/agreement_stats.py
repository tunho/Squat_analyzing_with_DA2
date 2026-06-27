from __future__ import annotations

"""
일치도(agreement) 통계 — J-BHI 필수. 각 loso_predictions.csv(도메인×추정기)에 대해:
  - Bland-Altman: bias(평균차) ± LoA(±1.96·SD), %within-LoA
  - CCC (Lin's concordance) : 일치도(ICC 유사, 임상표준)
  - Pearson r, RMSE, MAE
  - Wilcoxon 검정: |raw오차| vs |보정오차| (보정 유의성)
보정 전(raw=mp_knee_angle vs gt) / 보정 후(corrected vs gt) 둘 다.

사용: ../.venv-paper/bin/python agreement_stats.py --glob "experiments/paper/loso_*" --out ...
"""

import argparse
import glob as _glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, pearsonr


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    return float(2 * cov / (vx + vy + (mx - my) ** 2 + 1e-12))


def icc_2_1(x, y):
    """ICC(2,1): two-way random-effects, single rater, absolute agreement
    (Shrout & Fleiss). 각 프레임=관측(row), 두 측정치=평가자(method, GT, k=2).
    리뷰어가 CCC와 별개로 명시 요구하는 임상 일치도 지표."""
    m = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n, k = m.shape  # k=2
    grand = m.mean()
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)
    # 평방합
    ss_total = ((m - grand) ** 2).sum()
    ss_row = k * ((row_means - grand) ** 2).sum()
    ss_col = n * ((col_means - grand) ** 2).sum()
    ss_err = ss_total - ss_row - ss_col
    msr = ss_row / (n - 1)
    msc = ss_col / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    return float((msr - mse) / (denom + 1e-12))


def bland_altman(est, gt):
    d = np.asarray(est, float) - np.asarray(gt, float)
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
    within = float(np.mean((d >= loa_lo) & (d <= loa_hi)) * 100)
    return bias, sd, loa_lo, loa_hi, within


def stats_block(pred, est_col, gt_col="gt_angle"):
    e, g = pred[est_col].to_numpy(float), pred[gt_col].to_numpy(float)
    bias, sd, lo, hi, within = bland_altman(e, g)
    r = float(pearsonr(e, g)[0])
    return {
        "MAE": float(np.mean(np.abs(e - g))),
        "RMSE": float(np.sqrt(np.mean((e - g) ** 2))),
        "bias": bias, "LoA_lo": lo, "LoA_hi": hi, "within_LoA%": within,
        "CCC": ccc(e, g), "ICC": icc_2_1(e, g), "Pearson_r": r,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="experiments/paper/loso_*")
    ap.add_argument("--out", default="experiments/paper/agreement_stats.csv")
    args = ap.parse_args()

    rows = []
    for d in sorted(_glob.glob(args.glob)):
        f = os.path.join(d, "loso_predictions.csv")
        if not os.path.exists(f):
            continue
        pred = pd.read_csv(f)
        if not {"gt_angle", "mp_knee_angle", "corrected_angle"} <= set(pred.columns):
            continue
        name = os.path.basename(d).replace("loso_", "")
        raw = stats_block(pred, "mp_knee_angle")
        cor = stats_block(pred, "corrected_angle")
        # Wilcoxon: |raw err| vs |corr err|
        ra = np.abs(pred.mp_knee_angle - pred.gt_angle)
        ca = np.abs(pred.corrected_angle - pred.gt_angle)
        try:
            _, pv = wilcoxon(ra, ca)
        except Exception:
            pv = float("nan")
        rows.append({
            "case": name, "n": len(pred),
            "raw_MAE": round(raw["MAE"], 2), "corr_MAE": round(cor["MAE"], 2),
            "raw_bias": round(raw["bias"], 2), "corr_bias": round(cor["bias"], 2),
            "raw_LoA": f"[{raw['LoA_lo']:.1f},{raw['LoA_hi']:.1f}]",
            "corr_LoA": f"[{cor['LoA_lo']:.1f},{cor['LoA_hi']:.1f}]",
            "raw_CCC": round(raw["CCC"], 3), "corr_CCC": round(cor["CCC"], 3),
            "raw_ICC": round(raw["ICC"], 3), "corr_ICC": round(cor["ICC"], 3),
            "raw_r": round(raw["Pearson_r"], 3), "corr_r": round(cor["Pearson_r"], 3),
            "wilcoxon_p": f"{pv:.2e}",
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(out.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
