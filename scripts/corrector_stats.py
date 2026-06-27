from __future__ import annotations
"""
보정기 통계 엄밀성 (J-BHI):
  1) 제안법(ExtraTrees) subject-mean 향상%의 부트스트랩 95% CI
  2) ExtraTrees vs 각 경쟁 보정기: subject별 보정후 MAE paired Wilcoxon + Holm 다중비교 보정
각 baselines_* 셀의 persubject_<method>.csv 사용(같은 subject로 paired).

사용: ../.venv-paper/bin/python corrector_stats.py --glob "experiments/paper/baselines_*" --out ...
"""
import argparse, glob as _glob, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

METHODS = {  # 파일 → 표기
    "extratrees_residual": "ExtraTrees", "mlp_residual": "MLP", "mlp_direct": "MLPdirect",
    "savgol": "SavGol", "kalman": "Kalman", "smoothnet": "SmoothNet", "tcn": "TCN", "diffusion": "Diffusion",
}
OURS = "extratrees_residual"


def load_cell(d):
    """subject_id 인덱스로 각 method 의 mae_corrected, 공통 mae_raw 반환."""
    out = {}
    raw = None
    for f, _ in METHODS.items():
        p = os.path.join(d, f"persubject_{f}.csv")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        try:
            s = pd.read_csv(p).set_index("subject_id")
        except Exception:
            continue  # 작성 중/빈 파일 스킵
        if "mae_corrected" not in s.columns:
            continue
        out[f] = s["mae_corrected"]
        if raw is None:
            raw = s["mae_raw"]
    return raw, out


def boot_ci(imp, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(imp), (n, len(imp)))
    means = imp[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="experiments/paper/baselines_*")
    ap.add_argument("--out", default="experiments/paper/corrector_stats.csv")
    args = ap.parse_args()

    rows = []
    for d in sorted(_glob.glob(args.glob)):
        if not os.path.isdir(d):
            continue
        raw, mc = load_cell(d)
        if raw is None or OURS not in mc:
            continue
        cell = os.path.basename(d).replace("baselines_", "")
        ours = mc[OURS].reindex(raw.index)
        imp = ((raw - ours) / raw * 100).dropna().to_numpy()
        lo, hi = boot_ci(imp)
        row = {"cell": cell, "n_subj": len(imp),
               "ExtraTrees_imp%": round(float(imp.mean()), 1),
               "CI95": f"[{lo:.1f},{hi:.1f}]"}
        # ExtraTrees vs 경쟁 보정기 (paired Wilcoxon, lower MAE 우월), Holm 보정
        comps, pvals = [], []
        for f in mc:
            if f == OURS:
                continue
            a, b = ours.align(mc[f], join="inner")
            a, b = a.dropna(), b.reindex(a.index)
            mask = a.notna() & b.notna()
            if mask.sum() < 6:
                comps.append(f); pvals.append(np.nan); continue
            try:
                _, p = wilcoxon(a[mask].to_numpy(), b[mask].to_numpy())
            except Exception:
                p = np.nan
            comps.append(f); pvals.append(p)
        # Holm 보정
        order = np.argsort([1e9 if np.isnan(p) else p for p in pvals])
        m = sum(1 for p in pvals if not np.isnan(p))
        adj = [np.nan] * len(pvals)
        k = 0
        for rank, i in enumerate(order):
            if np.isnan(pvals[i]):
                continue
            adj[i] = min(1.0, pvals[i] * (m - k))
            k += 1
        for f, pa in zip(comps, adj):
            sig = "" if np.isnan(pa) else ("*" if pa < 0.05 else "ns")
            mean_ours = ours.mean(); mean_other = mc[f].mean()
            better = "ours<" if mean_ours < mean_other else "ours>"
            row[f"vs_{METHODS[f]}"] = "" if np.isnan(pa) else f"{better}{sig}(p={pa:.1e})"
        rows.append(row)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    print(out.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
