#!/usr/bin/env python
"""multiseed_out/ 를 걸어 (보정기 × 셀)별 mean±std 개선율 표를 만든다.

각 seed 의 persubject_*.csv 에서 subject-mean improvement_pct 를 구하고,
시드들에 대해 mean±std 를 계산 → multiseed_summary.csv + 콘솔 표.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "experiments" / "paper" / "multiseed_out"

def main():
    rows = []
    for corr_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        corr = corr_dir.name
        for cell_dir in sorted(p for p in corr_dir.iterdir() if p.is_dir()):
            cell = cell_dir.name
            per_seed = {}
            for seed_dir in sorted(cell_dir.glob("seed*")):
                seed = int(re.sub(r"\D", "", seed_dir.name))
                csvs = list(seed_dir.glob("persubject_*.csv"))
                if not csvs:
                    continue
                df = pd.read_csv(csvs[0])
                if "improvement_pct" not in df.columns or df.empty:
                    continue
                per_seed[seed] = float(df["improvement_pct"].mean())  # subject-mean
            if not per_seed:
                continue
            vals = np.array(sorted(per_seed.values()))
            rows.append({
                "corrector": corr, "cell": cell, "n_seeds": len(vals),
                "mean_impr": round(float(vals.mean()), 2),
                "std_impr": round(float(vals.std(ddof=0)), 2),
                "min": round(float(vals.min()), 2), "max": round(float(vals.max()), 2),
                "seeds": ",".join(f"{s}:{per_seed[s]:+.1f}" for s in sorted(per_seed)),
            })
    if not rows:
        print("집계할 결과 없음 (multiseed_out 비어있음)."); return
    out = pd.DataFrame(rows).sort_values(["corrector", "cell"])
    dst = OUT / "multiseed_summary.csv"
    out.to_csv(dst, index=False)
    # 콘솔 표: 보정기 × 셀, mean±std
    print(f"\n=== multiseed 개선율 (mean±std over seeds) ===")
    for corr, g in out.groupby("corrector"):
        print(f"\n[{corr}]")
        for _, r in g.iterrows():
            print(f"  {r['cell']:<18} {r['mean_impr']:+6.1f} ± {r['std_impr']:<4} %   "
                  f"(n={r['n_seeds']}, range {r['min']:+.0f}~{r['max']:+.0f})")
    print(f"\n저장: {dst}")

if __name__ == "__main__":
    main()
