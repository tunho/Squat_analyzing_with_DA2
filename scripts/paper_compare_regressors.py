"""여러 회귀기 + 단순 선형 baseline 을 동일 LOSO 조건에서 비교.

같은 prepare_v6_data 피처/스케일러/잔차목표를 쓰고 회귀기만 교체한다.
전체(mean) 및 깊은굴곡(<110) 개선율을 피험자단위 LOSO-mean 으로 보고.

PYTHONPATH=src .venv-paper/bin/python scripts/paper_compare_regressors.py \
  --features experiments/paper/features_aihub_squat.csv \
  --feature-set enhanced_safe \
  --output outputs/paper/regressor_compare_aihub
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from paper_common import FEATURE_SETS
from paper_train_loso import prepare_v6_data


def build_regressors(seed: int) -> dict:
    """빠른 모델부터(실시간 진행 확인용). 트리 앙상블은 200개로 비교(상대순위 충분)."""
    regs = {"Linear(Ridge)": lambda: Ridge(alpha=1.0)}
    try:
        from lightgbm import LGBMRegressor
        regs["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=400, num_leaves=63, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
    except Exception:
        pass
    try:
        from xgboost import XGBRegressor
        regs["XGBoost"] = lambda: XGBRegressor(
            n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", random_state=seed, n_jobs=-1)
    except Exception:
        pass
    regs["HistGBM"] = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed)
    regs["ExtraTrees(현행)"] = lambda: ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=seed, n_jobs=-1)
    regs["RandomForest"] = lambda: RandomForestRegressor(n_estimators=200, max_depth=15, random_state=seed, n_jobs=-1)
    return regs


def loso_corrected(df: pd.DataFrame, feature_cols: list[str], make_reg) -> pd.DataFrame:
    """피험자단위 LOSO 로 corrected_angle 컬럼이 채워진 예측 프레임 반환."""
    out = []
    for subj in sorted(df["subject_id"].unique()):
        tr = df[df["subject_id"] != subj]
        te = df[df["subject_id"] == subj].copy()
        if tr.empty or te.empty:
            continue
        scaler = StandardScaler()
        xtr = scaler.fit_transform(tr[feature_cols])
        reg = make_reg()
        reg.fit(xtr, tr["residual"])
        pred = reg.predict(scaler.transform(te[feature_cols]))
        te["corrected_angle"] = te["mp_knee_angle"].to_numpy(float) + pred
        out.append(te)
    return pd.concat(out, ignore_index=True)


def loso_mean_improvement(p: pd.DataFrame) -> tuple[float, float, float]:
    r = (p["mp_knee_angle"] - p["gt_angle"]).abs()
    c = (p["corrected_angle"] - p["gt_angle"]).abs()
    g = pd.DataFrame({"s": p["subject_id"], "r": r, "c": c}).groupby("s").mean()
    mr, mc = g["r"].mean(), g["c"].mean()
    return mr, mc, (1 - mc / mr) * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--feature-set", default="enhanced_safe", choices=sorted(FEATURE_SETS))
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--deep-threshold", type=float, default=110.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feature_cols = FEATURE_SETS[args.feature_set]
    df = prepare_v6_data(pd.read_csv(args.features), feature_set=args.feature_set)
    df["residual"] = df["gt_angle"] - df["mp_knee_angle"]

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'회귀기':<18}{'전체 raw→cor (개선)':>26}{'deep raw→cor (개선)':>26}")
    print("-" * 70)
    import time
    for name, make_reg in build_regressors(args.seed).items():
        t0 = time.time()
        print(f"... {name} 학습 중", flush=True)
        p = loso_corrected(df, feature_cols, make_reg)
        mr, mc, mi = loso_mean_improvement(p)
        d = p[p["gt_angle"] < args.deep_threshold]
        dr, dc, di = loso_mean_improvement(d)
        rows.append({"regressor": name, "mae_raw": mr, "mae_corrected": mc, "improvement_pct": mi,
                     "deep_mae_raw": dr, "deep_mae_corrected": dc, "deep_improvement_pct": di})
        print(f"{name:<18}{mr:8.2f}→{mc:6.2f} ({mi:5.1f}%){dr:12.2f}→{dc:6.2f} ({di:5.1f}%)  [{time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(args.output / "regressor_comparison.csv", index=False)  # 부분 저장

    pd.DataFrame(rows).to_csv(args.output / "regressor_comparison.csv", index=False)
    print(f"\nSaved: {args.output/'regressor_comparison.csv'}")


if __name__ == "__main__":
    main()
