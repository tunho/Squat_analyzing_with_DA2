from __future__ import annotations
"""
Phase B — LODO / few-shot 를 학습형 보정기 5종(extratrees·mlp·smoothnet·tcn·diffusion)으로 일반화.
평활기(savgol/kalman)는 '도메인 학습' 개념이 없어 LODO/few-shot N/A (제외).

통합 인터페이스 fit_predict(train_df, test_df, corrector) → test_df + corrected_angle.
  - extratrees/mlp : 잔차회귀 (sklearn)
  - diffusion      : CARD식 (diffusion_corrector.DiffusionCorrector)
  - smoothnet/tcn  : 시계열 윈도우 학습 (해당 모듈 재사용)

사용: python gen_allcorr.py --features features_pooled4_X.csv --corrector tcn --eval lodo --output ...
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from paper_common import FEATURE_SETS
from paper_train_loso import prepare_v6_data, summarize_frame_set
from lodo_eval import dataset_of


def fit_predict(train_df, test_df, corrector, feat_cols, device="cuda", epochs=None):
    """train_df 로 학습 → test_df 에 corrected_angle 부여하여 반환."""
    out = test_df.copy()
    if corrector in ("extratrees", "mlp"):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xtr = sc.fit_transform(train_df[feat_cols]); Xte = sc.transform(test_df[feat_cols])
        ytr = train_df["residual"].to_numpy(float)
        if corrector == "extratrees":
            from sklearn.ensemble import ExtraTreesRegressor
            m = ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
        else:
            from sklearn.neural_network import MLPRegressor
            m = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
        m.fit(Xtr, ytr)
        out["corrected_angle"] = test_df["mp_knee_angle"].to_numpy(float) + m.predict(Xte)
    elif corrector == "diffusion":
        from diffusion_corrector import DiffusionCorrector
        Xtr = train_df[feat_cols].to_numpy(float); Xte = test_df[feat_cols].to_numpy(float)
        m = DiffusionCorrector(len(feat_cols), device=device, epochs=epochs or 200).fit(Xtr, train_df["residual"].to_numpy(float))
        out["corrected_angle"] = test_df["mp_knee_angle"].to_numpy(float) + m.predict(Xte)
    elif corrector in ("smoothnet", "tcn"):
        from smoothnet_corrector import collect_windows, predict_seq
        if corrector == "smoothnet":
            from smoothnet_corrector import train_smoothnet as train_fn
        else:
            from tcn_corrector import train_tcn as train_fn
        window, stride = 32, 8
        A, R = collect_windows(train_df, window, stride)
        if A is None:
            out["corrected_angle"] = test_df["mp_knee_angle"].astype(float); return out
        model, stats = train_fn(A, R, window, device, epochs=epochs or 60)
        out = predict_seq(model, stats, test_df, window, stride, device)
    elif corrector == "excelformer":
        from excelformer_corrector import ExcelFormerCorrector
        Xtr = train_df[feat_cols].to_numpy(float); Xte = test_df[feat_cols].to_numpy(float)
        m = ExcelFormerCorrector(len(feat_cols), device=device, epochs=epochs or 40).fit(Xtr, train_df["residual"].to_numpy(float))
        out["corrected_angle"] = test_df["mp_knee_angle"].to_numpy(float) + m.predict(Xte)
    elif corrector == "tabm":
        from tabm_corrector import TabMCorrector
        Xtr = train_df[feat_cols].to_numpy(float); Xte = test_df[feat_cols].to_numpy(float)
        m = TabMCorrector(len(feat_cols), device=device, epochs=epochs or 40).fit(Xtr, train_df["residual"].to_numpy(float))
        out["corrected_angle"] = test_df["mp_knee_angle"].to_numpy(float) + m.predict(Xte)
    else:
        raise ValueError(f"unknown corrector {corrector}")
    return out


def run_lodo(df, corrector, feat_cols, device, epochs):
    rows = []
    for held in ["AIHub", "SUMediPose", "REHAB", "FiT3D"]:
        tr = df[df["dataset"] != held]; te = df[df["dataset"] == held]
        if tr.empty or te.empty:
            continue
        pred = fit_predict(tr, te, corrector, feat_cols, device, epochs)
        ov = summarize_frame_set(pred, held)
        subj = [summarize_frame_set(g, str(s)) for s, g in pred.groupby("subject_id")]
        deep = pred[pred["gt_angle"] < 110.0]
        rows.append({"held_out": held, "n_subjects": te["subject_id"].nunique(),
                     "mae_raw": ov["mae_raw"], "mae_corrected": ov["mae_corrected"],
                     "improvement_pooled_pct": ov["improvement_pct"],
                     "improvement_subjmean_pct": sum(r["improvement_pct"] for r in subj)/len(subj),
                     "deep_improvement_pct": summarize_frame_set(deep, held)["improvement_pct"] if len(deep) else float("nan")})
        print(f"  [LODO {corrector}] {held}: {ov['improvement_pct']:+.1f}%", flush=True)
    return pd.DataFrame(rows)


def run_fewshot(df, corrector, feat_cols, device, epochs, kvals=(0,1,2,3), repeats=5, seed=42):
    rng = np.random.default_rng(seed); rows = []
    for held in ["AIHub", "SUMediPose", "REHAB", "FiT3D"]:
        base = df[df["dataset"] != held]; tgt = df[df["dataset"] == held]
        subs = sorted(tgt["subject_id"].unique())
        for k in kvals:
            draws = 1 if k == 0 else min(repeats, max(1, len(subs)-1))
            accs = []
            for _ in range(draws):
                add = list(rng.choice(subs, size=min(k, max(0,len(subs)-1)), replace=False)) if k>0 else []
                test_subs = [s for s in subs if s not in add]
                if not test_subs:
                    continue
                tr = pd.concat([base, tgt[tgt["subject_id"].isin(add)]]) if add else base
                te = tgt[tgt["subject_id"].isin(test_subs)]
                pred = fit_predict(tr, te, corrector, feat_cols, device, epochs)
                accs.append(summarize_frame_set(pred, held)["improvement_pct"])
            if accs:
                rows.append({"held_out": held, "k": k, "n_draws": len(accs),
                             "pooled_mean": float(np.mean(accs)), "pooled_std": float(np.std(accs))})
                print(f"  [fewshot {corrector}] {held} k={k}: {np.mean(accs):+.1f}%", flush=True)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--corrector", required=True, choices=["extratrees","mlp","mlptorch","smoothnet","tcn","diffusion","excelformer","tabm"])
    p.add_argument("--eval", required=True, choices=["lodo","fewshot"])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--feature-set", default="v6")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    if args.seed is not None:
        from seed_util import set_all_seeds; set_all_seeds(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    feat_cols = FEATURE_SETS[args.feature_set]
    df = prepare_v6_data(pd.read_csv(args.features, low_memory=False), feature_set=args.feature_set)
    df["dataset"] = df["subject_id"].map(dataset_of)
    if args.eval == "lodo":
        run_lodo(df, args.corrector, feat_cols, args.device, args.epochs).to_csv(args.output/f"lodo_{args.corrector}.csv", index=False)
    else:
        run_fewshot(df, args.corrector, feat_cols, args.device, args.epochs).to_csv(args.output/f"fewshot_{args.corrector}.csv", index=False)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
