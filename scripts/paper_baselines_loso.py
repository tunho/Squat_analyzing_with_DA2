from __future__ import annotations

"""
교정(보정) 방법 비교 베이스라인 — J-BHI 심사 대응(표2).

목적: 우리 ExtraTrees 잔차보정이 '특정 보정 알고리즘 트릭'이 아님을 보이기 위해,
동일한 feature(prepare_v6_data) + 동일한 LOSO subject split 위에서 보정 '방법'만 바꿔
공정 비교한다. 추정기(앞단)는 고정(MediaPipe), 보정기(뒷단)만 교체한다.

방법군:
  - extratrees_residual : 우리 방법 (gt - mp_knee_angle 잔차회귀, 기준선)
  - mlp_residual        : 딥러닝(다층 신경망) 잔차회귀 — '다른 딥러닝' 보정
  - mlp_direct          : 딥러닝으로 gt_angle 직접회귀 (잔차 vs 직접 ablation)
  - savgol              : Savitzky-Golay 평활 — 일반 각도 후처리(비학습)
  - kalman              : 상수속도 Kalman 평활 — 일반 각도 후처리(비학습)
  - raw                 : 보정 없음(= mp_knee_angle) 참조용

학습형(extratrees/mlp)은 LOSO. 비학습 평활형(savgol/kalman)은 시퀀스별 후처리라
fold 개념이 없지만, 동일 프레임 집합에 corrected_angle을 만들어 같은 지표로 집계한다.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from paper_common import FEATURE_SETS, improvement_pct, mae, rmse, write_json
from paper_train_loso import prepare_v6_data


# 시퀀스 그룹키 (prepare_v6_data 와 동일 규칙: camera 가 있으면 포함)
def seq_group_cols(df: pd.DataFrame) -> list[str]:
    return ["subject_id", "view_type"] + (["camera"] if "camera" in df.columns else [])


# ----------------------------- 학습형(LOSO) -----------------------------

def _make_regressor(kind: str, random_state: int):
    if kind == "extratrees":
        return ExtraTreesRegressor(n_estimators=500, max_depth=15,
                                   random_state=random_state, n_jobs=-1)
    if kind == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=random_state,
        )
    raise ValueError(f"unknown regressor kind: {kind}")


def run_learning_loso(df: pd.DataFrame, feature_cols: list[str], kind: str,
                      target: str, random_state: int) -> pd.DataFrame:
    """target='residual' 이면 잔차회귀(corrected = mp + pred),
    target='gt_angle' 이면 직접회귀(corrected = pred)."""
    subjects = sorted(df["subject_id"].unique())
    preds = []
    for i, subject in enumerate(subjects, 1):
        train_df = df[df["subject_id"] != subject]
        test_df = df[df["subject_id"] == subject].copy()
        if train_df.empty or test_df.empty:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_df[feature_cols])
        model = _make_regressor(kind, random_state)
        model.fit(x_train, train_df[target].to_numpy(dtype=float))
        pred = model.predict(scaler.transform(test_df[feature_cols]))
        if target == "residual":
            test_df["corrected_angle"] = test_df["mp_knee_angle"].to_numpy(dtype=float) + pred
        else:
            test_df["corrected_angle"] = pred
        preds.append(test_df)
        print(f"  [{kind}/{target}] fold {i}/{len(subjects)} {subject}", flush=True)
    return pd.concat(preds, ignore_index=True)


# ----------------------------- 비학습 평활형 -----------------------------

def _kalman_smooth_1d(z: np.ndarray, q: float, r: float) -> np.ndarray:
    """상수속도(위치,속도) 2-state Kalman forward + RTS smoother. z=각도 측정열(deg)."""
    n = len(z)
    if n == 0:
        return z
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    R = np.array([[r]])
    x = np.array([z[0], 0.0])
    P = np.eye(2) * 10.0
    xf = np.zeros((n, 2)); Pf = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2))
    for k in range(n):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xp[k] = x; Pp[k] = P
        y = z[k] - (H @ x)[0]
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T / S).reshape(2)
        x = x + K * y
        P = (np.eye(2) - np.outer(K, H)) @ P
        xf[k] = x; Pf[k] = P
    # RTS smoother
    xs = xf.copy()
    Ps = Pf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs[:, 0]


def _clip_runs(frame_index: np.ndarray) -> list[np.ndarray]:
    """frame_index 가 안 늘어나는(<=이전) 지점을 클립 경계로 보고 연속 구간 인덱스로 분할.
    pooled 데이터에서 한 (subject,view,camera) 그룹에 여러 클립이 섞여도
    평활이 클립 경계를 넘어 뭉개지 않도록 한다."""
    n = len(frame_index)
    if n == 0:
        return []
    cuts = [0]
    for i in range(1, n):
        if frame_index[i] <= frame_index[i - 1]:
            cuts.append(i)
    cuts.append(n)
    return [np.arange(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]


def run_smoothing(df: pd.DataFrame, method: str, window: int, polyorder: int,
                  q: float, r: float) -> pd.DataFrame:
    out = df.copy()
    out["corrected_angle"] = out["mp_knee_angle"].astype(float)
    gcols = seq_group_cols(df)
    for _, idx in out.groupby(gcols).groups.items():
        grp = out.loc[idx].sort_values("frame_index")
        for run in _clip_runs(grp["frame_index"].to_numpy()):
            seq = grp.iloc[run]
            z = seq["mp_knee_angle"].to_numpy(dtype=float)
            if method == "savgol":
                w = min(window, len(z) if len(z) % 2 == 1 else len(z) - 1)
                if w >= 3 and w > polyorder:
                    sm = savgol_filter(z, w, polyorder)
                else:
                    sm = z
            elif method == "kalman":
                sm = _kalman_smooth_1d(z, q, r)
            else:
                raise ValueError(method)
            out.loc[seq.index, "corrected_angle"] = sm
    return out


# ----------------------------- 집계 -----------------------------

def summarize(pred_df: pd.DataFrame, method: str, deep_threshold: float) -> dict:
    def block(d: pd.DataFrame) -> dict:
        raw = mae(d["gt_angle"], d["mp_knee_angle"])
        corr = mae(d["gt_angle"], d["corrected_angle"])
        return {
            "mae_raw": raw,
            "mae_corrected": corr,
            "rmse_raw": rmse(d["gt_angle"], d["mp_knee_angle"]),
            "rmse_corrected": rmse(d["gt_angle"], d["corrected_angle"]),
            "improvement_pct": improvement_pct(raw, corr),
        }
    # subject 평균(LOSO 관행) 과 pooled 둘 다 보고
    per_subj = []
    for s, g in pred_df.groupby("subject_id"):
        b = block(g); b["subject_id"] = str(s); per_subj.append(b)
    subj_df = pd.DataFrame(per_subj)
    res = {
        "method": method,
        "frames": int(len(pred_df)),
        "subjects": int(pred_df["subject_id"].nunique()),
        "loso_mae_raw": float(subj_df["mae_raw"].mean()),
        "loso_mae_corrected": float(subj_df["mae_corrected"].mean()),
        "loso_improvement_pct": improvement_pct(float(subj_df["mae_raw"].mean()),
                                                float(subj_df["mae_corrected"].mean())),
        "pooled": block(pred_df),
    }
    deep = pred_df[pred_df["gt_angle"] < deep_threshold]
    if not deep.empty:
        res["deep_flexion_pooled"] = block(deep)
    return res, subj_df


def main() -> None:
    p = argparse.ArgumentParser(description="Correction-method baselines on shared LOSO (J-BHI table2).")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="v6")
    p.add_argument("--methods", nargs="+",
                   default=["raw", "savgol", "kalman", "mlp_direct", "mlp_residual", "extratrees_residual"])
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--deep-flexion-threshold", type=float, default=110.0)
    p.add_argument("--savgol-window", type=int, default=11)
    p.add_argument("--savgol-poly", type=int, default=2)
    p.add_argument("--kalman-q", type=float, default=1.0)
    p.add_argument("--kalman-r", type=float, default=25.0)
    p.add_argument("--time-aware", action="store_true")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--save-predictions", action="store_true", help="predictions_{method}.csv 저장(clinical용)")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(args.features)
    feature_cols = FEATURE_SETS[args.feature_set]
    df = prepare_v6_data(raw_df, feature_set=args.feature_set,
                         time_aware=args.time_aware, fps=args.fps)

    summaries = []
    for method in args.methods:
        print(f"== method: {method} ==", flush=True)
        if method == "raw":
            pred = df.copy(); pred["corrected_angle"] = pred["mp_knee_angle"].astype(float)
        elif method == "savgol":
            pred = run_smoothing(df, "savgol", args.savgol_window, args.savgol_poly, args.kalman_q, args.kalman_r)
        elif method == "kalman":
            pred = run_smoothing(df, "kalman", args.savgol_window, args.savgol_poly, args.kalman_q, args.kalman_r)
        elif method == "mlp_residual":
            pred = run_learning_loso(df, feature_cols, "mlp", "residual", args.random_state)
        elif method == "mlp_direct":
            pred = run_learning_loso(df, feature_cols, "mlp", "gt_angle", args.random_state)
        elif method == "extratrees_residual":
            pred = run_learning_loso(df, feature_cols, "extratrees", "residual", args.random_state)
        elif method == "extratrees_direct":
            pred = run_learning_loso(df, feature_cols, "extratrees", "gt_angle", args.random_state)
        else:
            raise ValueError(f"unknown method: {method}")

        summary, subj_df = summarize(pred, method, args.deep_flexion_threshold)
        summaries.append(summary)
        subj_df.to_csv(args.output / f"persubject_{method}.csv", index=False)
        if getattr(args, "save_predictions", False):
            keep = [c for c in ["subject_id","view_type","camera","dataset","frame_index",
                                "gt_angle","mp_knee_angle","corrected_angle"] if c in pred.columns]
            pred[keep].to_csv(args.output / f"predictions_{method}.csv", index=False)
        print(f"   LOSO MAE {summary['loso_mae_raw']:.2f}° -> "
              f"{summary['loso_mae_corrected']:.2f}° ({summary['loso_improvement_pct']:+.1f}%) | "
              f"pooled {summary['pooled']['mae_corrected']:.2f}°", flush=True)

    # 비교표(표2)
    table = pd.DataFrame([{
        "method": s["method"],
        "loso_mae_corrected": round(s["loso_mae_corrected"], 3),
        "loso_improvement_pct": round(s["loso_improvement_pct"], 2),
        "pooled_mae_corrected": round(s["pooled"]["mae_corrected"], 3),
        "pooled_rmse_corrected": round(s["pooled"]["rmse_corrected"], 3),
        "deep_mae_corrected": round(s.get("deep_flexion_pooled", {}).get("mae_corrected", float("nan")), 3),
    } for s in summaries])
    table.to_csv(args.output / "method_comparison.csv", index=False)
    write_json(args.output / "method_comparison.json",
               {"feature_set": args.feature_set, "features": feature_cols, "results": summaries})
    print("\n=== method comparison (표2) ===")
    print(table.to_string(index=False))
    print(f"\nSaved: {args.output / 'method_comparison.csv'}")


if __name__ == "__main__":
    main()
