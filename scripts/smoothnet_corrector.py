from __future__ import annotations

"""
SmoothNet (ECCV 2022) 스타일 학습형 시계열 refinement 보정기 — 표2 비교용.

원논문: "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos"
(arXiv 2112.13715). 핵심 = '시간축 전용 fully-connected residual 네트워크'로
한 윈도우(T 프레임) 신호를 정제.

여기서는 추정기 무릎각 1채널 시퀀스를 입력받아, 윈도우 내 프레임별 '잔차(gt-mp)'를
시간축 refinement로 예측 → corrected = mp + 예측잔차. (입력 z-score 정규화, 잔차 스케일
정규화로 학습 안정화). 시간 전용이라 연속 시퀀스 필요 → 단일 데이터셋(REHAB)용.

출력: paper_baselines_loso.py 와 동일한 persubject_smoothnet.csv (비교표 합류).
실행 venv: torch 있는 환경(.venv-nlf). self-contained(sklearn/scipy 불필요).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def seq_group_cols(df):
    return ["subject_id", "view_type"] + (["camera"] if "camera" in df.columns else [])


def clip_runs(frame_index: np.ndarray):
    n = len(frame_index)
    if n == 0:
        return []
    cuts = [0]
    for i in range(1, n):
        if frame_index[i] <= frame_index[i - 1]:
            cuts.append(i)
    cuts.append(n)
    return [np.arange(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


# ----------------------------- SmoothNet (1채널) -----------------------------

class _ResBlock(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        return self.norm(x + self.net(x))


class SmoothNet(nn.Module):
    """시간축 전용 FC refinement. 입력 [B,T](정규화 각도) → 출력 [B,T](정규화 잔차)."""

    def __init__(self, window, hidden=256, depth=3):
        super().__init__()
        self.encoder = nn.Linear(window, hidden)
        self.blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(depth)])
        self.decoder = nn.Linear(hidden, window)

    def forward(self, x):  # [B, T]
        h = self.encoder(x)
        for blk in self.blocks:
            h = blk(h)
        return self.decoder(h)  # [B, T]


# ----------------------------- 윈도우 -----------------------------

def make_windows(seq: pd.DataFrame, window, stride):
    mp = seq["mp_knee_angle"].to_numpy(np.float32)
    gt = seq["gt_angle"].to_numpy(np.float32)
    L = len(seq)
    A, R, I = [], [], []  # angle, residual(gt-mp), index
    if L < window:
        return A, R, I
    for s in range(0, L - window + 1, stride):
        e = s + window
        A.append(mp[s:e]); R.append(gt[s:e] - mp[s:e]); I.append(np.arange(s, e))
    return A, R, I


def collect_windows(df, window, stride):
    gcols = seq_group_cols(df)
    A, R = [], []
    for _, idx in df.groupby(gcols).groups.items():
        g = df.loc[idx]   # 원본(영상→프레임) 순서 유지; 전역 정렬하면 다영상 그룹이 뒤섞여 런 검출이 깨짐
        for run in clip_runs(g["frame_index"].to_numpy()):
            a, r, _ = make_windows(g.iloc[run], window, stride)
            A += a; R += r
    if not A:
        return None, None
    return np.stack(A), np.stack(R)


# ----------------------------- 학습/예측 -----------------------------

def train_smoothnet(A, R, window, device, epochs=60, lr=1e-3, bs=256):
    a_mean, a_std = float(A.mean()), float(A.std() + 1e-6)
    r_std = float(np.abs(R).mean() + 1e-6)  # 잔차 스케일
    An = (A - a_mean) / a_std
    Rn = R / r_std
    model = SmoothNet(window).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    Xt = torch.from_numpy(An.astype(np.float32)).to(device)
    Yt = torch.from_numpy(Rn.astype(np.float32)).to(device)
    n = len(Xt)
    lossf = nn.SmoothL1Loss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), Yt[b])
            loss.backward(); opt.step()
    return model, (a_mean, a_std, r_std)


def predict_seq(model, stats, df_subj, window, stride, device):
    a_mean, a_std, r_std = stats
    out = df_subj.copy()
    out["corrected_angle"] = out["mp_knee_angle"].astype(float)
    gcols = seq_group_cols(df_subj)
    model.eval()
    for _, idx in out.groupby(gcols).groups.items():
        g = out.loc[idx]   # 원본 순서 유지(다영상 그룹 정렬 시 런 검출 깨짐)
        for run in clip_runs(g["frame_index"].to_numpy()):
            seq = g.iloc[run]
            a, _, iss = make_windows(seq, window, stride)
            if not a:
                continue
            An = (np.stack(a) - a_mean) / a_std
            X = torch.from_numpy(An.astype(np.float32)).to(device)
            with torch.inference_mode():
                pred = model(X).cpu().numpy() * r_std   # 예측 잔차(deg)
            L = len(seq)
            acc = np.zeros(L); cnt = np.zeros(L)
            for k, ii in enumerate(iss):
                acc[ii] += pred[k]; cnt[ii] += 1.0
            mp = seq["mp_knee_angle"].to_numpy(float).copy()
            cov = cnt > 0
            mp[cov] = mp[cov] + acc[cov] / cnt[cov]   # corrected = mp + 잔차
            out.loc[seq.index, "corrected_angle"] = mp
    return out


def run_loso(df, window, stride, device, epochs):
    subjects = sorted(df["subject_id"].unique())
    rows, preds = [], []
    for i, s in enumerate(subjects, 1):
        tr = df[df["subject_id"] != s]
        te = df[df["subject_id"] == s]
        A, R = collect_windows(tr, window, stride)
        if A is None or len(te) == 0:
            continue
        model, stats = train_smoothnet(A, R, window, device, epochs=epochs)
        pe = predict_seq(model, stats, te, window, stride, device)
        raw = mae(pe["gt_angle"], pe["mp_knee_angle"])
        corr = mae(pe["gt_angle"], pe["corrected_angle"])
        rows.append({"subject_id": str(s), "mae_raw": raw, "mae_corrected": corr,
                     "rmse_raw": rmse(pe["gt_angle"], pe["mp_knee_angle"]),
                     "rmse_corrected": rmse(pe["gt_angle"], pe["corrected_angle"]),
                     "improvement_pct": (raw - corr) / raw * 100 if raw else 0.0})
        preds.append(pe)
        print(f"  [smoothnet] fold {i}/{len(subjects)} {s}: {raw:.2f}° -> {corr:.2f}° "
              f"({rows[-1]['improvement_pct']:+.1f}%)", flush=True)
    return pd.DataFrame(rows), (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


def main():
    p = argparse.ArgumentParser(description="SmoothNet-style temporal refinement corrector (table2).")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    from seed_util import set_all_seeds; set_all_seeds(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.features)
    need = {"subject_id", "view_type", "frame_index", "mp_knee_angle", "gt_angle"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"missing cols: {sorted(miss)}")

    subj_df, preds = run_loso(df, args.window, args.stride, device, args.epochs)
    subj_df.to_csv(args.output / "persubject_smoothnet.csv", index=False)
    if len(preds):
        keep=[c for c in ["subject_id","view_type","camera","dataset","frame_index","gt_angle","mp_knee_angle","corrected_angle"] if c in preds.columns]
        preds[keep].to_csv(args.output / "predictions_smoothnet.csv", index=False)
    raw = subj_df["mae_raw"].mean(); corr = subj_df["mae_corrected"].mean()
    print(f"\n[smoothnet] LOSO MAE {raw:.2f}° -> {corr:.2f}° "
          f"({(raw-corr)/raw*100:+.1f}%) | device={device}")
    print(f"Saved: {args.output / 'persubject_smoothnet.csv'}")


if __name__ == "__main__":
    main()
