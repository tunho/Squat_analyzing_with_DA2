from __future__ import annotations
"""
TCN (Bai et al. 2018, "An Empirical Evaluation of Generic Convolutional and Recurrent
Networks for Sequence Modeling", arXiv 1803.01271) 스타일 시계열 보정기 — 표2 비교용.

SmoothNet 과 동일한 윈도우/LOSO 골격(scripts/smoothnet_corrector.py 재사용), 네트워크만
dilated causal TCN 으로 교체. 입력 = 추정기 무릎각 1채널 시퀀스, 출력 = 프레임별 잔차(gt-mp).
시간 전용이라 연속 시퀀스 필요(비연속 데이터엔 SmoothNet 처럼 취약) → robustness 비교의 일부.

출력: persubject_tcn.csv (paper_baselines_loso.py 비교표 합류). 실행 venv: .venv-nlf(torch).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from smoothnet_corrector import (
    collect_windows, predict_seq, make_windows, clip_runs, seq_group_cols, mae, rmse,
)


class Chomp1d(nn.Module):
    """causal conv 의 미래쪽 패딩 제거."""
    def __init__(self, s):
        super().__init__(); self.s = s

    def forward(self, x):
        return x[:, :, :-self.s].contiguous() if self.s > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, ci, co, k, d, dropout=0.1):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, k, padding=pad, dilation=d), Chomp1d(pad), nn.BatchNorm1d(co), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(co, co, k, padding=pad, dilation=d), Chomp1d(pad), nn.BatchNorm1d(co), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(ci, co, 1) if ci != co else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """[B,T](정규화 각도) → [B,T](정규화 잔차). dilated causal conv 스택."""
    def __init__(self, window, channels=(64, 64, 64, 64), k=3, dropout=0.1):
        super().__init__()
        layers = []; ci = 1
        for i, co in enumerate(channels):
            layers.append(TemporalBlock(ci, co, k, 2 ** i, dropout)); ci = co
        self.tcn = nn.Sequential(*layers)
        self.out = nn.Conv1d(ci, 1, 1)

    def forward(self, x):           # [B, T]
        h = x.unsqueeze(1)          # [B, 1, T]
        h = self.tcn(h)
        return self.out(h).squeeze(1)  # [B, T]


def train_tcn(A, R, window, device, epochs=60, lr=5e-4, bs=256):
    a_mean, a_std = float(A.mean()), float(A.std() + 1e-6)
    r_std = float(np.abs(R).mean() + 1e-6)
    An = (A - a_mean) / a_std
    Rn = R / r_std
    model = TCN(window).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    Xt = torch.from_numpy(An.astype(np.float32)).to(device)
    Yt = torch.from_numpy(Rn.astype(np.float32)).to(device)
    n = len(Xt); lossf = nn.SmoothL1Loss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), Yt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient explosion→NaN 방지
            opt.step()
    return model, (a_mean, a_std, r_std)


def run_loso(df, window, stride, device, epochs):
    subjects = sorted(df["subject_id"].unique())
    rows = []; preds = []
    for i, s in enumerate(subjects, 1):
        tr = df[df["subject_id"] != s]
        te = df[df["subject_id"] == s]
        A, R = collect_windows(tr, window, stride)
        if A is None or len(te) == 0:
            continue
        model, stats = train_tcn(A, R, window, device, epochs=epochs)
        pe = predict_seq(model, stats, te, window, stride, device)
        preds.append(pe)
        raw = mae(pe["gt_angle"], pe["mp_knee_angle"]); corr = mae(pe["gt_angle"], pe["corrected_angle"])
        rows.append({"subject_id": str(s), "mae_raw": raw, "mae_corrected": corr,
                     "rmse_raw": rmse(pe["gt_angle"], pe["mp_knee_angle"]),
                     "rmse_corrected": rmse(pe["gt_angle"], pe["corrected_angle"]),
                     "improvement_pct": (raw - corr) / raw * 100 if raw else 0.0})
        print(f"  [tcn] fold {i}/{len(subjects)} {s}: {raw:.2f}° -> {corr:.2f}° "
              f"({rows[-1]['improvement_pct']:+.1f}%)", flush=True)
    return pd.DataFrame(rows), (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


def main():
    p = argparse.ArgumentParser(description="TCN temporal refinement corrector (table2).")
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
    subj, preds = run_loso(df, args.window, args.stride, device, args.epochs)
    subj.to_csv(args.output / "persubject_tcn.csv", index=False)
    if subj.empty:
        print("[tcn] 빈 결과 — 윈도우 생성 실패(시퀀스 부족?). 스킵."); return
    if len(preds):
        keep=[c for c in ["subject_id","view_type","camera","dataset","frame_index","gt_angle","mp_knee_angle","corrected_angle"] if c in preds.columns]
        preds[keep].to_csv(args.output / "predictions_tcn.csv", index=False)
    raw = subj["mae_raw"].mean(); corr = subj["mae_corrected"].mean()
    print(f"\n[tcn] LOSO MAE {raw:.2f}° -> {corr:.2f}° ({(raw-corr)/raw*100:+.1f}%) | device={device}")
    print(f"Saved: {args.output / 'persubject_tcn.csv'}")


if __name__ == "__main__":
    main()
