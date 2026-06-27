from __future__ import annotations
"""
조건부 디퓨전 잔차 보정기 (CARD, NeurIPS 2022 식) — 2023+ SOTA 보정 패러다임 baseline.
ExtraTrees 와 동일한 v6 피처를 조건으로, 잔차(gt - mp_knee_angle)를 denoising diffusion 으로 예측.
프레임 단위(시계열 윈도우 불필요) → SmoothNet 과 달리 비연속 데이터에도 적용 가능.
LOSO subject split + 출력 persubject_diffusion.csv (paper_baselines_loso.py 비교표 합류).

사용: ../.venv-nlf/bin/python diffusion_corrector.py --features ... --output ... --device cuda
"""
import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(__file__))
from paper_common import FEATURE_SETS
from paper_train_loso import prepare_v6_data


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


# ---------- diffusion schedule ----------
def cosine_betas(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class SinusoidalTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: [B] long
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        a = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class MeanEstimator(nn.Module):
    """f_φ(x) — CARD 의 사전학습 조건부 평균추정기(디퓨전 엔드포인트 앵커)."""
    def __init__(self, n_feat, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class Denoiser(nn.Module):
    """ε_θ(y_t, t, x, f) — 잔차 1채널 노이즈 예측, 피처 x + 앵커 f(x) 조건."""
    def __init__(self, n_feat, tdim=64, hidden=256):
        super().__init__()
        self.temb = nn.Sequential(SinusoidalTime(tdim), nn.Linear(tdim, tdim), nn.SiLU())
        self.cond = nn.Sequential(nn.Linear(n_feat, hidden), nn.SiLU())
        self.inp = nn.Linear(1 + 1 + tdim + hidden, hidden)   # y_t, f(x), t, x
        self.body = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, y, t, x, f):
        h = torch.cat([y, f, self.temb(t), self.cond(x)], dim=-1)
        h = self.inp(h)
        h = self.body(h)
        return self.out(h)


class DiffusionCorrector:
    """CARD (Han et al., NeurIPS 2022): f_φ(x) 앵커 조건부 디퓨전 회귀.
    forward: y_t = √ᾱ_t·y_0 + (1-√ᾱ_t)·f(x) + √(1-ᾱ_t)·ε
    reverse: y_T ~ N(f(x), I) 에서 시작 → 최악도 회귀평균으로 폴백(폭주 방지)."""
    def __init__(self, n_feat, T=100, device="cuda", epochs=200, lr=1e-3, n_samples=20, mean_epochs=120):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.T, self.epochs, self.lr, self.n_samples, self.mean_epochs = T, epochs, lr, n_samples, mean_epochs
        self.mean = MeanEstimator(n_feat).to(self.device)
        self.model = Denoiser(n_feat).to(self.device)
        betas = cosine_betas(T).to(self.device)
        self.betas = betas
        self.alphas = 1 - betas
        self.acp = torch.cumprod(self.alphas, 0)            # ᾱ_t
        self.acp_prev = torch.cat([torch.ones(1, device=self.device), self.acp[:-1]])
        self.sqrt_acp = torch.sqrt(self.acp)
        self.sqrt_acp_prev = torch.sqrt(self.acp_prev)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_1macp = torch.sqrt(1 - self.acp)
        self.one_m_acp = 1 - self.acp

    def fit(self, X, y):
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-6
        self.ym, self.ys = y.mean(), y.std() + 1e-6
        Xn = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32, device=self.device)
        yn = torch.tensor(((y - self.ym) / self.ys).reshape(-1, 1), dtype=torch.float32, device=self.device)
        # clip_denoised: 표준화공간 ±3.5σ로 강건 클램프(데이터 min/max는 이상치에 취약) — 발산 방지.
        self.y0_lo, self.y0_hi = -3.5, 3.5
        n = Xn.shape[0]
        bs = min(4096, n)
        # 1) 평균추정기 f_φ(x) 사전학습 (MSE)
        opt_m = torch.optim.Adam(self.mean.parameters(), lr=self.lr)
        self.mean.train()
        for ep in range(self.mean_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                loss = F.mse_loss(self.mean(Xn[idx]), yn[idx])
                opt_m.zero_grad(); loss.backward(); opt_m.step()
        self.mean.eval()
        with torch.no_grad():
            fall = self.mean(Xn)                              # f(x) for all train rows
        # 2) CARD 디퓨전 학습 (앵커 forward)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb, yb, fb = Xn[idx], yn[idx], fall[idx]
                t = torch.randint(0, self.T, (xb.shape[0],), device=self.device)
                noise = torch.randn_like(yb)
                sa = self.sqrt_acp[t][:, None]; s1 = self.sqrt_1macp[t][:, None]
                yt = sa * yb + (1 - sa) * fb + s1 * noise     # CARD forward
                pred = self.model(yt, t, xb, fb)
                loss = F.mse_loss(pred, noise)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        Xn = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32, device=self.device)
        m = Xn.shape[0]
        S = self.n_samples
        xrep = Xn.repeat_interleave(S, dim=0)
        f = torch.clamp(self.mean(xrep), -3.5, 3.5)          # 앵커 f(x), OOD 폭주 방지 클램프
        y = f + torch.randn_like(f)                          # y_T ~ N(f(x), I)
        for ti in reversed(range(self.T)):
            t = torch.full((xrep.shape[0],), ti, device=self.device, dtype=torch.long)
            eps = self.model(y, t, xrep, f)
            sa, s1 = self.sqrt_acp[ti], self.sqrt_1macp[ti]
            # y_0 복원 (CARD): y0 = (y_t - (1-√ᾱ_t)f - √(1-ᾱ_t)ε) / √ᾱ_t
            y0 = (y - (1 - sa) * f - s1 * eps) / sa
            y0 = torch.clamp(y0, self.y0_lo, self.y0_hi)     # clip_denoised: 발산 방지
            if ti > 0:
                b = self.betas[ti]; acp = self.acp[ti]; acp_p = self.acp_prev[ti]
                sa_p = self.sqrt_acp_prev[ti]; salpha = self.sqrt_alphas[ti]
                g0 = b * sa_p / (1 - acp)
                g1 = (1 - acp_p) * salpha / (1 - acp)
                g2 = 1 + (sa - 1) * (salpha + sa_p) / (1 - acp)
                mean = g0 * y0 + g1 * y + g2 * f              # CARD posterior mean
                var = (1 - acp_p) / (1 - acp) * b
                y = mean + torch.sqrt(var) * torch.randn_like(y)
            else:
                y = y0
        y = y.reshape(m, S).mean(1).cpu().numpy()
        return y * self.ys + self.ym


def run_loso(df, feats, device, epochs, T, n_samples):
    subjects = sorted(df["subject_id"].unique())
    rows = []
    for i, s in enumerate(subjects, 1):
        tr = df[df["subject_id"] != s]
        te = df[df["subject_id"] == s].copy()
        Xtr, ytr = tr[feats].to_numpy(float), tr["residual"].to_numpy(float)
        Xte = te[feats].to_numpy(float)
        model = DiffusionCorrector(len(feats), T=T, device=device, epochs=epochs, n_samples=n_samples).fit(Xtr, ytr)
        pred_res = model.predict(Xte)
        te["corrected_angle"] = te["mp_knee_angle"].to_numpy(float) + pred_res
        raw = mae(te["gt_angle"], te["mp_knee_angle"])
        corr = mae(te["gt_angle"], te["corrected_angle"])
        rows.append({"subject_id": str(s), "mae_raw": raw, "mae_corrected": corr,
                     "rmse_raw": rmse(te["gt_angle"], te["mp_knee_angle"]),
                     "rmse_corrected": rmse(te["gt_angle"], te["corrected_angle"]),
                     "improvement_pct": (raw - corr) / raw * 100 if raw else 0.0})
        print(f"  [diffusion] fold {i}/{len(subjects)} {s}: {raw:.2f}° -> {corr:.2f}° "
              f"({rows[-1]['improvement_pct']:+.1f}%)", flush=True)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--feature-set", default="v6", choices=sorted(FEATURE_SETS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=20)
    args = p.parse_args()

    df = prepare_v6_data(pd.read_csv(args.features, low_memory=False), feature_set=args.feature_set)
    feats = FEATURE_SETS[args.feature_set]
    args.output.mkdir(parents=True, exist_ok=True)
    subj = run_loso(df, feats, args.device, args.epochs, args.timesteps, args.n_samples)
    subj.to_csv(args.output / "persubject_diffusion.csv", index=False)
    print(f"\n[diffusion] subject-mean improvement: {subj['improvement_pct'].mean():+.2f}% "
          f"| raw {subj['mae_raw'].mean():.2f}° -> corr {subj['mae_corrected'].mean():.2f}°")
    print(f"Saved: {args.output / 'persubject_diffusion.csv'}")


if __name__ == "__main__":
    main()
