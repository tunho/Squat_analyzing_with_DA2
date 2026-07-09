"""도메인 내 실패 피험자의 프레임 단위 실패분석 + 그림 (v2).

시계열은 클립/뷰가 겹쳐 지저분 → 두 가지 명확한 뷰로 대체:
 (1) 케이스별 요약 막대: raw MAE vs corrected MAE (전체/심굴곡)
 (2) 케이스별 각도구간 MAE: 어느 무릎각에서 보정이 망치는지
 (3) 대표 산점도: 프레임별 raw AE vs corrected AE (대각선 위=악화)
"""
from __future__ import annotations
import os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt

for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

import pathlib
ROOT = str(pathlib.Path(os.environ.get("EXE_EST_ROOT", pathlib.Path(__file__).resolve().parent.parent)) / "experiments" / "paper")
PRES = f"{ROOT}/presentation"
CASES = [
    ("fit3d_motionbert", "fit3d_s04", "FiT3D s04\n(MotionBERT)"),
    ("sume_nlf",         "S22",       "SUMe S22\n(NLF)"),
    ("sume_nlf",         "S1",        "SUMe S1\n(NLF)"),
    ("fit3d_rtmw3d",     "fit3d_s05", "FiT3D s05\n(RTMW3D)"),
    ("ex6_nlf",          "PM_105",    "REHAB PM_105\n(NLF)"),
]

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

def load(cell, subj):
    df = pd.read_csv(f"{ROOT}/failcase_{cell}/loso_predictions.csv")
    sd = df[df["subject_id"].astype(str) == subj].copy()
    sd["ae_raw"] = (sd["gt_angle"] - sd["mp_knee_angle"]).abs()
    sd["ae_cor"] = (sd["gt_angle"] - sd["corrected_angle"]).abs()
    return sd

data = {label: load(cell, subj) for cell, subj, label in CASES}

# ---- (1) 요약 막대: 케이스별 raw vs corrected MAE ----
fig, ax = plt.subplots(figsize=(9, 4.6))
labels = list(data)
raw = [mae(d.gt_angle, d.mp_knee_angle) for d in data.values()]
cor = [mae(d.gt_angle, d.corrected_angle) for d in data.values()]
x = np.arange(len(labels)); w = 0.38
b1 = ax.bar(x - w/2, raw, w, label="Raw (보정 전)", color="#1f77b4")
b2 = ax.bar(x + w/2, cor, w, label="Corrected (보정 후)", color="#d62728")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15, f"{b.get_height():.1f}",
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("MAE (°)")
ax.set_title("도메인 내 실패 피험자: 보정이 오히려 오차 키움 (원본이 이미 정확한 케이스)")
ax.legend()
plt.tight_layout(); plt.savefig(f"{PRES}/failcase_summary_bars.png", dpi=190); plt.close()
print("그림: failcase_summary_bars.png")

# ---- (2) 각도구간 MAE (대표 2케이스: S22, PM_105) : 어느 각도에서 실패? ----
bins = [(160,180,"거의 선자세\n160-180°"),(140,160,"얕은굴곡\n140-160°"),
        (110,140,"중간\n110-140°"),(0,110,"심굴곡\n<110°")]
for cell, subj, label in [("sume_nlf","S22","SUMe S22 (NLF)"),("ex6_nlf","PM_105","REHAB PM_105 (NLF)")]:
    d = data[[l for c,s,l in CASES if s==subj and c==cell][0]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rr=[]; cc=[]; xl=[]
    for lo,hi,name in bins:
        g = d[(d.gt_angle>=lo)&(d.gt_angle<hi)]
        if len(g)<10: continue
        rr.append(mae(g.gt_angle,g.mp_knee_angle)); cc.append(mae(g.gt_angle,g.corrected_angle)); xl.append(f"{name}\n(n={len(g)})")
    x=np.arange(len(xl)); w=0.38
    ax.bar(x-w/2, rr, w, label="Raw", color="#1f77b4")
    ax.bar(x+w/2, cc, w, label="Corrected", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=8); ax.set_ylabel("MAE (°)")
    ax.set_title(f"{label}: 무릎 각도 구간별 — 원본 정확한 구간일수록 보정이 악화")
    ax.legend()
    plt.tight_layout(); plt.savefig(f"{PRES}/failcase_byangle_{subj}.png", dpi=190); plt.close()
    print(f"그림: failcase_byangle_{subj}.png")

# ---- (3) 산점도 (대표 S22): 프레임별 raw AE vs cor AE, 대각선 위=악화 ----
d = data["SUMe S22\n(NLF)"]
fig, ax = plt.subplots(figsize=(5.6, 5.4))
sc=ax.scatter(d.ae_raw, d.ae_cor, s=6, alpha=0.25, c=d.gt_angle, cmap="viridis")
m=max(d.ae_raw.max(), d.ae_cor.max())
ax.plot([0,m],[0,m], "k--", lw=1)
ax.fill_between([0,m],[0,m],[m,m], color="red", alpha=0.06)
ax.text(m*0.25, m*0.8, "이 위=보정이\n악화시킴", color="#a11", fontsize=9)
ax.set_xlabel("Raw 오차 |gt-raw| (°)"); ax.set_ylabel("Corrected 오차 |gt-cor| (°)")
plt.colorbar(sc, label="GT 무릎각 (°)")
ax.set_title("SUMe S22: 프레임별 오차 (대부분 대각선 위=악화)")
plt.tight_layout(); plt.savefig(f"{PRES}/failcase_scatter_S22.png", dpi=190); plt.close()
print("그림: failcase_scatter_S22.png")

# ---- 요약 표 저장 ----
rows=[]
for label, d in data.items():
    deep=d[d.gt_angle<110]; shal=d[d.gt_angle>=110]
    rows.append(dict(case=label.replace("\n"," "), n=len(d),
        raw=mae(d.gt_angle,d.mp_knee_angle), cor=mae(d.gt_angle,d.corrected_angle),
        raw_deep=mae(deep.gt_angle,deep.mp_knee_angle) if len(deep) else np.nan,
        cor_deep=mae(deep.gt_angle,deep.corrected_angle) if len(deep) else np.nan,
        raw_shal=mae(shal.gt_angle,shal.mp_knee_angle) if len(shal) else np.nan,
        cor_shal=mae(shal.gt_angle,shal.corrected_angle) if len(shal) else np.nan))
R=pd.DataFrame(rows); R.to_csv(f"{ROOT}/failcase_frame_summary.csv", index=False)
pd.set_option("display.width",160)
print("\n=== 프레임 요약 (심굴곡<110 vs 그외) ===")
print(R.to_string(index=False))
