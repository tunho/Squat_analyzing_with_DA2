"""gD_corrector_bar.png + summary_all.png 재생성 (seeded 데이터에서 직접 계산).
 - within: 신경망 5종=multiseed_out/seed42, 나머지=baselines_ (subject-mean, 16칸 평균)
 - pooled: pooled5_out (subject-mean, 4추정기 평균)
 - LODO:   lodo_base (improvement_pooled_pct, 추정기×held-out 평균)
"""
from __future__ import annotations
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp); plt.rcParams["font.family"]=fm.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"]=False
R="experiments/paper"; PRES=f"{R}/presentation"
DOMS=["ex6","fit3d","sume","aihub"]; ESTS=["offstd","nlf","rtmw3d","motionbert"]
CORRS=["ExtraTrees","ExcelFormer","SmoothNet","TCN","Diffusion","TabM","MLP","SavGol","Kalman"]
KEY={"ExtraTrees":"extratrees","ExcelFormer":"excelformer","SmoothNet":"smoothnet","TCN":"tcn",
     "Diffusion":"diffusion","TabM":"tabm","MLP":"mlp","SavGol":"savgol","Kalman":"kalman"}
NEURAL={"smoothnet","tcn","diffusion","excelformer","tabm"}

def smean(f):
    if not os.path.exists(f): return np.nan
    x=pd.read_csv(f); return x["improvement_pct"].mean() if "improvement_pct" in x and len(x) else np.nan

def within(corr):
    k=KEY[corr]; vals=[]
    for d in DOMS:
        for e in ESTS:
            de=d if e=="offstd" else f"{d}_{e}"
            if k in NEURAL: f=f"{R}/multiseed_out/{k}/{d}_{e}/seed42/persubject_{k}.csv"
            elif k=="extratrees": f=f"{R}/baselines_{de}/persubject_extratrees_residual.csv"
            elif k=="mlp": f=f"{R}/baselines_{de}/persubject_mlp_residual.csv"
            else: f=f"{R}/baselines_{de}/persubject_{k}.csv"
            v=smean(f)
            if not np.isnan(v): vals.append(v)
    return np.mean(vals) if vals else np.nan

def pooled(corr):
    k=KEY[corr]; vals=[]
    for e in ESTS:
        v=smean(f"{R}/pooled5_out/{k}_{e}/persubject_{k}.csv")
        if not np.isnan(v): vals.append(v)
    return np.mean(vals) if vals else np.nan

def lodo(corr):
    k=KEY[corr]; vals=[]
    if k=="extratrees":  # 별도 위치
        for f in glob.glob(f"{R}/lodo_*/lodo_metrics.csv"):
            d=pd.read_csv(f)
            if "improvement_pooled_pct" in d: vals+=list(d["improvement_pooled_pct"])
    else:
        for f in glob.glob(f"{R}/lodo_base/outB_*/lodo_{k}.csv"):
            d=pd.read_csv(f)
            if "improvement_pooled_pct" in d: vals+=list(d["improvement_pooled_pct"])
    return np.mean(vals) if vals else np.nan

W={c:within(c) for c in CORRS}; Pl={c:pooled(c) for c in CORRS}; L={c:lodo(c) for c in CORRS}

def color(v): return "#1b7837" if v>=20 else "#7fbf7b" if v>=0 else "#d6604d"
# ---- gD_corrector_bar: within 평균 순위 ----
order=sorted(CORRS,key=lambda c:-W[c])
fig,ax=plt.subplots(figsize=(10,5))
vals=[W[c] for c in order]
b=ax.bar(range(len(order)),vals,color=[color(v) for v in vals])
for i,v in enumerate(vals): ax.text(i,v+(0.6 if v>=0 else -2.2),f"{v:+.1f}",ha="center",fontsize=10,fontweight="bold")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=20,ha="right",fontsize=11)
ax.axhline(0,color="gray",lw=0.8); ax.set_ylabel("도메인 내 평균 개선율 %")
ax.set_title("보정기 9종 — 도메인 내 평균 개선율 (seed42 반영)")
plt.tight_layout(); plt.savefig(f"{PRES}/gD_corrector_bar.png",dpi=150); plt.close()
print("saved gD_corrector_bar.png")

# ---- summary_all: 9종 × 3실험 그룹막대 ----
fig,ax=plt.subplots(figsize=(12,5.5))
x=np.arange(len(CORRS)); w=0.26
ax.bar(x-w,[W[c] for c in CORRS],w,label="① 도메인 내",color="#1b7837")
ax.bar(x,  [Pl[c] for c in CORRS],w,label="② pooled",color="#2166ac")
ax.bar(x+w,[L[c] if not np.isnan(L[c]) else 0 for c in CORRS],w,label="③ 도메인 간(LODO)",color="#d97706")
ax.axhline(0,color="gray",lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(CORRS,rotation=20,ha="right",fontsize=10)
ax.set_ylabel("개선율 %"); ax.legend()
ax.set_title("종합 — 보정기 9종 × 실험 3방식 (①② 개선 / ③ 실패, seed42)")
# LODO N/A 표시 (savgol/kalman/extratrees는 LODO 없음)
for i,c in enumerate(CORRS):
    if np.isnan(L[c]): ax.text(i+w,1,"N/A",ha="center",fontsize=7,color="#d97706",rotation=90)
plt.tight_layout(); plt.savefig(f"{PRES}/summary_all.png",dpi=150); plt.close()
print("saved summary_all.png")

print("\n=== 값 확인 ===")
print(f"{'보정기':12s} {'within':>8s} {'pooled':>8s} {'LODO':>8s}")
for c in CORRS:
    print(f"{c:12s} {W[c]:8.1f} {Pl[c]:8.1f} {('%.1f'%L[c]) if not np.isnan(L[c]) else 'N/A':>8s}")
