"""실험 3방식별 히트맵 — 슬라이드 분할(①②③)에 맞춘 큐브 요약.
 ① within : 보정기(9) × 추정기(4), 도메인 4개 평균
 ② pooled : 보정기(9) × 추정기(4)
 ③ LODO   : 보정기(6) × held-out 도메인(4), 추정기 평균
색 = 개선율%(초록 개선 / 빨강 악화).
"""
from __future__ import annotations
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp); plt.rcParams["font.family"]=fm.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"]=False
R="experiments/paper"; PRES=f"{R}/presentation"

# 행 순서(모든 히트맵 공통): 대체로 좋은→나쁜
CORR_ORDER=["TabM","ExtraTrees","ExcelFormer","Diffusion","MLP","TCN","SmoothNet","SavGol","Kalman"]
ESTS=[("offstd","MediaPipe"),("nlf","NLF"),("rtmw3d","RTMW3D"),("motionbert","MotionBERT")]
DOMS=["ex6","fit3d","sume","aihub"]

def imp_mean(path):
    if not os.path.exists(path): return np.nan
    d=pd.read_csv(path)
    return float(d["improvement_pct"].mean()) if "improvement_pct" in d else np.nan

# ---------- ① within: 도메인 평균 ----------
def within_file(corr,dom,est):
    de = dom if est=="offstd" else f"{dom}_{est}"
    # 신경망 5종: seeded 재실행(multiseed_out/seed42) 사용 → 재현 가능
    NEURAL={"SmoothNet":"smoothnet","TCN":"tcn","Diffusion":"diffusion",
            "ExcelFormer":"excelformer","TabM":"tabm"}
    if corr in NEURAL:
        k=NEURAL[corr]
        return f"{R}/multiseed_out/{k}/{dom}_{est}/seed42/persubject_{k}.csv"
    # 결정적(sklearn/평활): 기존 baselines_
    if corr=="ExtraTrees": return f"{R}/baselines_{de}/persubject_extratrees_residual.csv"
    if corr=="MLP":        return f"{R}/baselines_{de}/persubject_mlp_residual.csv"
    if corr in ("SavGol","Kalman"):
        return f"{R}/baselines_{de}/persubject_{corr.lower()}.csv"
    return None
M1=np.full((len(CORR_ORDER),len(ESTS)),np.nan)
for i,c in enumerate(CORR_ORDER):
    for j,(ek,_) in enumerate(ESTS):
        vals=[imp_mean(within_file(c,d,ek)) for d in DOMS]
        vals=[v for v in vals if not np.isnan(v)]
        if vals: M1[i,j]=np.mean(vals)

# ---------- ② pooled ----------
KEY={"TabM":"tabm","ExtraTrees":"extratrees","ExcelFormer":"excelformer","Diffusion":"diffusion",
     "MLP":"mlp","TCN":"tcn","SmoothNet":"smoothnet","SavGol":"savgol","Kalman":"kalman"}
M2=np.full((len(CORR_ORDER),len(ESTS)),np.nan)
for i,c in enumerate(CORR_ORDER):
    for j,(ek,_) in enumerate(ESTS):
        M2[i,j]=imp_mean(f"{R}/pooled5_out/{KEY[c]}_{ek}/persubject_{KEY[c]}.csv")

# ---------- ③ LODO: held-out × 보정기(6), 추정기 평균 ----------
HELD=["AIHub","SUMediPose","REHAB","FiT3D"]; HELD_LBL=["AIHub","SUMe","REHAB","FiT3D"]
LC=["ExtraTrees","TabM","ExcelFormer","SmoothNet","TCN","Diffusion","MLP"]
M3=np.full((len(LC),len(HELD)),np.nan)
lodo={}
for f in glob.glob(f"{R}/lodo_base/outB_*/lodo_*.csv"):
    corr=os.path.basename(f).replace("lodo_","").replace(".csv","")
    d=pd.read_csv(f)
    for _,r in d.iterrows():
        lodo.setdefault((corr,r["held_out"]),[]).append(r["improvement_pooled_pct"])
# ExtraTrees LODO는 별도 위치(lodo_{est}/lodo_metrics.csv)
for f in glob.glob(f"{R}/lodo_*/lodo_metrics.csv"):
    d=pd.read_csv(f)
    for _,r in d.iterrows():
        lodo.setdefault(("extratrees",r["held_out"]),[]).append(r["improvement_pooled_pct"])
for i,c in enumerate(LC):
    for j,h in enumerate(HELD):
        v=lodo.get((c.lower(),h));
        if v: M3[i,j]=np.mean(v)

def heat(M,rows,cols,title,fn,vlim=60):
    fig,ax=plt.subplots(figsize=(1.4*len(cols)+3.2,0.62*len(rows)+1.6))
    norm=TwoSlopeNorm(vmin=-vlim,vcenter=0,vmax=vlim)
    im=ax.imshow(M,cmap="RdYlGn",norm=norm,aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,fontsize=10)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows,fontsize=10)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(M[i,j]):
                ax.text(j,i,f"{M[i,j]:+.0f}",ha="center",va="center",fontsize=9,
                        color="black")
    ax.set_title(title,fontsize=13,pad=8)
    cb=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.02); cb.set_label("개선율 %",fontsize=9)
    plt.tight_layout(); plt.savefig(f"{PRES}/{fn}",dpi=190); plt.close()
    print(f"{fn}: rows={len(rows)} cols={len(cols)}  결측={int(np.isnan(M).sum())}")

heat(M1,CORR_ORDER,[l for _,l in ESTS],
     "① 도메인 내 — 보정기×추정기 (도메인 4개 평균, 개선율%)","cube_within.png")
heat(M2,CORR_ORDER,[l for _,l in ESTS],
     "② pooled — 보정기×추정기 (개선율%)","cube_pooled.png")
heat(M3,LC,HELD_LBL,
     "③ 도메인 간(LODO) — 보정기×held-out도메인 (추정기 평균, 개선율%)","cube_lodo.png",vlim=30)
