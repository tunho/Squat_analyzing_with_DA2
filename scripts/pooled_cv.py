#!/usr/bin/env python
"""pooled(②) 평가 — 사람 단위 K-fold CV (기본 5). LOSO(N-fold) 대신 K번만 재학습 → 훨씬 빠름.
같은 '안 본 사람 일반화' 지표(improvement_pct)를 K-fold 평균으로 낸다.
사용: python pooled_cv.py --features features_pooled4_offstd.csv --corrector tabm --output ... --device cuda
"""
import argparse, os, time
from pathlib import Path
import numpy as np, pandas as pd
import sys; sys.path.insert(0, os.path.dirname(__file__))
from paper_common import FEATURE_SETS
from paper_train_loso import prepare_v6_data
from sklearn.model_selection import GroupKFold

def mae(a,b): return float(np.mean(np.abs(np.asarray(a,float)-np.asarray(b,float))))

def build(corrector, nfeat, device):
    if corrector=="tabm":
        from tabm_corrector import TabMCorrector; return TabMCorrector(nfeat, device=device)
    if corrector=="excelformer":
        from excelformer_corrector import ExcelFormerCorrector; return ExcelFormerCorrector(nfeat, device=device)
    if corrector=="extratrees":
        from sklearn.ensemble import ExtraTreesRegressor
        class ET:
            def __init__(s): s.m=ExtraTreesRegressor(n_estimators=300,max_depth=15,random_state=42,n_jobs=-1)
            def fit(s,X,y): s.m.fit(X,y); return s
            def predict(s,X): return s.m.predict(X)
        return ET()
    if corrector=="mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        class ML:
            def __init__(s): s.sc=StandardScaler(); s.m=MLPRegressor(hidden_layer_sizes=(256,128),max_iter=300,random_state=42)
            def fit(s,X,y): s.m.fit(s.sc.fit_transform(X),y); return s
            def predict(s,X): return s.m.predict(s.sc.transform(X))
        return ML()
    if corrector=="diffusion":
        from diffusion_corrector import DiffusionCorrector
        return DiffusionCorrector(nfeat, device=device, epochs=200)
    raise ValueError(corrector)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True,type=Path); p.add_argument("--output",required=True,type=Path)
    p.add_argument("--corrector",required=True,choices=["tabm","excelformer","extratrees","mlp","diffusion","smoothnet","tcn","savgol","kalman"])
    p.add_argument("--feature-set",default="v6"); p.add_argument("--device",default="cuda")
    p.add_argument("--folds",type=int,default=5,help="0 이하면 LOSO(leave-one-subject-out)"); p.add_argument("--seed",type=int,default=None)
    a=p.parse_args()
    if a.seed is not None:
        from seed_util import set_all_seeds; set_all_seeds(a.seed)
    df=prepare_v6_data(pd.read_csv(a.features,low_memory=False),feature_set=a.feature_set)
    feats=FEATURE_SETS[a.feature_set]; a.output.mkdir(parents=True,exist_ok=True)
    subs=df["subject_id"].to_numpy()
    if a.folds<=0:
        from sklearn.model_selection import LeaveOneGroupOut
        splitter=LeaveOneGroupOut(); nsplit=len(set(subs))
    else:
        splitter=GroupKFold(n_splits=a.folds); nsplit=a.folds
    WINDOWED={"smoothnet","tcn"}; SMOOTHER={"savgol","kalman"}
    WIN,STR=32,8
    rows=[]; preds=[]
    for fi,(tr_idx,te_idx) in enumerate(splitter.split(df,groups=subs),1):
        t0=time.time()
        tr=df.iloc[tr_idx]; te=df.iloc[te_idx].copy()
        if a.corrector in WINDOWED:
            from smoothnet_corrector import collect_windows, predict_seq
            train_fn=__import__("smoothnet_corrector").train_smoothnet if a.corrector=="smoothnet" else __import__("tcn_corrector").train_tcn
            A,R=collect_windows(tr,WIN,STR)
            if A is None: te["corrected_angle"]=te["mp_knee_angle"].astype(float)
            else:
                model,stats=train_fn(A,R,WIN,a.device,epochs=60); te=predict_seq(model,stats,te,WIN,STR,a.device)
        elif a.corrector in SMOOTHER:
            from scipy.signal import savgol_filter
            from paper_baselines_loso import _kalman_smooth_1d
            te=te.sort_values(["subject_id","view_type","frame_index"]).copy()
            cor=te["mp_knee_angle"].to_numpy(float).copy()
            gcols=["subject_id","view_type"]+(["camera"] if "camera" in te.columns else [])
            for _,idx in te.groupby(gcols).groups.items():
                pos=[te.index.get_loc(i) for i in idx]; z=cor[pos]
                if len(z)<5: continue
                if a.corrector=="savgol":
                    w=min(11,len(z)-(1-len(z)%2)); w=w if w%2==1 else w-1
                    if w>=3: cor[pos]=savgol_filter(z,w,2)
                else: cor[pos]=_kalman_smooth_1d(z,1.0,25.0)
            te["corrected_angle"]=cor
        else:
            m=build(a.corrector,len(feats),a.device).fit(tr[feats].to_numpy(float),tr["residual"].to_numpy(float))
            te["corrected_angle"]=te["mp_knee_angle"].to_numpy(float)+m.predict(te[feats].to_numpy(float))
        # fold 내 사람별 개선율
        for s,g in te.groupby("subject_id"):
            raw=mae(g.gt_angle,g.mp_knee_angle); cor=mae(g.gt_angle,g.corrected_angle)
            rows.append(dict(subject_id=str(s),fold=fi,mae_raw=raw,mae_corrected=cor,
                             improvement_pct=(raw-cor)/raw*100 if raw else 0.0))
        preds.append(te)
        print(f"  [{a.corrector} pooled] fold {fi}/{nsplit}: n_test_subj={te['subject_id'].nunique()} ({time.time()-t0:.0f}s)",flush=True)
    per=pd.DataFrame(rows); per.to_csv(a.output/f"persubject_{a.corrector}.csv",index=False)
    print(f"\n[{a.corrector} pooled-{'LOSO' if a.folds<=0 else 'CV'}] subject-mean improvement: {per.improvement_pct.mean():+.2f}%  (folds={nsplit}, subj={per.subject_id.nunique()})")

if __name__=="__main__": main()
