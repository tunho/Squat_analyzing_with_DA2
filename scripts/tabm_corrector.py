from __future__ import annotations
"""TabM (ICLR 2025, arXiv 2410.24210) 잔차 보정기 — 최신 tabular DL baseline.
MLP + parameter-efficient ensembling(BatchEnsemble). ExtraTrees/CARD/ExcelFormer와 동일한
v6 피처·동일 LOSO 프로토콜. 전체 데이터 학습(서브샘플 불필요).
출력: persubject_tabm.csv (기존 비교표 합류).
사용: ../.venv-nlf/bin/python tabm_corrector.py --features ... --output ... --device cuda
"""
import argparse, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import sys; sys.path.insert(0, os.path.dirname(__file__))
from paper_common import FEATURE_SETS
from paper_train_loso import prepare_v6_data
from tabm import TabM

def mae(a,b): return float(np.mean(np.abs(np.asarray(a,float)-np.asarray(b,float))))
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))

class TabMCorrector:
    """프레임별 v6 피처 -> 잔차(gt-mp) 회귀. 예측 = k개 앙상블 멤버 평균."""
    def __init__(self, n_feat, device="cuda", epochs=40, lr=2e-3, bs=8192,
                 k=16, n_blocks=3, d_block=512, dropout=0.1):
        self.n=n_feat; self.device=torch.device(device if torch.cuda.is_available() else "cpu")
        self.epochs=epochs; self.lr=lr; self.bs=bs
        self.k=k; self.n_blocks=n_blocks; self.d_block=d_block; self.dropout=dropout
    def fit(self,X,y):
        self.xm,self.xs=X.mean(0),X.std(0)+1e-6; self.ym,self.ys=y.mean(),y.std()+1e-6
        Xn=torch.tensor((X-self.xm)/self.xs,dtype=torch.float32,device=self.device)
        yn=torch.tensor((y-self.ym)/self.ys,dtype=torch.float32,device=self.device)
        self.model=TabM(n_num_features=self.n,d_out=1,k=self.k,n_blocks=self.n_blocks,
            d_block=self.d_block,dropout=self.dropout,arch_type="tabm",
            start_scaling_init="random-signs").to(self.device)
        opt=torch.optim.AdamW(self.model.parameters(),lr=self.lr,weight_decay=1e-4)
        n=Xn.shape[0]; self.model.train()
        for ep in range(self.epochs):
            perm=torch.randperm(n,device=self.device)
            for i in range(0,n,self.bs):
                idx=perm[i:i+self.bs]
                out=self.model(Xn[idx])                     # [B, k, 1]
                loss=((out-yn[idx][:,None,None])**2).mean() # 각 멤버 독립 학습
                opt.zero_grad(); loss.backward(); opt.step()
        return self
    @torch.no_grad()
    def predict(self,X):
        Xn=torch.tensor((X-self.xm)/self.xs,dtype=torch.float32,device=self.device)
        self.model.eval(); out=[]
        for i in range(0,Xn.shape[0],self.bs):
            o=self.model(Xn[i:i+self.bs])                   # [B, k, 1]
            out.append(o.mean(1).squeeze(-1).cpu())         # 앙상블 평균
        return (torch.cat(out).numpy())*self.ys+self.ym

def run_loso(df,feats,device,epochs):
    subs=sorted(df["subject_id"].unique()); rows=[]; preds=[]
    for i,s in enumerate(subs,1):
        t0=time.time()
        tr=df[df.subject_id!=s]; te=df[df.subject_id==s].copy()
        m=TabMCorrector(len(feats),device=device,epochs=epochs).fit(
            tr[feats].to_numpy(float),tr["residual"].to_numpy(float))
        te["corrected_angle"]=te["mp_knee_angle"].to_numpy(float)+m.predict(te[feats].to_numpy(float))
        raw=mae(te.gt_angle,te.mp_knee_angle); corr=mae(te.gt_angle,te.corrected_angle)
        rows.append(dict(subject_id=str(s),mae_raw=raw,mae_corrected=corr,
            rmse_raw=rmse(te.gt_angle,te.mp_knee_angle),rmse_corrected=rmse(te.gt_angle,te.corrected_angle),
            improvement_pct=(raw-corr)/raw*100 if raw else 0.0))
        preds.append(te)
        print(f"  [tabm] fold {i}/{len(subs)} {s}: {raw:.2f}->{corr:.2f} ({rows[-1]['improvement_pct']:+.1f}%) {time.time()-t0:.0f}s",flush=True)
    return pd.DataFrame(rows),(pd.concat(preds,ignore_index=True) if preds else pd.DataFrame())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True,type=Path); p.add_argument("--output",required=True,type=Path)
    p.add_argument("--feature-set",default="v6"); p.add_argument("--device",default="cuda"); p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--seed",type=int,default=None)
    a=p.parse_args()
    from seed_util import set_all_seeds; set_all_seeds(a.seed)
    df=prepare_v6_data(pd.read_csv(a.features,low_memory=False),feature_set=a.feature_set)
    feats=FEATURE_SETS[a.feature_set]; a.output.mkdir(parents=True,exist_ok=True)
    subj,preds=run_loso(df,feats,a.device,a.epochs)
    subj.to_csv(a.output/"persubject_tabm.csv",index=False)
    if len(preds):
        keep=[c for c in ["subject_id","view_type","camera","dataset","frame_index","gt_angle","mp_knee_angle","corrected_angle"] if c in preds.columns]
        preds[keep].to_csv(a.output/"predictions_tabm.csv",index=False)
    print(f"\n[tabm] subject-mean improvement: {subj.improvement_pct.mean():+.2f}% | raw {subj.mae_raw.mean():.2f} -> corr {subj.mae_corrected.mean():.2f}")
if __name__=="__main__": main()
