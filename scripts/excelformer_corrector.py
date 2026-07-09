from __future__ import annotations
"""ExcelFormer (KDD 2024, arXiv 2023) 잔차 보정기 — 최신 tabular DL baseline.
ExtraTrees/CARD와 동일한 v6 피처·동일 LOSO 프로토콜. 전체 데이터 학습(서브샘플 불필요).
출력: persubject_excelformer.csv (기존 비교표 합류).
사용: ../.venv-nlf/bin/python excelformer_corrector.py --features ... --output ... --device cuda
"""
import argparse, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import sys; sys.path.insert(0, os.path.dirname(__file__))
from paper_common import FEATURE_SETS
from paper_train_loso import prepare_v6_data
import torch_frame
from torch_frame import stype
from torch_frame.data import Dataset
from torch_frame.nn.models import ExcelFormer

def mae(a,b): return float(np.mean(np.abs(np.asarray(a,float)-np.asarray(b,float))))
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))

class ExcelFormerCorrector:
    def __init__(self, n_feat, device="cuda", epochs=40, lr=1e-3, bs=8192,
                 dim=32, layers=3, heads=4):
        self.n=n_feat; self.device=torch.device(device if torch.cuda.is_available() else "cpu")
        self.epochs=epochs; self.lr=lr; self.bs=bs; self.dim=dim; self.layers=layers; self.heads=heads
        self.cols=[f"f{i}" for i in range(n_feat)]
        self.c2s={c:stype.numerical for c in self.cols}; self.c2s["target"]=stype.numerical
    def _tf(self,X,y=None):
        Xn=(X-self.xm)/self.xs
        df=pd.DataFrame(Xn,columns=self.cols)
        df["target"]=(0.0 if y is None else (y-self.ym)/self.ys)
        ds=Dataset(df,col_to_stype=self.c2s,target_col="target"); ds.materialize()
        return ds
    def fit(self,X,y):
        self.xm,self.xs=X.mean(0),X.std(0)+1e-6; self.ym,self.ys=y.mean(),y.std()+1e-6
        ds=self._tf(X,y); tf=ds.tensor_frame.to(self.device)
        self.model=ExcelFormer(in_channels=self.dim,out_channels=1,num_cols=self.n,
            num_layers=self.layers,num_heads=self.heads,
            col_stats=ds.col_stats,col_names_dict=tf.col_names_dict).to(self.device)
        opt=torch.optim.AdamW(self.model.parameters(),lr=self.lr,weight_decay=1e-5)
        n=tf.num_rows; self.model.train()
        for ep in range(self.epochs):
            perm=torch.randperm(n,device=self.device)
            for i in range(0,n,self.bs):
                idx=perm[i:i+self.bs]; sub=tf[idx]
                out=self.model(sub).squeeze(-1); loss=torch.mean((out-sub.y)**2)
                opt.zero_grad(); loss.backward(); opt.step()
        return self
    @torch.no_grad()
    def predict(self,X):
        ds=self._tf(X,None); tf=ds.tensor_frame.to(self.device); self.model.eval()
        out=[]
        for i in range(0,tf.num_rows,self.bs):
            out.append(self.model(tf[torch.arange(i,min(i+self.bs,tf.num_rows),device=self.device)]).squeeze(-1).cpu())
        return (torch.cat(out).numpy())*self.ys+self.ym

def run_loso(df,feats,device,epochs):
    subs=sorted(df["subject_id"].unique()); rows=[]; preds=[]
    for i,s in enumerate(subs,1):
        t0=time.time()
        tr=df[df.subject_id!=s]; te=df[df.subject_id==s].copy()
        m=ExcelFormerCorrector(len(feats),device=device,epochs=epochs).fit(
            tr[feats].to_numpy(float),tr["residual"].to_numpy(float))
        te["corrected_angle"]=te["mp_knee_angle"].to_numpy(float)+m.predict(te[feats].to_numpy(float))
        raw=mae(te.gt_angle,te.mp_knee_angle); corr=mae(te.gt_angle,te.corrected_angle)
        rows.append(dict(subject_id=str(s),mae_raw=raw,mae_corrected=corr,
            rmse_raw=rmse(te.gt_angle,te.mp_knee_angle),rmse_corrected=rmse(te.gt_angle,te.corrected_angle),
            improvement_pct=(raw-corr)/raw*100 if raw else 0.0))
        preds.append(te)
        print(f"  [excelformer] fold {i}/{len(subs)} {s}: {raw:.2f}->{corr:.2f} ({rows[-1]['improvement_pct']:+.1f}%) {time.time()-t0:.0f}s",flush=True)
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
    subj.to_csv(a.output/"persubject_excelformer.csv",index=False)
    if len(preds):
        keep=[c for c in ["subject_id","view_type","camera","dataset","frame_index","gt_angle","mp_knee_angle","corrected_angle"] if c in preds.columns]
        preds[keep].to_csv(a.output/"predictions_excelformer.csv",index=False)
    print(f"\n[excelformer] subject-mean improvement: {subj.improvement_pct.mean():+.2f}% | raw {subj.mae_raw.mean():.2f} -> corr {subj.mae_corrected.mean():.2f}")
if __name__=="__main__": main()
