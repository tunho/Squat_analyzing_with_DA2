"""통계: ours 개선율 부트스트랩 95% CI(피험자 재표집) + ours vs linreg paired Wilcoxon(피험자별 MAE)."""
import argparse, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression
from paper_train_loso import prepare_v6_data
from paper_common import FEATURE_SETS

def mae(a,b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--ours-pred", required=True)
    ap.add_argument("--feature-set", default="enhanced_safe")
    ap.add_argument("--boot", type=int, default=10000)
    args=ap.parse_args()

    # ours: 피험자별 raw/corr MAE + 개선율
    p=pd.read_csv(args.ours_pred)
    rows=[]
    for s,g in p.groupby("subject_id"):
        r=mae(g.gt_angle,g.mp_knee_angle); c=mae(g.gt_angle,g.corrected_angle)
        rows.append((s,r,c,(r-c)/r*100))
    o=pd.DataFrame(rows,columns=["subject_id","raw","ours","imp"]).set_index("subject_id")

    # linreg 피험자별 corr MAE (동일 LOSO)
    df=prepare_v6_data(pd.read_csv(args.features), args.feature_set)
    feats=FEATURE_SETS[args.feature_set]
    subs=sorted(df.subject_id.unique())
    lin={}
    for s in subs:
        tr=df[df.subject_id!=s]; te=df[df.subject_id==s]
        m=LinearRegression().fit(tr[feats], tr["residual"])
        corr=te["mp_knee_angle"]+m.predict(te[feats])
        lin[s]=mae(te.gt_angle, corr)
    o["linreg"]=o.index.map(lambda s: lin.get(str(s), lin.get(s, np.nan)))

    # 부트스트랩 CI (피험자 재표집) on 평균 개선율
    imp=o["imp"].to_numpy(); n=len(imp); rng=np.random.default_rng(42)
    boot=np.array([imp[rng.integers(0,n,n)].mean() for _ in range(args.boot)])
    lo,hi=np.percentile(boot,[2.5,97.5]); mean=imp.mean()

    # Wilcoxon: ours corr MAE vs linreg corr MAE (피험자별, paired)
    valid=o.dropna(subset=["linreg"])
    W,pv=wilcoxon(valid["ours"], valid["linreg"])

    print(f"=== 통계 (pooled4, {n}명) ===")
    print(f"평균 개선율: {mean:.1f}%  (95% CI [{lo:.1f}, {hi:.1f}])")
    print(f"피험자별 개선 {sum(o.imp>0)}/{n}")
    print(f"ours vs linreg (피험자 MAE, Wilcoxon): ours {valid.ours.mean():.2f}° vs linreg {valid.linreg.mean():.2f}°, p={pv:.2e}")
    o.round(2).to_csv(args.ours_pred.rsplit('/',1)[0]+"/per_subject_stats.csv")

if __name__=="__main__":
    main()
