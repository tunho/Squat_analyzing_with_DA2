"""베이스라인 비교 (동일 LOSO·동일 std CSV). ours(ExtraTrees)는 기존 결과 재사용.
비교군: raw / savgol(시간평활) / global affine / per-subject calib(초기K) / linear-reg(피처) / ours.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from paper_train_loso import prepare_v6_data
from paper_common import FEATURE_SETS

def mae(a,b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))

def src(s):
    s=str(s)
    if s.startswith('SUME_'): return 'SUMediPose'
    if s[:2] in ('CA','CB','CI'): return 'AIHub'
    if s.startswith('PM'): return 'REHAB'
    return 'FiT3D'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--feature-set", default="enhanced_causal")
    ap.add_argument("--ours-pred", required=True, help="기존 ExtraTrees loso_predictions.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--calib-k", type=int, default=15)
    args=ap.parse_args()

    df=prepare_v6_data(pd.read_csv(args.features), args.feature_set)
    feats=FEATURE_SETS[args.feature_set]
    seqcols=["subject_id","view_type"]+(["camera"] if "camera" in df.columns else [])
    subs=sorted(df.subject_id.unique())

    df=df.sort_values(seqcols+["frame_index"]).reset_index(drop=True)
    df["raw"]=df["mp_knee_angle"]
    # savgol: 시퀀스별 mp 평활 (학습불필요), 위치기반 정렬로 안전하게
    sgv=df["mp_knee_angle"].to_numpy().copy()
    for _,g in df.groupby(seqcols):
        idx=g.index.to_numpy(); n=len(idx)
        if n>=7:
            w=min(31, n if n%2==1 else n-1)
            if 5<=w<=n:
                sgv[idx]=savgol_filter(df.loc[idx,"mp_knee_angle"].to_numpy(), w, 2)
    df["savgol"]=sgv

    df["affine"]=np.nan; df["calib"]=np.nan; df["linreg"]=np.nan
    for s in subs:
        tr=df[df.subject_id!=s]; te=df[df.subject_id==s]
        # global affine: gt ~ a*mp+b (train)
        a=LinearRegression().fit(tr[["mp_knee_angle"]], tr["gt_angle"])
        df.loc[te.index,"affine"]=a.predict(te[["mp_knee_angle"]])
        # per-subject calib: test 초기 K프레임 평균 오프셋
        off=(te["gt_angle"]-te["mp_knee_angle"]).iloc[:args.calib_k].mean()
        df.loc[te.index,"calib"]=te["mp_knee_angle"]+off
        # linear reg on features → residual
        lr=LinearRegression().fit(tr[feats], tr["residual"])
        df.loc[te.index,"linreg"]=te["mp_knee_angle"]+lr.predict(te[feats])

    # ours
    op=pd.read_csv(args.ours_pred)
    # 정합: subject_id+frame_index+camera로 merge (가능하면), 아니면 행순
    key=[c for c in ["subject_id","frame_index","camera","view_type"] if c in op.columns and c in df.columns]
    ours=op.set_index(key)["corrected_angle"]
    df=df.set_index(key)
    df["ours"]=ours
    df=df.reset_index()

    df["dsname"]=df.subject_id.map(src)
    methods=["raw","savgol","affine","calib","linreg","ours"]

    def metrics_table(frame):
        rows=[]
        for scope,sub in [("전체",frame)]+[(ds,g) for ds,g in frame.groupby("dsname")]:
            rec={"scope":scope,"n":len(sub)}
            for m in methods:
                v=sub.dropna(subset=[m])
                rec[m]=round(mae(v.gt_angle, v[m]),2)
            rows.append(rec)
        return pd.DataFrame(rows)

    out=metrics_table(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    # deep flexion(<110°) 버전도 출력
    deep_path=str(args.output).replace(".csv","_deep.csv")
    metrics_table(df[df.gt_angle<110]).to_csv(deep_path, index=False)
    print(f"[deep<110] saved: {deep_path}")
    # 개선율(vs raw) 표 출력
    print("=== 베이스라인 MAE (°) ===")
    print(out.to_string(index=False))
    print("\n=== 개선율 vs raw (%) ===")
    for _,r in out.iterrows():
        imps=" ".join(f"{m}:{(r['raw']-r[m])/r['raw']*100:+.1f}%" for m in methods[1:])
        print(f"  {r['scope']:<11}(raw {r['raw']}°) {imps}")

if __name__=="__main__":
    main()
