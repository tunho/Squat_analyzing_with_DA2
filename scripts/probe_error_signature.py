"""
Probe ① estimator×pose error signature + ①-b directional (hysteresis) error.

Question ①: Does each estimator have a reproducible residual-vs-flexion-angle
curve (a "signature"), and does that curve TRANSFER across domains?
 - If curves are estimator-distinct AND domain-stable -> genuine architecture
   signature (interesting: NOT reducible to covariate shift).
 - If curves reshuffle per domain -> just covariate shift (dead).

Question ①-b: At the SAME knee angle, does residual differ between descent
(flexing, dv<0) and ascent (extending, dv>0)? A consistent gap = velocity/
direction-dependent bias, i.e. error is not a pure function of pose.
"""
import sys
import numpy as np
import pandas as pd

FILES = {
    "mediapipe": "experiments/paper/features_pooled4_offstd.csv",
    "nlf": "experiments/paper/features_pooled4_nlf.csv",
    "rtmw3d": "experiments/paper/features_pooled4_rtmw3d.csv",
    "motionbert": "experiments/paper/features_pooled4_motionbert.csv",
}

BINS = np.arange(60, 181, 10)  # flexion-angle bins (deg)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2


def domain_of(sid: str) -> str:
    s = str(sid)
    if s.startswith("fit3d"):
        return "fit3d"
    if s.startswith("PM_"):
        return "ex6"
    if s.startswith(("CA", "CB")):
        return "aihub"
    if s.startswith("S"):
        return "sume"
    return "other"


def load(est: str, path: str) -> pd.DataFrame:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    want = ["frame_index", "subject_id", "mp_knee_angle", "gt_angle"]
    idcols = [c for c in ["camera", "take", "ex_type", "cam_id"] if c in cols]
    df = pd.read_csv(path, usecols=want + idcols, low_memory=False)
    df = df.dropna(subset=["mp_knee_angle", "gt_angle"])
    df["domain"] = df["subject_id"].map(domain_of)
    df["residual"] = df["gt_angle"] - df["mp_knee_angle"]
    df["estimator"] = est
    # group key = one continuous video clip
    gk = ["subject_id"] + idcols
    df = df.sort_values(gk + ["frame_index"])
    grp = df.groupby(gk, sort=False)
    # velocity of gt angle across consecutive frames (frame gap == 1 only)
    df["fdiff"] = grp["frame_index"].diff()
    df["gt_dv"] = grp["gt_angle"].diff()
    df.loc[df["fdiff"] != 1, "gt_dv"] = np.nan
    return df


def curve(df: pd.DataFrame) -> pd.Series:
    """mean residual per flexion-angle bin (indexed by bin center)."""
    b = pd.cut(df["gt_angle"], BINS, labels=BIN_CENTERS)
    return df.groupby(b, observed=False)["residual"].mean()


def main():
    frames = []
    for est, path in FILES.items():
        try:
            frames.append(load(est, path))
        except Exception as e:  # noqa
            print(f"[skip] {est}: {e}")
    data = pd.concat(frames, ignore_index=True)
    domains = ["ex6", "fit3d", "sume", "aihub"]
    estimators = list(FILES.keys())

    # ---- residual curves: estimator x domain ----
    print("\n===== ① residual-vs-flexion curves (mean residual per 10deg bin) =====")
    curves = {}
    for est in estimators:
        for dom in domains:
            sub = data[(data.estimator == est) & (data.domain == dom)]
            if len(sub) < 200:
                continue
            curves[(est, dom)] = curve(sub)
    # print compact table per estimator
    for est in estimators:
        print(f"\n-- {est} --  (rows: bins {list(BIN_CENTERS.astype(int))})")
        for dom in domains:
            c = curves.get((est, dom))
            if c is None:
                continue
            vals = " ".join(f"{v:6.1f}" if pd.notna(v) else "   nan" for v in c.values)
            print(f"  {dom:6s}: {vals}")

    # ---- Q1: are estimators distinct within a fixed domain? ----
    print("\n===== Q1: estimator curves distinct? (pairwise corr within domain) =====")
    for dom in domains:
        cs = {est: curves[(est, dom)] for est in estimators if (est, dom) in curves}
        ests = list(cs)
        corrs = []
        for i in range(len(ests)):
            for j in range(i + 1, len(ests)):
                a, b = cs[ests[i]], cs[ests[j]]
                m = a.notna() & b.notna()
                if m.sum() >= 4:
                    corrs.append(np.corrcoef(a[m], b[m])[0, 1])
        if corrs:
            print(f"  {dom:6s}: mean pairwise curve-corr = {np.mean(corrs):+.2f}  "
                  f"(low/neg = estimators distinct; high = redundant)")

    # ---- Q2: does an estimator's curve TRANSFER across domains? ----
    print("\n===== Q2: does each estimator's signature transfer across domains? =====")
    print("       (corr of same estimator's curve between domain pairs; "
          "high = domain-invariant signature)")
    for est in estimators:
        cs = {dom: curves[(est, dom)] for dom in domains if (est, dom) in curves}
        doms = list(cs)
        corrs = []
        for i in range(len(doms)):
            for j in range(i + 1, len(doms)):
                a, b = cs[doms[i]], cs[doms[j]]
                m = a.notna() & b.notna()
                if m.sum() >= 4:
                    corrs.append((f"{doms[i]}/{doms[j]}", np.corrcoef(a[m], b[m])[0, 1]))
        if corrs:
            mean_c = np.mean([c for _, c in corrs])
            detail = " ".join(f"{p}={c:+.2f}" for p, c in corrs)
            print(f"  {est:11s}: mean cross-domain curve-corr = {mean_c:+.2f}   [{detail}]")

    # ---- ①-b: directional / hysteresis gap ----
    print("\n===== ①-b directional (hysteresis) error: descent(dv<0) - ascent(dv>0) =====")
    print("       per bin: mean residual_descent - residual_ascent (deg). "
          "consistent sign across bins/domains = velocity-dependent bias.")
    d = data.dropna(subset=["gt_dv"])
    d = d[d["gt_dv"].abs() > 0.3]  # ignore near-static frames
    for est in estimators:
        print(f"\n-- {est} --")
        gaps_all = []
        for dom in domains:
            sub = d[(d.estimator == est) & (d.domain == dom)]
            if len(sub) < 400:
                continue
            b = pd.cut(sub["gt_angle"], BINS, labels=BIN_CENTERS)
            desc = sub[sub.gt_dv < 0].groupby(b, observed=False)["residual"].mean()
            asc = sub[sub.gt_dv > 0].groupby(b, observed=False)["residual"].mean()
            gap = (desc - asc)
            valid = gap.dropna()
            if len(valid) < 3:
                continue
            gaps_all.extend(valid.values)
            vals = " ".join(f"{v:+5.1f}" if pd.notna(v) else "  nan" for v in gap.values)
            print(f"  {dom:6s}: {vals}   | mean|gap|={valid.abs().mean():.2f}")
        if gaps_all:
            ga = np.array(gaps_all)
            frac_same = max((ga > 0).mean(), (ga < 0).mean())
            print(f"  >> pooled: mean gap={ga.mean():+.2f}, mean|gap|={np.abs(ga).mean():.2f}, "
                  f"sign-consistency={frac_same:.0%}")


if __name__ == "__main__":
    main()
