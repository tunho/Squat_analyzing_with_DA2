"""
Decisive test: does ① have ANY structure beyond the already-refuted
"monotone-residual-vs-angle + per-domain affine offset"?

Two confounds to kill:
 1. Pearson corr is affine-invariant -> high Q2 corr just means curves are
    affine-related = the REFUTED shared-shape x domain-affine finding.
 2. All curves are ~monotone in angle -> corr trivially high.

So we ask the question that actually matters for correction:
  If you take estimator E's residual curve from domain A and USE IT to correct
  domain B (curve-level transfer, no refit), how much error remains vs (a) doing
  nothing, (b) an oracle per-domain affine refit of A's curve onto B?

 - If A->B raw transfer already ~ oracle-affine -> only affine differs = REFUTED.
 - If detrending (remove linear-in-angle trend) leaves near-zero residual
   structure -> the only signal is monotone trend = trivial.
 - Anything survives BOTH -> genuinely new structure worth chasing.
"""
import numpy as np
import pandas as pd
from probe_error_signature import FILES, BINS, BIN_CENTERS, domain_of

DOMAINS = ["ex6", "fit3d", "sume", "aihub"]


def curves_by(est, path):
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    df = pd.read_csv(path, usecols=["subject_id", "mp_knee_angle", "gt_angle"],
                     low_memory=False).dropna()
    df["domain"] = df["subject_id"].map(domain_of)
    df["residual"] = df["gt_angle"] - df["mp_knee_angle"]
    out = {}
    for dom in DOMAINS:
        sub = df[df.domain == dom]
        if len(sub) < 200:
            continue
        b = pd.cut(sub["gt_angle"], BINS, labels=BIN_CENTERS)
        out[dom] = sub.groupby(b, observed=False)["residual"].mean()
    return out


def detrend(c):
    x = np.array([float(i) for i in c.index], float)
    y = c.values.astype(float)
    m = ~np.isnan(y)
    if m.sum() < 4:
        return c * np.nan
    a, b = np.polyfit(x[m], y[m], 1)
    return pd.Series(y - (a * x + b), index=c.index)


def affine_fit(src, tgt):
    """best a*src+b to match tgt; return residual RMSE after affine."""
    m = src.notna() & tgt.notna()
    if m.sum() < 4:
        return np.nan
    s, t = src[m].values, tgt[m].values
    a, b = np.polyfit(s, t, 1)
    return float(np.sqrt(np.mean((t - (a * s + b)) ** 2)))


def rmse(a, b):
    m = a.notna() & b.notna()
    return float(np.sqrt(np.mean((a[m].values - b[m].values) ** 2))) if m.sum() else np.nan


def main():
    print("Curve-level transfer test (units = deg of residual error on the curve)\n")
    print(f"{'est':11s} | {'A->B raw':>9s} {'oracle-affine':>13s} {'do-nothing':>11s} "
          f"| {'detrend xcorr':>13s} {'detrend-distinct':>16s}")
    print("-" * 92)
    for est, path in FILES.items():
        cv = curves_by(est, path)
        doms = list(cv)
        raw, aff, none = [], [], []
        for i in range(len(doms)):
            for j in range(len(doms)):
                if i == j:
                    continue
                A, B = cv[doms[i]], cv[doms[j]]
                raw.append(rmse(A, B))          # use A's curve as-is on B
                aff.append(affine_fit(A, B))    # oracle affine refit A->B
                none.append(rmse(B, B * 0))     # residual if uncorrected (=|B|)
        # detrended structure
        dt = {d: detrend(cv[d]) for d in doms}
        dxc = []
        for i in range(len(doms)):
            for j in range(i + 1, len(doms)):
                a, b = dt[doms[i]], dt[doms[j]]
                m = a.notna() & b.notna()
                if m.sum() >= 4 and a[m].std() > 1e-6 and b[m].std() > 1e-6:
                    dxc.append(np.corrcoef(a[m], b[m])[0, 1])
        # detrended estimator-distinctness handled globally below
        print(f"{est:11s} | {np.nanmean(raw):9.2f} {np.nanmean(aff):13.2f} "
              f"{np.nanmean(none):11.2f} | {np.nanmean(dxc):+13.2f} "
              f"{'(see below)':>16s}")

    # After detrending, are estimators still distinct within a domain,
    # or is the only per-estimator signal the linear trend slope?
    print("\nDetrended within-domain estimator distinctness "
          "(mean pairwise corr of detrended curves; ~0 = no structure left):")
    all_cv = {est: curves_by(est, path) for est, path in FILES.items()}
    for dom in DOMAINS:
        dts = {}
        for est in FILES:
            if dom in all_cv[est]:
                dts[est] = detrend(all_cv[est][dom])
        ests = list(dts)
        cc = []
        for i in range(len(ests)):
            for j in range(i + 1, len(ests)):
                a, b = dts[ests[i]], dts[ests[j]]
                m = a.notna() & b.notna()
                if m.sum() >= 4 and a[m].std() > 1e-6 and b[m].std() > 1e-6:
                    cc.append(np.corrcoef(a[m], b[m])[0, 1])
        print(f"  {dom:6s}: {np.nanmean(cc):+.2f}")

    print("\nRead: if 'A->B raw' >> 'oracle-affine' ~ nonzero, the residual after "
          "affine is the ONLY untransferred part.\nIf 'oracle-affine' is tiny, "
          "everything is affine = REFUTED shared-shape. If detrend xcorr ~0, "
          "signal is just monotone trend = trivial.")


if __name__ == "__main__":
    main()
