"""
SUMediPose: does per-subject knee-angle residual (bias / |error|) correlate with
REAL demographics (age, sex, height, weight, BMI, skintone)?

C10 showed subject bias is unpredictable from *pose features* (thigh_len etc.).
Demographics are a NEW, different signal (esp. skintone -> pose-estimation fairness).
If a demographic explains the bias, it is a partial mechanism for the "unmeasured latent".
n = SUMe subjects (~26) at subject level -> enough for correlation.
"""
import numpy as np
import pandas as pd
from scipy import stats

EST_FILES = {
    "mediapipe": "experiments/paper/features_pooled4_offstd.csv",
    "nlf": "experiments/paper/features_pooled4_nlf.csv",
    "rtmw3d": "experiments/paper/features_pooled4_rtmw3d.csv",
    "motionbert": "experiments/paper/features_pooled4_motionbert.csv",
}
INFO = "dataset/raw/sumedipose_a3/subject_info.tab"


def is_sume(sid: str) -> bool:
    s = str(sid)
    # SUMe subjects are 'S1'..'S28'; exclude fit3d_s*, PM_, CA/CB
    return s.startswith("S") and not s.startswith(("SS",)) and s[1:].isdigit()


def load_demo():
    d = pd.read_csv(INFO, sep="\t")
    d["subject"] = d["subject"].str.strip('"')
    for c in ["sex", "work_intensity"]:
        if c in d:
            d[c] = d[c].str.strip('"')
    d["bmi"] = d["weight"] / (d["height"] / 100.0) ** 2
    d["sex_m"] = (d["sex"] == "Male").astype(int)
    return d


def per_subject_residual(path):
    df = pd.read_csv(path, usecols=["subject_id", "mp_knee_angle", "gt_angle"],
                     low_memory=False).dropna()
    df = df[df["subject_id"].map(is_sume)]
    if df.empty:
        return None
    df["residual"] = df["gt_angle"] - df["mp_knee_angle"]
    g = df.groupby("subject_id")["residual"]
    return pd.DataFrame({
        "subject": g.mean().index,
        "bias": g.mean().values,          # signed subject bias
        "abserr": g.apply(lambda x: x.abs().mean()).values,  # subject error magnitude
        "n": g.size().values,
    })


def corr_report(merged, targets, feats):
    for tgt in targets:
        print(f"\n  target = {tgt} (n={merged[tgt].notna().sum()} subjects)")
        rows = []
        for f in feats:
            m = merged[[tgt, f]].dropna()
            if len(m) < 6 or m[f].std() == 0:
                continue
            r, p = stats.spearmanr(m[f], m[tgt])
            rows.append((f, r, p, len(m)))
        rows.sort(key=lambda x: -abs(x[1]))
        for f, r, p, n in rows:
            flag = " *" if p < 0.05 else ""
            print(f"    {f:14s} Spearman r={r:+.2f}  p={p:.3f}  (n={n}){flag}")


def main():
    demo = load_demo()
    print(f"SUMe demographics loaded: {len(demo)} subjects")
    feats = ["age", "height", "weight", "bmi", "skintone_scale", "sex_m"]
    for est, path in EST_FILES.items():
        res = per_subject_residual(path)
        if res is None:
            print(f"\n=== {est}: no SUMe rows ===")
            continue
        merged = res.merge(demo, on="subject", how="inner")
        print(f"\n===== {est}  ({len(merged)} matched subjects, "
              f"bias range [{merged.bias.min():+.1f},{merged.bias.max():+.1f}], "
              f"mean|err| {merged.abserr.mean():.1f}) =====")
        corr_report(merged, ["bias", "abserr"], feats)


if __name__ == "__main__":
    main()
