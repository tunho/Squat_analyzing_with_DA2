# Squat Knee-Angle Correction for Monocular Pose Estimation

> Residual learning that corrects the systematic knee-angle error of MediaPipe's
> monocular 3D pose, validated across **four independent datasets** with
> Leave-One-Subject-Out (LOSO) and Leave-One-Dataset-Out (LODO) protocols.

🇰🇷 한국어 설명은 [README.ko.md](README.ko.md) 참고.

---

## TL;DR

MediaPipe's world-coordinate knee angle carries a **systematic, learnable error**
relative to 3D ground truth. We train an `ExtraTreesRegressor` to predict the
residual `gt_angle − mp_knee_angle` and apply it as a correction:

```
corrected_angle = mp_knee_angle + predicted_residual
```

Pooled across 4 datasets (84 subjects), this cuts knee-angle MAE from
**13.74° → 8.47° (+38.4%)**, and **+43.5%** in the clinically important deep-flexion
range (knee < 110°). The gain is consistent across an 80–130° threshold sweep
(Δ < 3 %p), so it is **robust, not cherry-picked**.

## Datasets

| Dataset | Role | GT source | Subjects |
|---|---|---|--:|
| **REHAB24-6** | primary | 3D marker joints (`.npy`) | 9 |
| **FiT3D** | external validation | 3D mocap | 3 |
| **AIHub CrossFit** | scale / domain | 3D labels | 46 |
| **SUMediPose** | gold standard | Vicon optical mocap | 26 |

All ground truth is 3D joint position; the model only ever sees monocular
MediaPipe output at inference, so the correction is camera-only at deploy time.

## Key results

### 1. In-domain LOSO — strong and consistent on every dataset

| Dataset | Raw MAE | Corrected MAE | Full | Deep (<110°) |
|---|--:|--:|--:|--:|
| AIHub | 14.31° | 7.97° | **+44.3%** | +39.3% |
| SUMediPose (Vicon) | 12.65° | 7.26° | **+42.6%** | +53.5% |
| REHAB24-6 | 12.29° | 8.11° | **+34.0%** | +61.9% |
| FiT3D | 20.25° | 11.94° | **+41.0%** | +62.1% |

### 2. Baseline comparison (pooled LOSO MAE °)

| Range | raw | savgol | affine | calib | linreg | **ours** |
|---|--:|--:|--:|--:|--:|--:|
| Full | 13.74 | 24.20 | 13.45 | 17.41 | 11.89 | **8.47** |
| Deep <110° | 19.42 | 39.20 | 18.92 | 29.73 | 15.55 | **10.97** |

The advantage over a linear residual baseline *widens* in the deep range
(+43.5% vs +19.9% over raw — a 2.2× margin).

### 3. Cross-dataset generalization — an honest boundary

Leave-One-Dataset-Out (train on 3 datasets, test the held-out 4th, zero-shot):

| Held-out | Per-subject improvement | 95% CI |
|---|--:|--:|
| AIHub | −11.4% | [−14.0, −8.8] |
| SUMediPose | +21.5% | [+18.3, +24.7] |
| REHAB | +2.5% | [−4.8, +10.5] |
| FiT3D | +25.2% | [+16.4, +33.1] |
| **Overall** | **+1.6%** | **[−2.2, +5.7]** |

Zero-shot transfer is **not** statistically reliable overall (CI includes 0). We
report this explicitly rather than overclaiming. However, the deep-flexion range
generalizes partially in 3 of 4 domains (AIHub +28%, FiT3D +35%, SUME +17%).

### 4. Few-shot recovery — 1 target subject fixes a failing domain

Adding just **k=1** subject from the target domain to training (test on the rest):

| Held-out | k=0 (zero-shot) | k=1 | k=2 | standalone LOSO |
|---|--:|--:|--:|--:|
| AIHub | −11.1 | **+13.0** | +20.3 | +44.3 |
| REHAB | +2.1 | **+15.9** | +19.9 | +34.0 |
| SUMediPose | +21.3 | +23.9 | +26.7 | +42.6 |
| FiT3D | +23.8 | +29.7 | +34.6 | +41.0 |

Performance saturates around k≈2 — quantifying the *minimal target supervision*
needed to adapt to a new domain.

## Method

```
Video (.mp4) + 3D GT joints (.npy)
        │  scripts/paper_extract*.py   (MediaPipe world coords, hip-centred, length-normalised)
        ▼
Feature CSV  [mp_knee_angle, gt_angle, k_x/y/z, a_x/y/z, vis, view_type, ...]
        │  scripts/paper_train_loso.py  (+temporal/derived features → ExtraTrees, LOSO)
        ▼
Model bundle (.pkl)
        │  scripts/paper_evaluate.py
        ▼
predictions.csv  [mp_knee_angle, corrected_angle, gt_angle, predicted_residual]
```

The default feature set (`enhanced_safe`, 39 features) combines current/lagged
hip-relative coordinates, joint visibility, segment-length stability and angular
velocity. A real-time-safe variant (`enhanced_causal`, 34 features) drops
offline-only normalisation for streaming use.

## Repository layout

```
src/squat/            Core library (pose pipeline, geometry, analyzers, rep counter)
scripts/              Paper pipeline: extraction, LOSO/LODO training, few-shot, baselines, figures
experiments/paper/    Aggregated metrics & summaries (large data CSVs are gitignored, regenerable)
experiments/legacy/   Earlier experimental scripts (kept for provenance)
docs/                 Research notes, paper drafts, analysis plans (Korean)
```

> Large raw dumps, per-frame `predictions.csv`, derived feature CSVs and `.pkl`
> models are **not committed** (size + dataset licensing). They are fully
> reproducible from the extraction scripts.

## Setup

```bash
# Core runtime (MediaPipe, OpenCV, NumPy)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full paper pipeline (+ pandas, scikit-learn, matplotlib)
python -m venv .venv-paper && source .venv-paper/bin/activate
pip install -r requirements-paper.txt
```

### Example: extract → train → evaluate

```bash
cd scripts
PYTHONPATH=../src python paper_extract.py \
  --dataset ../dataset/raw/REHAB24-6 --exercise Ex6 \
  --output ../experiments/paper/features_ex6.csv

python paper_train_loso.py \
  --features ../experiments/paper/features_ex6.csv \
  --output ../experiments/paper/loso_out --feature-set v6

python paper_evaluate.py \
  --features ../experiments/paper/features_ex6.csv \
  --model ../experiments/paper/loso_out/v6_single_residual_model.pkl \
  --output ../experiments/paper/eval_out
```

## License & data

Code is research-oriented. Datasets (REHAB24-6, FiT3D, AIHub CrossFit,
SUMediPose) are governed by their respective licenses and are **not** included
here — obtain them from their original sources.
