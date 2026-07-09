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

We benchmark **9 correctors** (classic smoothers → 2025 tabular deep learning:
SavGol, Kalman, MLP, ExtraTrees, SmoothNet, TCN, Diffusion/CARD, ExcelFormer,
TabM) × **4 pose estimators** × **4 datasets** under 3 protocols (within-domain,
pooled, leave-one-dataset-out) + few-shot recovery. **ExtraTrees is the most
robust** everywhere; cross-domain transfer fails for all methods (a depth-shift
limit). All learning runs are **seed-fixed (42) and reproduced bit-exact** on the
same GPU — see [REPRODUCE.md](REPRODUCE.md). Clinically, correction reaches
**ICC 0.97** (agreement "excellent"), doubles the ±5° pass rate (33% → 53%), and
removes the −13° bias.

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

### 2. Corrector benchmark — 9 correctors × 3 protocols (improvement %, seed 42)

Every corrector is evaluated under identical splits across all 4 datasets and 4
pose estimators (MediaPipe, NLF, RTMW3D, MotionBERT). Values are mean improvement
% over the raw estimator (negative = worse).

| Corrector (year) | ① within | ② pooled (5-fold) | ③ LODO (cross-domain) |
|---|--:|--:|--:|
| **ExtraTrees** (trees) | **+39.2** | +49.5 | **+3.6** |
| ExcelFormer (KDD'24) | +34.7 | +46.6 | −0.1 |
| SmoothNet (ECCV'22) | +26.7 | +7.6 | −0.3 |
| TCN (Bai'18) | +25.5 | +7.7 | −1.3 |
| Diffusion / CARD (NeurIPS'22) | +18.4 | +45.3 | −11.1 |
| TabM (ICLR'25) | +15.6 | **+51.6** | +1.6 |
| MLP (sklearn) | −5.2 | +42.9 | −18.4 |
| SavGol (smoother) | −7.4 | −65.3 | N/A |
| Kalman (smoother) | −25.4 | −73.5 | N/A |

**Takeaways.** (i) In-domain & pooled work well; **ExtraTrees is the most robust**
— wins small data, ties the 2025 SOTA (TabM) at scale, and is best cross-domain.
(ii) Windowed smoothers (SmoothNet/TCN) collapse under pooled mixing; classic
smoothers (SavGol/Kalman) hurt. (iii) **Cross-domain (LODO) is a structural
failure for all correctors** (best only +3.6%) — a depth covariate-shift limit,
not a corrector flaw. (iv) The gain scales with raw error: correction helps most
where the estimator is worst (r = 0.64, p ≪ 0.001).

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

The paper default feature set (`v6`, 21 features) combines current/lagged
hip-relative coordinates, joint visibility, segment-length stability and angular
velocity. Extended (`enhanced_safe`, 37) and real-time-safe (`enhanced_causal`,
34, drops offline-only normalisation for streaming) variants are also available.

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

Full benchmark (9 correctors × 4 estimators × 4 datasets × 3 protocols),
figures and statistics: see **[REPRODUCE.md](REPRODUCE.md)**.

## Reproducibility

- All learning code fixes **seed 42** via `scripts/seed_util.set_all_seeds()`
  (random / numpy / torch / CUDA + `cudnn.deterministic=True`).
- Verified **bit-exact** on re-run for the same GPU (incl. neural correctors like
  TabM). sklearn (ExtraTrees, MLP) and smoothers are hardware-independent.
- **Caveat:** GPU neural correctors may differ slightly across GPU/cuDNN versions
  (V100 vs 4070Ti confirmed); reported numbers use a **4070Ti**.
- What is / isn't seeded and where each result lives:
  `experiments/paper/재현성_재실행_대장.md`.

## License & data

Code is research-oriented. Datasets (REHAB24-6, FiT3D, AIHub CrossFit,
SUMediPose) are governed by their respective licenses and are **not** included
here — obtain them from their original sources.
