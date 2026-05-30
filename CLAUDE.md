# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for a paper on **squat knee-angle estimation improvement using MediaPipe**. The core idea: MediaPipe's world-coordinate knee angle predictions contain systematic error (residual) relative to ground-truth 3D joint data. An ExtraTreesRegressor is trained to predict `gt_angle - mp_knee_angle` (the residual), then applied as a correction: `corrected_angle = mp_knee_angle + predicted_residual`. Evaluation uses **Leave-One-Subject-Out (LOSO)** cross-validation.

**Datasets**: REHAB24-6 (primary, squat videos with 3D GT joints as `.npy`), FiT3D (external validation), NTU RGB+D (supplementary).

## Environment Setup

```bash
# Core runtime (MediaPipe, OpenCV, NumPy only)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full paper pipeline (adds pandas, scikit-learn, matplotlib)
python -m venv .venv-paper && source .venv-paper/bin/activate
pip install -r requirements-paper.txt
```

All `src/squat` imports require `PYTHONPATH=src`. All `scripts/` imports (e.g. `paper_common`) require running from `scripts/` or adding it to `PYTHONPATH`.

## Key Commands

### Syntax check
```bash
PYTHONPATH=src python -m compileall src -q
```

### Feature extraction (video → CSV)
```bash
cd scripts
PYTHONPATH=../src python paper_extract.py \
  --dataset ../dataset/raw/REHAB24-6 \
  --exercise Ex6 \
  --output ../experiments/paper/features_ex6.csv
```

### LOSO training
```bash
cd scripts
python paper_train_loso.py \
  --features ../experiments/paper/features_ex6.csv \
  --output ../experiments/paper/loso_out \
  --feature-set v6
```

### Evaluate a saved model
```bash
cd scripts
python paper_evaluate.py \
  --features ../experiments/paper/features_ex6.csv \
  --model ../experiments/paper/loso_out/v6_single_residual_model.pkl \
  --output ../experiments/paper/eval_out
```

### Run analyzer on a single video
```bash
PYTHONPATH=src python -c "
from squat.analyzer import MPWorldRawSquatAnalyzer
a = MPWorldRawSquatAnalyzer()
count, angles, _ = a.analyze_video('path/to/video.mp4', max_frames=300)
print(count, len(angles))
"
```

## Architecture

### `src/squat/` — Core Library

```
domain/types.py      SquatFrame, Point3D dataclasses
config.py            SQUAT_THRESHOLDS, MODEL_CONFIG, FILTER_CONFIG
geometry/
  angles.py          calculate_knee_angle(hip, knee, ankle) → float
  world_knee_ops.py  numpy helpers, EMA smoother, segment-length ops
pipeline/
  frame_builder.py   build_squat_frame() — assembles SquatFrame from landmarks + optional world_landmarks
  frame_processor.py process_frame() — runs pose estimator, extracts landmarks per video frame
models/
  pose_landmarker.py MediaPipe Pose wrapper (PoseEstimator)
  depth_anything.py  DepthAnythingV2 depth estimator (used only when depth features needed)
state/squat_counter.py SquatCounter — angle-threshold state machine for rep counting
analyzer/
  base.py            BaseSquatAnalyzer — video I/O loop, calls compute_frame_angle() per frame
  knee_base_world.py BaseWorldKneeAnalyzer — uses mp_world_* coords; overrideable transform_world_triplet()
  knee_mp_world_*.py concrete analyzer variants (raw, smooth, len_stable, visibility_filtered, outlier_corrected, selective_repair, burst_repair, hinge_repair, descent_repair)
io/                  video writer helpers, result serialization
visualization/       dashboard overlay, multiview drawing
```

**Analyzer class hierarchy**:
`BaseSquatAnalyzer` → `BaseWorldKneeAnalyzer` → `MPWorldRawSquatAnalyzer` (and all siblings)

Each variant overrides either `transform_world_triplet()` (to repair/stabilize hip/knee/ankle Point3D before angle computation) or `build_final_angle_sequence()` (to post-process the angle list).

### `scripts/` — Paper Pipeline

| Script | Purpose |
|---|---|
| `paper_extract.py` | Video → feature CSV. Extracts MediaPipe world coords + GT angles, normalizes coordinates by median segment length, computes `view_type` from camera name. |
| `paper_common.py` | Shared: `FEATURE_SETS` dict, `mae/rmse/improvement_pct`, `save_model_bundle/load_model_bundle` (pickle). |
| `paper_train_loso.py` | `prepare_v6_data()` adds temporal/derived features then trains ExtraTreesRegressor via LOSO. Saves `loso_metrics.csv`, `summary_metrics.json`, and final model `.pkl`. |
| `paper_evaluate.py` | Load a `.pkl` model bundle, run on new feature CSV, write `predictions.csv` + `metrics.csv`. |
| `paper_ablation.py` | Ablation over `FEATURE_SETS` keys. |
| `paper_make_tables.py` | Formats experiment JSONs/CSVs into LaTeX tables. |
| `generate_paper_plot.py` | Produces figures for the paper. |
| `paper_analyze_failure.py` | Failure-case analysis (view-specific, deep-flexion, etc.). |

### Data Flow

```
Video (.mp4) + GT joints (.npy)
        ↓  paper_extract.py
Feature CSV  [frame_index, subject_id, view_type, mp_knee_angle, gt_angle, k_x/y/z, a_x/y/z, vis, ...]
        ↓  paper_train_loso.py  (prepare_v6_data adds: angle_velocity, lag1/2, thigh_len, ...)
Trained model bundle (.pkl)  →  paper_evaluate.py
        ↓
predictions.csv  [mp_knee_angle, corrected_angle, gt_angle, predicted_residual]
```

### Feature Sets (`paper_common.FEATURE_SETS`)

- `v6` — default for the paper (21 features incl. shoulder, depth-norm, lag)
- `enhanced_safe` — extended set (37 features, no raw shoulder, adds direction-change)
- Ablation sets: `angle_only`, `lower_body_current`, `temporal_angle`, `coordinate_lag_delta`, `segment_stability`, `visibility`, `view`

### MediaPipe Landmark Indices

Left leg: hip=23, knee=25, ankle=27, shoulder=11  
Right leg: hip=24, knee=26, ankle=28, shoulder=12  
(Defined in `pipeline/frame_builder.py:SIDE_LANDMARKS`)

### GT Data Format

REHAB24-6 ground-truth: `.npy` arrays of shape `(frames, num_joints, 3)`.  
Default left-leg joint indices: hip=16, knee=17, ankle=18.  
Right-leg indices: hip=21, knee=22, ankle=23 (from `Segmentation.csv`).

## Conventions

- Coordinates in `SquatFrame` use two spaces: 2D pixel (`hip`, `knee`, `ankle`) and MediaPipe world 3D (`mp_world_hip`, etc., in meters, hip-centered).
- Feature coordinates (`k_x/y/z`, `a_x/y/z`) are hip-relative and normalized by `median(thigh_len + shank_len)`.
- Model bundles are plain `dict` pickles with keys: `model`, `scaler`, `features`, `target`, `method`, `feature_set`.
- `view_type` is determined from video filename: `"Camera18"` → `"side"`, else `"front"`.
- `hip_depth_norm` = normalized vertical position of hip in its temporal range (proxy for squat depth from front view).
