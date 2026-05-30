# Paper Code Cleanup Plan

## Goal

Finalize the repository around the v6 single residual model for the paper. The official claim is: MediaPipe knee-angle error is corrected by learning the residual `gt_angle - mp_knee_angle` with subject-level LOSO validation.

## 1. SE-Oriented Code Cleanup

### 1.1 Freeze the Official Paper Path

Keep one canonical pipeline:

- Feature extraction: `src/extract_squat_features.py`
- Training and LOSO: `src/train_squat_corrector_v6_single.py`
- External/task checks: `src/evaluate_v6_on_ex4.py`, `src/evaluate_v6_on_ex5.py`
- Primary result: `outputs/ml_correction/v6_single_final/loso_single_v6.csv`

Do not delete legacy experiments immediately. Move them to `experiments/legacy/` after confirming they are not imported by the official path.

### 1.2 Proposed Repository Layout

```text
src/squat/                    # reusable package code
scripts/paper_extract.py       # official feature extraction entrypoint
scripts/paper_train_loso.py    # official v6 residual LOSO training
scripts/paper_evaluate.py      # official raw vs corrected evaluation
scripts/paper_make_tables.py   # tables/figures for the manuscript
experiments/legacy/            # v3/v4/v5/v7/v13/v15 exploratory scripts
outputs/paper/                 # regenerated paper results only
outputs/archive/               # old exploratory outputs, optional
```

### 1.3 Refactoring Rules

- Preserve `src/squat/` as shared library code.
- Remove hard-coded subject lists from official scripts; read subject IDs from dataset files.
- Replace script-local constants with CLI arguments: dataset root, exercise, view, output directory.
- Keep the v6 target explicit: `residual = gt_angle - mp_knee_angle`.
- Save model bundles with `model`, `scaler`, `features`, `target`, `version`, and `created_from`.
- Add a `requirements.txt` update or `requirements-paper.txt` containing all real dependencies.

### 1.4 Must-Fix Engineering Issues

- Make the official pipeline runnable from a clean checkout.
- Avoid importing unused heavy dependencies during paper training.
- Ensure frame alignment is identical for raw and corrected comparisons.
- Make all generated result paths deterministic.
- Add a small smoke test for feature schema and residual training.

## 2. Paper Reproduction Plan

### 2.1 Official Experiment Definition

Use v6 single residual learning as the proposed method:

```text
Input features: MediaPipe world knee angle, hip angle, normalized joints,
visibility, hip depth proxy, hip-ankle distance, velocity, lag features.
Target: gt_angle - mp_knee_angle.
Prediction: corrected_angle = mp_knee_angle + predicted_residual.
Validation: leave-one-subject-out cross validation.
```

### 2.2 Reproducible Commands

Target commands after cleanup:

```bash
PYTHONPATH=src python scripts/paper_extract.py --dataset dataset/raw/REHAB24-6 --exercise Ex6 --output outputs/paper/features_ex6.csv
PYTHONPATH=src python scripts/paper_train_loso.py --features outputs/paper/features_ex6.csv --output outputs/paper/v6_loso
PYTHONPATH=src python scripts/paper_make_tables.py --loso outputs/paper/v6_loso/loso_metrics.csv --output outputs/paper/tables
```

### 2.3 Required Paper Outputs

- `features_ex6.csv`: extracted feature table with `subject_id`, `view_type`, `frame_index`, `mp_knee_angle`, `gt_angle`, and all v6 features.
- `loso_metrics.csv`: per-subject Raw MAE/RMSE and Corrected MAE/RMSE.
- `summary_metrics.json`: mean, std, improvement percentage, and sample count.
- `deep_flexion_metrics.csv`: same metrics for `gt_angle < 110`.
- Figures:
  - Raw vs corrected MAE bar plot.
  - GT vs Raw vs Corrected time-series example.
  - GT/predicted scatter plot.
  - Residual distribution plot.

### 2.4 Current Baseline to Reconcile

Currently reproducible result file:

```text
outputs/ml_correction/v6_single_final/loso_single_v6.csv
Raw MAE average: 12.291 degrees
Corrected MAE average: 7.672 degrees
Improvement: 37.6%
Subjects: 9
```

The manuscript draft reports `13.22 -> 7.57` and `42.7%`. Before submission, either regenerate results that exactly match the draft or revise the manuscript to the reproducible v6 numbers.

## 3. Execution Order

1. Create `scripts/` and copy the v6 logic into paper-named entrypoints.
2. Add CLI arguments and remove hard-coded paths where possible.
3. Generate `outputs/paper/` from scratch.
4. Compare regenerated numbers against existing `v6_single_final`.
5. Archive unrelated experiment scripts and outputs.
6. Update manuscript tables and claims to match regenerated outputs.
7. Add smoke tests and dependency documentation.

## 4. Acceptance Criteria

- A new user can run the three official commands and regenerate the main table.
- The paper result files contain only v6 residual-learning outputs.
- The manuscript numbers match generated files exactly.
- Legacy scripts are clearly separated from official paper code.
- No direct GT angle regression model is described as residual learning.
