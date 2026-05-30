# World Shoulder Feature Recovery Plan

## Goal

Fix the feature pipeline so all pose features use the same MediaPipe world coordinate system, then rerun residual-learning experiments. This specifically addresses the current issue where `k_x/a_x` are world-coordinate features but `s_x/s_y/s_z` and `mp_hip_angle` were derived from 2D shoulder coordinates.

## 1. Re-Extract Features With `mp_world_shoulder`

### Objective

Regenerate the paper feature CSV using the corrected `SquatFrame.mp_world_shoulder` path.

### Required Code State

Already implemented:

- `SquatFrame.mp_world_shoulder`
- `build_squat_frame()` stores world shoulder from MediaPipe world landmarks
- `paper_extract.py` computes:
  - `s_x/s_y/s_z` from world shoulder
  - `mp_hip_angle` from world shoulder, world hip, world knee

### Command

```bash
PYTHONPATH=src .venv-paper/bin/python scripts/paper_extract.py \
  --dataset dataset/raw/REHAB24-6 \
  --exercise Ex6 \
  --output outputs/paper/features_ex6_world_shoulder.csv
```

### Validation

Check that shoulder features are numerically aligned with other normalized world features:

```bash
.venv-paper/bin/python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/paper/features_ex6_world_shoulder.csv")
print(df[["k_x","a_x","s_x","k_y","a_y","s_y","mp_hip_angle"]].describe())
PY
```

Expected:

- `s_x/s_y/s_z` should be near normalized world-coordinate scale, not 1000-pixel scale.
- `mp_hip_angle` should no longer depend on mixed 2D/world coordinates.

## 2. Retrain Residual Model With Corrected Shoulder Features

### Objective

Rerun the official v6 residual LOSO model using the corrected feature CSV. This answers whether the original v6 performance was genuine or partly caused by the accidental shoulder proxy.

### Command

```bash
.venv-paper/bin/python scripts/paper_train_loso.py \
  --features outputs/paper/features_ex6_world_shoulder.csv \
  --output outputs/paper/v6_world_shoulder_loso \
  --feature-set v6
```

### Outputs

- `outputs/paper/v6_world_shoulder_loso/loso_metrics.csv`
- `outputs/paper/v6_world_shoulder_loso/summary_metrics.json`
- `outputs/paper/v6_world_shoulder_loso/deep_flexion_metrics.csv`
- `outputs/paper/v6_world_shoulder_loso/v6_single_residual_model.pkl`

### Decision Rule

Compare against current official v6:

```text
Current v6: Raw 12.291 -> Corrected 7.675, improvement 37.6%
```

If corrected world-shoulder v6 stays close to or improves on this, use it as the official paper result. If it drops substantially, revise the feature design and manuscript claims before submission.

## 3. Test Enhanced Feature Set After Shoulder Fix

### Objective

Evaluate whether previous-frame coordinates, coordinate deltas, angle acceleration, segment lengths, direction changes, and view encoding improve residual learning once the shoulder feature bug is removed.

### Command

```bash
.venv-paper/bin/python scripts/paper_train_loso.py \
  --features outputs/paper/features_ex6_world_shoulder.csv \
  --output outputs/paper/v6_world_shoulder_enhanced_loso \
  --feature-set enhanced_safe
```

### Comparison Targets

Compare three models:

```text
A. old v6 on old CSV
B. v6 on world-shoulder CSV
C. enhanced_safe on world-shoulder CSV
```

Important metrics:

- subject-wise MAE/RMSE
- mean LOSO MAE/RMSE
- deep flexion `gt_angle < 110` MAE
- number of subjects improved vs raw
- number of subjects worse than old v6

### Expected Interpretation

- If B improves or matches A, the shoulder fix validates the original direction.
- If C improves over B, previous-frame and motion features are useful and should become the official model.
- If C is worse than B, keep the simpler v6 model and mention that temporal/kinematic feature expansion did not improve LOSO generalization.

## Acceptance Criteria

- Feature CSV regenerated with `mp_world_shoulder`.
- `s_x/s_y/s_z` scale verified as normalized world features.
- LOSO results generated for both `v6` and `enhanced_safe`.
- Final paper model selected using reproducible metrics.
- Manuscript tables updated only after this corrected experiment is complete.
