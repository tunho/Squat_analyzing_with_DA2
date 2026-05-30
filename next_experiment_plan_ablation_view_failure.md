# Next Experiment Plan: Ablation, View Models, and PM_038 Analysis

## Goal

Improve the paper evidence around the final residual-learning model while waiting for additional datasets. The focus is not only higher MAE improvement, but also explaining which features help, whether view-specific modeling is better, and why weak cases such as PM_038 underperform.

## 1. Feature Ablation Study

### Objective

Identify which feature groups actually improve subject-level LOSO generalization.

### Baseline

Use the current final model:

```text
world_shoulder_enhanced
Raw MAE: 12.291
Corrected MAE: 8.064
Improvement: 34.4%
```

### Ablation Groups

Run LOSO with these feature sets:

```text
A. angle_only
   mp_knee_angle

B. lower_body_current
   mp_knee_angle
   k_x, k_y, k_z
   a_x, a_y, a_z
   hip_ankle_dist_norm
   leg_ratio

C. + temporal_angle
   lower_body_current
   knee_lag1, knee_lag2
   angle_velocity, angle_acceleration

D. + coordinate_lag_delta
   temporal_angle
   k_x_lag1, k_y_lag1, k_z_lag1
   a_x_lag1, a_y_lag1, a_z_lag1
   k_dx, k_dy, k_dz
   a_dx, a_dy, a_dz

E. + segment_stability
   coordinate_lag_delta
   thigh_len, shank_len
   thigh_len_dev, shank_len_dev
   thigh_dir_change, shank_dir_change

F. + visibility
   segment_stability
   h_vis, k_vis, a_vis
   min_leg_vis, mean_leg_vis, visibility_drop

G. + view
   visibility
   view_is_side
```

### Outputs

```text
outputs/paper/ablation/<feature_set>/loso_metrics.csv
outputs/paper/ablation/summary.csv
outputs/paper/ablation/deep_flexion_summary.csv
```

### Decision Criteria

- Keep feature groups that reduce mean LOSO MAE.
- Mark groups that improve deep flexion even if total MAE changes little.
- Remove groups that increase variance or hurt multiple subjects.

## 2. View-Specific Model Comparison

### Objective

Test whether front and side views should share one model or use separate residual models.

### Compared Models

```text
A. Unified model
   Train one model using both front and side, with view_is_side.

B. Front-only model
   Train and test only front-view rows.

C. Side-only model
   Train and test only side-view rows.

D. Two-expert model
   Train front-only and side-only models separately, then combine their predictions.
```

### Metrics

Report:

```text
front MAE raw/corrected
side MAE raw/corrected
overall weighted MAE
subject-level LOSO MAE
deep flexion MAE by view
```

### Outputs

```text
outputs/paper/view_models/unified_metrics.csv
outputs/paper/view_models/front_only_metrics.csv
outputs/paper/view_models/side_only_metrics.csv
outputs/paper/view_models/two_expert_metrics.csv
outputs/paper/view_models/summary.csv
```

### Decision Criteria

- If two-expert improves meaningfully, use it as an extension or ablation result.
- If unified is similar, keep unified for simpler paper claims.
- If one view performs poorly, analyze that view separately in the limitations section.

## 3. PM_038 Failure Case Analysis

### Objective

Explain why PM_038 has weak correction performance relative to other subjects.

### Checks

Run the following diagnostics for PM_038:

```text
1. view-level MAE
   front raw/corrected
   side raw/corrected

2. phase/deep-flexion MAE
   gt_angle < 110
   gt_angle >= 110

3. visibility statistics
   h_vis, k_vis, a_vis
   min_leg_vis
   visibility_drop

4. signal stability
   thigh_len_dev
   shank_len_dev
   thigh_dir_change
   shank_dir_change

5. temporal alignment
   inspect GT vs MediaPipe angle curve
   check if peak/bottom timing is shifted
```

### Outputs

```text
outputs/paper/failure_pm038/metrics_by_view.csv
outputs/paper/failure_pm038/metrics_by_angle_region.csv
outputs/paper/failure_pm038/feature_stats_vs_others.csv
outputs/paper/failure_pm038/pm038_timeseries.csv
outputs/paper/failure_pm038/pm038_gt_raw_corrected.png
```

### Expected Paper Use

Use this analysis to write a limitation paragraph:

```text
The residual model improves most subjects, but correction is weaker when MediaPipe world landmarks show unstable segment geometry or when view-specific distortion patterns deviate from the training subjects.
```

## Execution Order

1. Add configurable feature sets to `paper_common.py`.
2. Add `scripts/paper_ablation.py`.
3. Add `scripts/paper_view_models.py`.
4. Add `scripts/paper_analyze_failure.py --subject PM_038`.
5. Run all scripts on `outputs/paper/features_ex6_world_shoulder.csv`.
6. Select final model and manuscript claims from the resulting summaries.

## Acceptance Criteria

- Ablation summary identifies useful and harmful feature groups.
- View-specific comparison determines whether unified or two-expert modeling is better.
- PM_038 analysis produces a defensible explanation for weak performance.
- All outputs are generated under `outputs/paper/`.
