# Final Residual Model Parameters

## Model Choice

The current paper-facing model is `world_shoulder_enhanced`.

This model uses a corrected MediaPipe world-coordinate feature pipeline and excludes the old mixed-coordinate shoulder features. It predicts the residual between the MediaPipe knee angle and the GT knee angle.

## Target

```text
residual = gt_angle - mp_knee_angle
```

## Prediction

```text
corrected_angle = mp_knee_angle + predicted_residual
```

## Final Input Parameters

```text
mp_knee_angle

leg_ratio

angle_velocity
angle_acceleration

k_x, k_y, k_z
a_x, a_y, a_z

h_vis
k_vis
a_vis
min_leg_vis
mean_leg_vis
visibility_drop

hip_depth_norm
hip_ankle_dist_norm

knee_lag1
knee_lag2

k_x_lag1, k_y_lag1, k_z_lag1
a_x_lag1, a_y_lag1, a_z_lag1

k_dx, k_dy, k_dz
a_dx, a_dy, a_dz

thigh_len
shank_len
thigh_len_dev
shank_len_dev

thigh_dir_change
shank_dir_change

view_is_side
```

## Excluded Parameters

```text
mp_hip_angle
s_x, s_y, s_z
s_vis
```

These are excluded from the final enhanced feature set because the earlier feature CSV mixed 2D shoulder coordinates with MediaPipe world hip/knee/ankle coordinates. The extraction code now supports `mp_world_shoulder`, but the final selected feature set remains lower-body focused for cleaner paper claims.

## Current LOSO Performance

```text
Raw MAE          12.291 degrees
Corrected MAE     8.064 degrees
Improvement       34.4%
```

## Deep Flexion Performance

For `gt_angle < 110`:

```text
Raw MAE          19.595 degrees
Corrected MAE     7.244 degrees
Improvement       63.0%
```

## Interpretation

The final model uses MediaPipe knee angle, current lower-body world coordinates, previous-frame angle/coordinate information, motion deltas, segment stability features, visibility features, and view information to learn the MediaPipe-to-GT residual.
