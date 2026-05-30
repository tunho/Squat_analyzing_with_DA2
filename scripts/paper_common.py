from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_COLS = [
    "mp_knee_angle",
    "mp_hip_angle",
    "leg_ratio",
    "angle_vel_smooth",
    "k_x",
    "k_y",
    "k_z",
    "a_x",
    "a_y",
    "a_z",
    "s_x",
    "s_y",
    "s_z",
    "h_vis",
    "k_vis",
    "a_vis",
    "s_vis",
    "hip_depth_norm",
    "hip_ankle_dist_norm",
    "knee_lag1",
    "knee_lag2",
]


ENHANCED_SAFE_FEATURE_COLS = [
    "mp_knee_angle",
    "leg_ratio",
    "angle_velocity",
    "angle_acceleration",
    "k_x",
    "k_y",
    "k_z",
    "a_x",
    "a_y",
    "a_z",
    "h_vis",
    "k_vis",
    "a_vis",
    "min_leg_vis",
    "mean_leg_vis",
    "visibility_drop",
    "hip_depth_norm",
    "hip_ankle_dist_norm",
    "knee_lag1",
    "knee_lag2",
    "k_x_lag1",
    "k_y_lag1",
    "k_z_lag1",
    "a_x_lag1",
    "a_y_lag1",
    "a_z_lag1",
    "k_dx",
    "k_dy",
    "k_dz",
    "a_dx",
    "a_dy",
    "a_dz",
    "thigh_len",
    "shank_len",
    "thigh_len_dev",
    "shank_len_dev",
    "thigh_dir_change",
    "shank_dir_change",
    "view_is_side",
]


FEATURE_SETS = {
    "v6": FEATURE_COLS,
    "enhanced_safe": ENHANCED_SAFE_FEATURE_COLS,
    "angle_only": [
        "mp_knee_angle",
    ],
    "lower_body_current": [
        "mp_knee_angle",
        "k_x",
        "k_y",
        "k_z",
        "a_x",
        "a_y",
        "a_z",
        "hip_ankle_dist_norm",
        "leg_ratio",
    ],
    "temporal_angle": [
        "mp_knee_angle",
        "k_x",
        "k_y",
        "k_z",
        "a_x",
        "a_y",
        "a_z",
        "hip_ankle_dist_norm",
        "leg_ratio",
        "knee_lag1",
        "knee_lag2",
        "angle_velocity",
        "angle_acceleration",
    ],
    "coordinate_lag_delta": [
        "mp_knee_angle",
        "k_x",
        "k_y",
        "k_z",
        "a_x",
        "a_y",
        "a_z",
        "hip_ankle_dist_norm",
        "leg_ratio",
        "knee_lag1",
        "knee_lag2",
        "angle_velocity",
        "angle_acceleration",
        "k_x_lag1",
        "k_y_lag1",
        "k_z_lag1",
        "a_x_lag1",
        "a_y_lag1",
        "a_z_lag1",
        "k_dx",
        "k_dy",
        "k_dz",
        "a_dx",
        "a_dy",
        "a_dz",
    ],
    "segment_stability": [
        "mp_knee_angle",
        "k_x",
        "k_y",
        "k_z",
        "a_x",
        "a_y",
        "a_z",
        "hip_ankle_dist_norm",
        "leg_ratio",
        "knee_lag1",
        "knee_lag2",
        "angle_velocity",
        "angle_acceleration",
        "k_x_lag1",
        "k_y_lag1",
        "k_z_lag1",
        "a_x_lag1",
        "a_y_lag1",
        "a_z_lag1",
        "k_dx",
        "k_dy",
        "k_dz",
        "a_dx",
        "a_dy",
        "a_dz",
        "thigh_len",
        "shank_len",
        "thigh_len_dev",
        "shank_len_dev",
        "thigh_dir_change",
        "shank_dir_change",
    ],
    "visibility": [
        "mp_knee_angle",
        "k_x",
        "k_y",
        "k_z",
        "a_x",
        "a_y",
        "a_z",
        "hip_ankle_dist_norm",
        "leg_ratio",
        "knee_lag1",
        "knee_lag2",
        "angle_velocity",
        "angle_acceleration",
        "k_x_lag1",
        "k_y_lag1",
        "k_z_lag1",
        "a_x_lag1",
        "a_y_lag1",
        "a_z_lag1",
        "k_dx",
        "k_dy",
        "k_dz",
        "a_dx",
        "a_dy",
        "a_dz",
        "thigh_len",
        "shank_len",
        "thigh_len_dev",
        "shank_len_dev",
        "thigh_dir_change",
        "shank_dir_change",
        "h_vis",
        "k_vis",
        "a_vis",
        "min_leg_vis",
        "mean_leg_vis",
        "visibility_drop",
    ],
    "view": ENHANCED_SAFE_FEATURE_COLS,
}


def mae(y_true: Any, y_pred: Any) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(true_arr - pred_arr)))


def rmse(y_true: Any, y_pred: Any) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))


def improvement_pct(raw_value: float, corrected_value: float) -> float:
    if raw_value == 0.0:
        return 0.0
    return float((raw_value - corrected_value) / raw_value * 100.0)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_model_bundle(path: Path, bundle: dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_model_bundle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected model bundle dict, got {type(bundle)!r}")
    return bundle
