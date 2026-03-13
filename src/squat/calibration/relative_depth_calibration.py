from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class RelativeDepthCalibrationParams:
    thigh_a: float
    thigh_b: float
    shank_a: float
    shank_b: float
    ref_thigh_len: float
    ref_shank_len: float


def _max_reference(values: list[float]) -> float:
    if not values:
        return 0.0

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return 0.0

    return float(np.max(arr))


def _signed_target_dz(
    raw_dz_list: list[float],
    len_2d_list: list[float],
    ref_len: float,
) -> np.ndarray:
    targets = []

    for raw_dz, len_2d in zip(raw_dz_list, len_2d_list):
        z_mag = np.sqrt(max(0.0, ref_len ** 2 - len_2d ** 2))
        sign = 1.0 if raw_dz >= 0.0 else -1.0
        targets.append(sign * z_mag)

    return np.asarray(targets, dtype=float)


def _fit_linear(x_list: list[float], y_list: list[float]) -> tuple[float, float]:
    if len(x_list) < 2 or len(y_list) < 2:
        return 1.0, 0.0

    x = np.asarray(x_list, dtype=float)
    y = np.asarray(y_list, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 2:
        return 1.0, 0.0

    mat = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)

    return float(coef[0]), float(coef[1])


def fit_relative_depth_calibration(
    raw_dz_thigh_list: list[float],
    thigh_len_2d_list: list[float],
    raw_dz_shank_list: list[float],
    shank_len_2d_list: list[float],
) -> RelativeDepthCalibrationParams:
    ref_thigh_len = _max_reference(thigh_len_2d_list)
    ref_shank_len = _max_reference(shank_len_2d_list)

    thigh_target = _signed_target_dz(
        raw_dz_list=raw_dz_thigh_list,
        len_2d_list=thigh_len_2d_list,
        ref_len=ref_thigh_len,
    )
    shank_target = _signed_target_dz(
        raw_dz_list=raw_dz_shank_list,
        len_2d_list=shank_len_2d_list,
        ref_len=ref_shank_len,
    )

    thigh_a, thigh_b = _fit_linear(raw_dz_thigh_list, thigh_target.tolist())
    shank_a, shank_b = _fit_linear(raw_dz_shank_list, shank_target.tolist())

    return RelativeDepthCalibrationParams(
        thigh_a=thigh_a,
        thigh_b=thigh_b,
        shank_a=shank_a,
        shank_b=shank_b,
        ref_thigh_len=ref_thigh_len,
        ref_shank_len=ref_shank_len,
    )


def predict_relative_dz(raw_dz: float, a: float, b: float) -> float:
    return a * raw_dz + b