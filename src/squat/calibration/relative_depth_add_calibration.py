from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class RelativeDepthAddCalibrationParams:
    thigh_a: float
    thigh_c: float
    thigh_b: float
    shank_a: float
    shank_c: float
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


def _fit_additive_model(
    raw_dz_list: list[float],
    len_2d_list: list[float],
    target_dz_list: list[float],
    ref_len: float,
) -> tuple[float, float, float]:
    if len(raw_dz_list) < 2 or len(len_2d_list) < 2 or len(target_dz_list) < 2:
        return 1.0, 0.0, 0.0

    raw = np.asarray(raw_dz_list, dtype=float)
    len_2d = np.asarray(len_2d_list, dtype=float)
    target = np.asarray(target_dz_list, dtype=float)

    delta_len = ref_len - len_2d

    valid = np.isfinite(raw) & np.isfinite(delta_len) & np.isfinite(target)
    raw = raw[valid]
    delta_len = delta_len[valid]
    target = target[valid]

    if raw.size < 2:
        return 1.0, 0.0, 0.0

    mat = np.column_stack([raw, delta_len, np.ones_like(raw)])
    coef, _, _, _ = np.linalg.lstsq(mat, target, rcond=None)

    a, c, b = coef
    return float(a), float(c), float(b)


def fit_relative_depth_add_calibration(
    raw_dz_thigh_list: list[float],
    thigh_len_2d_list: list[float],
    raw_dz_shank_list: list[float],
    shank_len_2d_list: list[float],
) -> RelativeDepthAddCalibrationParams:
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

    thigh_a, thigh_c, thigh_b = _fit_additive_model(
        raw_dz_list=raw_dz_thigh_list,
        len_2d_list=thigh_len_2d_list,
        target_dz_list=thigh_target.tolist(),
        ref_len=ref_thigh_len,
    )
    shank_a, shank_c, shank_b = _fit_additive_model(
        raw_dz_list=raw_dz_shank_list,
        len_2d_list=shank_len_2d_list,
        target_dz_list=shank_target.tolist(),
        ref_len=ref_shank_len,
    )

    return RelativeDepthAddCalibrationParams(
        thigh_a=thigh_a,
        thigh_c=thigh_c,
        thigh_b=thigh_b,
        shank_a=shank_a,
        shank_c=shank_c,
        shank_b=shank_b,
        ref_thigh_len=ref_thigh_len,
        ref_shank_len=ref_shank_len,
    )


def predict_relative_add_dz(
    raw_dz: float,
    len_2d: float,
    ref_len: float,
    a: float,
    c: float,
    b: float,
) -> float:
    return a * raw_dz + c * (ref_len - len_2d) + b