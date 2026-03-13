from __future__ import annotations

from squat.calibration.relative_depth_calibration import (
    RelativeDepthCalibrationParams,
    fit_relative_depth_calibration,
    predict_relative_dz,
)
from squat.domain.types import Point3D, SquatFrame
from squat.geometry.angles import calculate_knee_angle


def fit_relative_linear_from_history(
    history_data: list[SquatFrame],
) -> RelativeDepthCalibrationParams | None:
    if not history_data:
        return None

    raw_dz_thigh_list = [frame.hip.z - frame.knee.z for frame in history_data]
    thigh_len_2d_list = [frame.thigh_len_2d for frame in history_data]
    raw_dz_shank_list = [frame.ankle.z - frame.knee.z for frame in history_data]
    shank_len_2d_list = [frame.shank_len_2d for frame in history_data]

    return fit_relative_depth_calibration(
        raw_dz_thigh_list=raw_dz_thigh_list,
        thigh_len_2d_list=thigh_len_2d_list,
        raw_dz_shank_list=raw_dz_shank_list,
        shank_len_2d_list=shank_len_2d_list,
    )


def build_relative_points(
    frame: SquatFrame,
    params: RelativeDepthCalibrationParams,
) -> tuple[Point3D, Point3D, Point3D]:
    raw_dz_thigh = frame.hip.z - frame.knee.z
    raw_dz_shank = frame.ankle.z - frame.knee.z

    thigh_dz = predict_relative_dz(
        raw_dz=raw_dz_thigh,
        a=params.thigh_a,
        b=params.thigh_b,
    )
    shank_dz = predict_relative_dz(
        raw_dz=raw_dz_shank,
        a=params.shank_a,
        b=params.shank_b,
    )

    knee = Point3D(frame.knee.x, frame.knee.y, 0.0)
    hip = Point3D(frame.hip.x, frame.hip.y, thigh_dz)
    ankle = Point3D(frame.ankle.x, frame.ankle.y, shank_dz)
    return hip, knee, ankle


def compute_relative_linear_angles(
    history_data: list[SquatFrame],
    params: RelativeDepthCalibrationParams,
) -> list[float]:
    final_angles: list[float] = []
    for frame in history_data:
        hip, knee, ankle = build_relative_points(frame, params)
        final_angles.append(calculate_knee_angle(hip, knee, ankle))
    return final_angles