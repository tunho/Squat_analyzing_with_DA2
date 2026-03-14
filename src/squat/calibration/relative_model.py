from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from squat.domain.types import Point3D, SquatFrame
from squat.geometry.angles import calculate_knee_angle


class RelativeCalibrationParams(Protocol):
    ref_thigh_len: float
    ref_shank_len: float


class RelativeDepthModel(Protocol):
    name: str

    def fit(self, history_data: list[SquatFrame]) -> RelativeCalibrationParams | None:
        ...

    def build_points(
        self,
        frame: SquatFrame,
        params: RelativeCalibrationParams,
    ) -> tuple[Point3D, Point3D, Point3D]:
        ...


@dataclass(slots=True)
class RelativeModelResult:
    params: RelativeCalibrationParams
    final_angles: list[float]


def extract_history_features(
    history_data: list[SquatFrame],
) -> tuple[list[float], list[float], list[float], list[float]]:
    raw_dz_thigh_list = [frame.hip.z - frame.knee.z for frame in history_data]
    thigh_len_2d_list = [frame.thigh_len_2d for frame in history_data]
    raw_dz_shank_list = [frame.ankle.z - frame.knee.z for frame in history_data]
    shank_len_2d_list = [frame.shank_len_2d for frame in history_data]
    return raw_dz_thigh_list, thigh_len_2d_list, raw_dz_shank_list, shank_len_2d_list


def compute_relative_angles(
    history_data: list[SquatFrame],
    model: RelativeDepthModel,
    params: RelativeCalibrationParams,
) -> list[float]:
    final_angles: list[float] = []
    for frame in history_data:
        hip, knee, ankle = model.build_points(frame, params)
        final_angles.append(calculate_knee_angle(hip, knee, ankle))
    return final_angles