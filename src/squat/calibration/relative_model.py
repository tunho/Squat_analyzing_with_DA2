from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from squat.domain.types import Point3D, SquatFrame
from squat.geometry.angles import calculate_knee_angle


class RelativeCalibrationParams(Protocol):
    pass


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


def get_frame_raw_dz(
    frame: SquatFrame,
    z_source: str = "depth",
) -> tuple[float, float]:
    if z_source == "depth":
        return frame.hip.z - frame.knee.z, frame.ankle.z - frame.knee.z

    if z_source == "mp_world":
        if (
            frame.mp_world_hip is None
            or frame.mp_world_knee is None
            or frame.mp_world_ankle is None
        ):
            raise ValueError("Missing MediaPipe world landmarks for this frame.")
        return (
            frame.mp_world_hip.z - frame.mp_world_knee.z,
            frame.mp_world_ankle.z - frame.mp_world_knee.z,
        )

    raise ValueError(f"Unsupported z_source={z_source!r}.")


def extract_history_features(
    history_data: list[SquatFrame],
    z_source: str = "depth",
) -> tuple[list[float], list[float], list[float], list[float], list[SquatFrame]]:
    raw_dz_thigh_list: list[float] = []
    thigh_len_2d_list: list[float] = []
    raw_dz_shank_list: list[float] = []
    shank_len_2d_list: list[float] = []
    valid_frames: list[SquatFrame] = []

    for frame in history_data:
        try:
            raw_dz_thigh, raw_dz_shank = get_frame_raw_dz(frame, z_source=z_source)
        except ValueError:
            continue

        raw_dz_thigh_list.append(raw_dz_thigh)
        thigh_len_2d_list.append(frame.thigh_len_2d)
        raw_dz_shank_list.append(raw_dz_shank)
        shank_len_2d_list.append(frame.shank_len_2d)
        valid_frames.append(frame)

    return (
        raw_dz_thigh_list,
        thigh_len_2d_list,
        raw_dz_shank_list,
        shank_len_2d_list,
        valid_frames,
    )


def compute_relative_angles(
    history_data: list[SquatFrame],
    model: RelativeDepthModel,
    params: RelativeCalibrationParams,
) -> list[float]:
    final_angles: list[float] = []
    for frame in history_data:
        try:
            hip, knee, ankle = model.build_points(frame, params)
        except ValueError:
            continue
        final_angles.append(calculate_knee_angle(hip, knee, ankle))
    return final_angles