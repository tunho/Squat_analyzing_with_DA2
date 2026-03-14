from __future__ import annotations

from dataclasses import dataclass

from squat.calibration.relative_depth_add_calibration import (
    RelativeDepthAddCalibrationParams,
    fit_relative_depth_add_calibration,
    predict_relative_add_dz,
)
from squat.calibration.relative_depth_calibration import (
    RelativeDepthCalibrationParams,
    fit_relative_depth_calibration,
    predict_relative_dz,
)
from squat.calibration.relative_model import (
    RelativeCalibrationParams,
    RelativeDepthModel,
    compute_relative_angles,
    extract_history_features,
    get_frame_raw_dz,
)
from squat.domain.types import Point3D, SquatFrame


@dataclass(slots=True)
class RelativeLinearModel(RelativeDepthModel):
    name: str = "relative_linear"
    z_source: str = "depth"

    def fit(self, history_data: list[SquatFrame]) -> RelativeDepthCalibrationParams | None:
        if not history_data:
            return None
        (
            raw_dz_thigh_list,
            thigh_len_2d_list,
            raw_dz_shank_list,
            shank_len_2d_list,
            valid_frames,
        ) = extract_history_features(history_data, z_source=self.z_source)
        if not valid_frames:
            return None
        return fit_relative_depth_calibration(
            raw_dz_thigh_list=raw_dz_thigh_list,
            thigh_len_2d_list=thigh_len_2d_list,
            raw_dz_shank_list=raw_dz_shank_list,
            shank_len_2d_list=shank_len_2d_list,
        )

    def build_points(
        self,
        frame: SquatFrame,
        params: RelativeCalibrationParams,
    ) -> tuple[Point3D, Point3D, Point3D]:
        if not isinstance(params, RelativeDepthCalibrationParams):
            raise TypeError("RelativeLinearModel requires RelativeDepthCalibrationParams.")

        raw_dz_thigh, raw_dz_shank = get_frame_raw_dz(frame, z_source=self.z_source)

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


@dataclass(slots=True)
class RelativeAddModel(RelativeDepthModel):
    name: str = "relative_add"
    z_source: str = "depth"

    def fit(self, history_data: list[SquatFrame]) -> RelativeDepthAddCalibrationParams | None:
        if not history_data:
            return None
        (
            raw_dz_thigh_list,
            thigh_len_2d_list,
            raw_dz_shank_list,
            shank_len_2d_list,
            valid_frames,
        ) = extract_history_features(history_data, z_source=self.z_source)
        if not valid_frames:
            return None
        return fit_relative_depth_add_calibration(
            raw_dz_thigh_list=raw_dz_thigh_list,
            thigh_len_2d_list=thigh_len_2d_list,
            raw_dz_shank_list=raw_dz_shank_list,
            shank_len_2d_list=shank_len_2d_list,
        )

    def build_points(
        self,
        frame: SquatFrame,
        params: RelativeCalibrationParams,
    ) -> tuple[Point3D, Point3D, Point3D]:
        if not isinstance(params, RelativeDepthAddCalibrationParams):
            raise TypeError("RelativeAddModel requires RelativeDepthAddCalibrationParams.")

        raw_dz_thigh, raw_dz_shank = get_frame_raw_dz(frame, z_source=self.z_source)

        thigh_dz = predict_relative_add_dz(
            raw_dz=raw_dz_thigh,
            len_2d=frame.thigh_len_2d,
            ref_len=params.ref_thigh_len,
            a=params.thigh_a,
            c=params.thigh_c,
            b=params.thigh_b,
        )
        shank_dz = predict_relative_add_dz(
            raw_dz=raw_dz_shank,
            len_2d=frame.shank_len_2d,
            ref_len=params.ref_shank_len,
            a=params.shank_a,
            c=params.shank_c,
            b=params.shank_b,
        )

        knee = Point3D(frame.knee.x, frame.knee.y, 0.0)
        hip = Point3D(frame.hip.x, frame.hip.y, thigh_dz)
        ankle = Point3D(frame.ankle.x, frame.ankle.y, shank_dz)
        return hip, knee, ankle


MODEL_REGISTRY: dict[str, RelativeDepthModel] = {
    "relative_linear": RelativeLinearModel(),
    "relative_add": RelativeAddModel(),
    "mp_world_linear": RelativeLinearModel(name="mp_world_linear", z_source="mp_world"),
    "mp_world_add": RelativeAddModel(name="mp_world_add", z_source="mp_world"),
}


def get_relative_model(name: str) -> RelativeDepthModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported relative model={name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def fit_model_from_history(
    history_data: list[SquatFrame],
    model: RelativeDepthModel,
) -> RelativeCalibrationParams | None:
    return model.fit(history_data)


def compute_model_angles(
    history_data: list[SquatFrame],
    model: RelativeDepthModel,
    params: RelativeCalibrationParams,
) -> list[float]:
    return compute_relative_angles(history_data, model, params)