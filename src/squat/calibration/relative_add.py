from __future__ import annotations

from squat.calibration.relative_depth_add_calibration import RelativeDepthAddCalibrationParams
from squat.calibration.relative_models import RelativeAddModel, compute_model_angles
from squat.domain.types import Point3D, SquatFrame

_ADD_MODEL = RelativeAddModel()


def fit_relative_add_from_history(
    history_data: list[SquatFrame],
) -> RelativeDepthAddCalibrationParams | None:
    return _ADD_MODEL.fit(history_data)


def build_relative_add_points(
    frame: SquatFrame,
    params: RelativeDepthAddCalibrationParams,
) -> tuple[Point3D, Point3D, Point3D]:
    return _ADD_MODEL.build_points(frame, params)


def compute_relative_add_angles(
    history_data: list[SquatFrame],
    params: RelativeDepthAddCalibrationParams,
) -> list[float]:
    return compute_model_angles(history_data, _ADD_MODEL, params)