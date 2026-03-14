from __future__ import annotations

from squat.calibration.relative_depth_calibration import RelativeDepthCalibrationParams
from squat.calibration.relative_model import extract_history_features
from squat.calibration.relative_models import RelativeLinearModel, compute_model_angles
from squat.domain.types import Point3D, SquatFrame

_LINEAR_MODEL = RelativeLinearModel()


def fit_relative_linear_from_history(
    history_data: list[SquatFrame],
) -> RelativeDepthCalibrationParams | None:
    return _LINEAR_MODEL.fit(history_data)


def build_relative_points(
    frame: SquatFrame,
    params: RelativeDepthCalibrationParams,
) -> tuple[Point3D, Point3D, Point3D]:
    return _LINEAR_MODEL.build_points(frame, params)


def compute_relative_linear_angles(
    history_data: list[SquatFrame],
    params: RelativeDepthCalibrationParams,
) -> list[float]:
    return compute_model_angles(history_data, _LINEAR_MODEL, params)