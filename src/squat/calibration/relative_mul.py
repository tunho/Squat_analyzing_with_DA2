from __future__ import annotations

from squat.calibration.relative_depth_mul_calibration import RelativeDepthMulCalibrationParams
from squat.calibration.relative_models import RelativeMulModel, compute_model_angles
from squat.domain.types import Point3D, SquatFrame

_MUL_MODEL = RelativeMulModel()


def fit_relative_mul_from_history(
    history_data: list[SquatFrame],
) -> RelativeDepthMulCalibrationParams | None:
    return _MUL_MODEL.fit(history_data)



def build_relative_mul_points(
    frame: SquatFrame,
    params: RelativeDepthMulCalibrationParams,
) -> tuple[Point3D, Point3D, Point3D]:
    return _MUL_MODEL.build_points(frame, params)



def compute_relative_mul_angles(
    history_data: list[SquatFrame],
    params: RelativeDepthMulCalibrationParams,
) -> list[float]:
    return compute_model_angles(history_data, _MUL_MODEL, params)