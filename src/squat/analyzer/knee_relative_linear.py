from __future__ import annotations

from squat.analyzer.base import BaseSquatAnalyzer
from squat.calibration.relative_depth_calibration import RelativeDepthCalibrationParams
from squat.calibration.relative_linear import (
    compute_relative_linear_angles,
    fit_relative_linear_from_history,
)


class RelativeLinearSquatAnalyzer(BaseSquatAnalyzer):
    def finalize_analysis(self) -> tuple[int, list[float], RelativeDepthCalibrationParams | None]:
        if not self.history_data:
            return 0, [], None

        params = fit_relative_linear_from_history(self.history_data)
        if params is None:
            return self.counter.count, [], None

        final_angles = compute_relative_linear_angles(self.history_data, params)
        return self.counter.count, final_angles, params