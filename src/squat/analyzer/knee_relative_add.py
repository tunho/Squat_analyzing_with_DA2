from __future__ import annotations

from squat.analyzer.knee_relative import RelativeModelSquatAnalyzer
from squat.calibration.relative_depth_add_calibration import RelativeDepthAddCalibrationParams


class RelativeAddSquatAnalyzer(RelativeModelSquatAnalyzer):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(model="relative_add", **kwargs)

    def finalize_analysis(self) -> tuple[int, list[float], RelativeDepthAddCalibrationParams | None]:
        return super().finalize_analysis()