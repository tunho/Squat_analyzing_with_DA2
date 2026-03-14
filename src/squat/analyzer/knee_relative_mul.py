from __future__ import annotations

from squat.analyzer.knee_relative import RelativeModelSquatAnalyzer
from squat.calibration.relative_depth_mul_calibration import RelativeDepthMulCalibrationParams


class RelativeMulSquatAnalyzer(RelativeModelSquatAnalyzer):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(model="relative_mul", **kwargs)

    def finalize_analysis(self) -> tuple[int, list[float], RelativeDepthMulCalibrationParams | None]:
        return super().finalize_analysis()