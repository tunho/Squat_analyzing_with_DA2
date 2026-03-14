from __future__ import annotations

from squat.analyzer.knee_relative import RelativeModelSquatAnalyzer
from squat.calibration.relative_depth_calibration import RelativeDepthCalibrationParams


class MPWorldLinearSquatAnalyzer(RelativeModelSquatAnalyzer):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(model="mp_world_linear", **kwargs)

    def finalize_analysis(self) -> tuple[int, list[float], RelativeDepthCalibrationParams | None]:
        return super().finalize_analysis()
