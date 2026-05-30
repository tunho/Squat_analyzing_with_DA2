from __future__ import annotations

from squat.analyzer.knee_base_world import BaseWorldKneeAnalyzer
from squat.geometry.world_knee_ops import ema_smooth_1d


class MPWorldSmoothSquatAnalyzer(BaseWorldKneeAnalyzer):
    def __init__(self, *args, smooth_alpha: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.smooth_alpha = smooth_alpha

    def build_final_angle_sequence(self) -> list[float]:
        raw_angles = super().build_final_angle_sequence()
        return ema_smooth_1d(raw_angles, alpha=self.smooth_alpha)