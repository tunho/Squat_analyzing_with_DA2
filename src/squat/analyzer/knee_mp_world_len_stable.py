from __future__ import annotations

import numpy as np

from squat.analyzer.knee_base_world import BaseWorldKneeAnalyzer
from squat.domain.types import Point3D, SquatFrame
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import (
    segment_lengths_from_triplet,
    stabilize_segment_lengths,
)


class MPWorldLenStableSquatAnalyzer(BaseWorldKneeAnalyzer):
    def __init__(self, *args, use_visibility_weight: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_visibility_weight = use_visibility_weight

    def _collect_target_lengths(self) -> tuple[float, float]:
        thigh_lengths = []
        shank_lengths = []

        for frame in self.history_data:
            triplet = self.get_world_triplet(frame)
            if triplet is None:
                continue
            hip, knee, ankle = triplet
            thigh_len, shank_len = segment_lengths_from_triplet(hip, knee, ankle)
            thigh_lengths.append(thigh_len)
            shank_lengths.append(shank_len)

        if not thigh_lengths or not shank_lengths:
            return 0.0, 0.0

        return float(np.median(thigh_lengths)), float(np.median(shank_lengths))

    def build_final_angle_sequence(self) -> list[float]:
        target_thigh_len, target_shank_len = self._collect_target_lengths()

        final_angles = []
        for frame in self.history_data:
            triplet = self.get_world_triplet(frame)
            if triplet is None:
                continue

            hip_s, knee_s, ankle_s = stabilize_segment_lengths(
                hip=triplet[0],
                knee=triplet[1],
                ankle=triplet[2],
                target_thigh_len=target_thigh_len,
                target_shank_len=target_shank_len,
            )
            final_angles.append(calculate_knee_angle(hip_s, knee_s, ankle_s))

        return final_angles