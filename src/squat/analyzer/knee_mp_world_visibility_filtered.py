from __future__ import annotations

import numpy as np

from squat.analyzer.knee_base_world import BaseWorldKneeAnalyzer
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import (
    blend_points,
    segment_lengths_from_triplet,
    stabilize_segment_lengths,
    visibility_trust,
)


class MPWorldVisibilityFilteredSquatAnalyzer(BaseWorldKneeAnalyzer):
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
        prev_triplet = None

        for frame in self.history_data:
            triplet = self.get_world_triplet(frame)
            if triplet is None:
                continue

            hip, knee, ankle = stabilize_segment_lengths(
                hip=triplet[0],
                knee=triplet[1],
                ankle=triplet[2],
                target_thigh_len=target_thigh_len,
                target_shank_len=target_shank_len,
            )

            if prev_triplet is not None:
                trust = visibility_trust(hip, knee, ankle, low=0.3, high=0.7)
                hip = blend_points(hip, prev_triplet[0], trust)
                knee = blend_points(knee, prev_triplet[1], trust)
                ankle = blend_points(ankle, prev_triplet[2], trust)

            prev_triplet = (hip, knee, ankle)
            final_angles.append(calculate_knee_angle(hip, knee, ankle))

        return final_angles