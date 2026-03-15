# squat/analyzer/knee_mp_world_raw.py

from __future__ import annotations

from squat.analyzer.base import BaseSquatAnalyzer
from squat.geometry.angles import calculate_knee_angle


class MPWorldRawSquatAnalyzer(BaseSquatAnalyzer):
    def finalize_analysis(self):
        final_angles = []
        for frame in self.history_data:
            if frame.mp_world_hip is None or frame.mp_world_knee is None or frame.mp_world_ankle is None:
                continue
            angle = calculate_knee_angle(
                frame.mp_world_hip,
                frame.mp_world_knee,
                frame.mp_world_ankle,
            )
            final_angles.append(angle)
        return self.counter.count, final_angles, None