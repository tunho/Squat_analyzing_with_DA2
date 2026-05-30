from __future__ import annotations

from typing import Any

from squat.analyzer.base import BaseSquatAnalyzer
from squat.domain.types import Point3D, SquatFrame
from squat.geometry.angles import calculate_knee_angle
from squat.state.squat_counter import SquatCounter


class BaseWorldKneeAnalyzer(BaseSquatAnalyzer):
    def get_world_triplet(self, frame: SquatFrame) -> tuple[Point3D, Point3D, Point3D] | None:
        if frame.mp_world_hip is None or frame.mp_world_knee is None or frame.mp_world_ankle is None:
            return None
        return frame.mp_world_hip, frame.mp_world_knee, frame.mp_world_ankle

    def transform_world_triplet(
        self,
        frame: SquatFrame,
        hip: Point3D,
        knee: Point3D,
        ankle: Point3D,
    ) -> tuple[Point3D, Point3D, Point3D]:
        return hip, knee, ankle

    def compute_frame_angle(self, squat_frame: SquatFrame) -> float | None:
        triplet = self.get_world_triplet(squat_frame)
        if triplet is None:
            return None
        hip, knee, ankle = self.transform_world_triplet(squat_frame, *triplet)
        return calculate_knee_angle(hip, knee, ankle)

    def build_final_angle_sequence(self) -> list[float]:
        final_angles = []
        for frame in self.history_data:
            angle = self.compute_frame_angle(frame)
            if angle is not None:
                final_angles.append(angle)
        return final_angles

    def recount_from_angles(self, angles: list[float]) -> int:
        counter = SquatCounter()
        for angle in angles:
            counter.update(angle)
        return counter.count

    def finalize_analysis(self) -> Any:
        final_angles = self.build_final_angle_sequence()
        final_count = self.recount_from_angles(final_angles)
        return final_count, final_angles, None