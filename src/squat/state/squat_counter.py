from __future__ import annotations

from dataclasses import dataclass

from squat.config import SQUAT_THRESHOLDS


@dataclass
class SquatCounter:
    state: str = "UP"
    count: int = 0

    def update(self, knee_angle: float) -> None:
        if knee_angle < SQUAT_THRESHOLDS["KNEE_ANGLE_DOWN"]:
            self.state = "DOWN"
        if self.state == "DOWN" and knee_angle > SQUAT_THRESHOLDS["KNEE_ANGLE_UP"]:
            self.count += 1
            self.state = "UP"