from __future__ import annotations

import numpy as np
from squat.domain.types import Point3D


def calculate_knee_angle(hip: Point3D, knee: Point3D, ankle: Point3D) -> float:
    v1 = np.array([hip.x - knee.x, hip.y - knee.y, hip.z - knee.z], dtype=float)
    v2 = np.array([ankle.x - knee.x, ankle.y - knee.y, ankle.z - knee.z], dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 180.0

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))