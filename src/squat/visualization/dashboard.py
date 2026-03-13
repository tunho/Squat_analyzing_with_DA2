from __future__ import annotations

import cv2
import numpy as np


def draw_dashboard(image: np.ndarray, reps: int, state: str, angle: float) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (400, 150), (245, 117, 16), -1)
    cv2.putText(canvas, f"REPS: {reps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(canvas, f"STATE: {state}", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(canvas, f"Knee 3D: {int(angle)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return canvas