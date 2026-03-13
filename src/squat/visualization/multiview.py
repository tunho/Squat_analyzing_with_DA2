from __future__ import annotations

import cv2
import numpy as np

from squat.domain.types import Point3D, SquatFrame


def _project(pt: Point3D, angle_deg: float, offset: int, pivot: Point3D, width: int, height: int) -> tuple[int, int]:
    x = (pt.x - pivot.x) * 0.8
    y = (pt.y - pivot.y) * 0.8
    z = (pt.z - pivot.z) * 0.8
    rad = np.radians(angle_deg)
    px = int(x * np.cos(rad) - z * np.sin(rad) + width // 2) + offset
    py = int(y + height - 50)
    return px, py


def render_multiview_video(
    history_data: list[SquatFrame],
    output_path: str,
    base_k: float,
    stable_radii: dict[str, float],
) -> None:
    if not history_data:
        return

    h = history_data[0].orig_h
    w = history_data[0].orig_w
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w * 3, h))

    t_max = max(f.thigh_len_2d for f in history_data)
    s_max = max(f.shank_len_2d for f in history_data)
    f_thigh = max(t_max, s_max * 1.2)
    f_shank = f_thigh / 1.2

    for f in history_data:
        kz = f.knee.z * base_k + stable_radii["knee"]
        tz = np.sqrt(max(0.0, f_thigh**2 - f.thigh_len_2d**2))
        sz = np.sqrt(max(0.0, f_shank**2 - f.shank_len_2d**2))

        hip = Point3D(f.hip.x, f.hip.y, kz + (1 if f.hip.z > f.knee.z else -1) * tz)
        knee = Point3D(f.knee.x, f.knee.y, kz)
        ankle = Point3D(f.ankle.x, f.ankle.y, kz + (1 if f.ankle.z > f.knee.z else -1) * sz)
        shoulder = None
        if f.shoulder is not None:
            shoulder = Point3D(f.shoulder.x, f.shoulder.y, kz + (1 if f.hip.z > f.knee.z else -1) * tz)

        joints = {"shoulder": shoulder, "hip": hip, "knee": knee, "ankle": ankle}

        canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
        if f.raw_image is not None:
            canvas[:h, :w] = f.raw_image

        for i, angle in enumerate([45, 90]):
            offset = (i + 1) * w
            cv2.rectangle(canvas, (offset, 0), (offset + w, h), (40 - i * 20, 40 - i * 20, 40 - i * 20), -1)
            pivot = ankle
            proj = {k: _project(v, angle, offset, pivot, w, h) for k, v in joints.items() if v is not None}

            cv2.line(canvas, (offset + 20, proj["ankle"][1]), (offset + w - 20, proj["ankle"][1]), (100, 100, 100), 2)
            for s, e in [("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle")]:
                if s in proj and e in proj:
                    cv2.line(canvas, proj[s], proj[e], (255, 255, 255), 4)

            for key, point in proj.items():
                color = (0, 0, 255) if key == "knee" else (255, 0, 0) if key == "hip" else (0, 255, 255)
                cv2.circle(canvas, point, 8, color, -1)

        out.write(canvas)

    out.release()