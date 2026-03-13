from __future__ import annotations

import os

import cv2

from squat.config import OUTPUT_CONFIG


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    return cap


def read_video_info(cap: cv2.VideoCapture) -> tuple[float, int, int]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 100:
        fps = OUTPUT_CONFIG.get("VIDEO_FPS", 30.0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, width, height


def create_video_writer(
    output_path: str | None,
    fps: float,
    width: int,
    height: int,
    save_video: bool,
) -> cv2.VideoWriter | None:
    if not save_video or not output_path:
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    return cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )