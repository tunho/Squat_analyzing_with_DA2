from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from squat.domain.types import SquatFrame
from squat.geometry.angles import calculate_knee_angle
from squat.pipeline.frame_builder import build_squat_frame, choose_side


@dataclass(slots=True)
class FrameProcessResult:
    squat_frame: SquatFrame | None
    output_frame: np.ndarray
    resolved_side: str | None
    raw_angle: float | None


def _extract_landmarks(pose_estimator: Any, frame: np.ndarray) -> tuple[list[dict] | None, Any | None]:
    extracted = pose_estimator.extract_keypoints_only(frame.copy())
    if isinstance(extracted, tuple):
        landmarks = extracted[0]
        result = extracted[1] if len(extracted) > 1 else None
    else:
        landmarks = extracted
        result = None
    return landmarks, result


def _extract_world_landmarks(result: Any | None) -> list[dict] | None:
    if result is None or not hasattr(result, "pose_world_landmarks"):
        return None
    pose_world = getattr(result, "pose_world_landmarks", None)
    if not pose_world:
        return None

    world_landmarks: list[dict] = []
    for idx, lm in enumerate(pose_world[0]):
        vis = getattr(lm, "visibility", None)
        presence = getattr(lm, "presence", None)
        world_landmarks.append({
            "id": idx,
            "x": float(getattr(lm, "x", 0.0)),
            "y": float(getattr(lm, "y", 0.0)),
            "z": float(getattr(lm, "z", 0.0)),
            "visibility": float(vis if vis is not None else (presence or 0.0)),
        })
    return world_landmarks


def should_skip_frame_by_visibility(
    squat_frame: SquatFrame,
    visibility_threshold: float = 0.1,
) -> bool:
    return min(
        squat_frame.hip.vis,
        squat_frame.knee.vis,
        squat_frame.ankle.vis,
    ) <= visibility_threshold


def process_frame(
    frame: np.ndarray,
    frame_index: int,
    side: str | None,
    pose_estimator: Any,
    depth_estimator: Any,
    store_debug_images: bool,
    store_depth_maps: bool,
    visibility_threshold: float = 0.1,
) -> FrameProcessResult:
    landmarks, pose_result = _extract_landmarks(pose_estimator, frame)
    if not landmarks:
        return FrameProcessResult(
            squat_frame=None,
            output_frame=frame,
            resolved_side=side,
            raw_angle=None,
        )

    depth_map = depth_estimator.estimate(frame)
    resolved_side = side or choose_side(landmarks)
    world_landmarks = _extract_world_landmarks(pose_result)

    image_draw = frame.copy()
    pose_estimator._draw_landmarks(image_draw, landmarks)

    squat_frame = build_squat_frame(
        frame_index=frame_index,
        side=resolved_side,
        landmarks=landmarks,
        depth_map=depth_map,
        frame_bgr=frame,
        drawn_image=image_draw,
        store_debug_images=store_debug_images,
        store_depth_maps=store_depth_maps,
        world_landmarks=world_landmarks,
    )

    if should_skip_frame_by_visibility(squat_frame, visibility_threshold=visibility_threshold):
        return FrameProcessResult(
            squat_frame=None,
            output_frame=frame,
            resolved_side=resolved_side,
            raw_angle=None,
        )

    raw_angle = calculate_knee_angle(
        squat_frame.hip,
        squat_frame.knee,
        squat_frame.ankle,
    )

    return FrameProcessResult(
        squat_frame=squat_frame,
        output_frame=image_draw,
        resolved_side=resolved_side,
        raw_angle=raw_angle,
    )