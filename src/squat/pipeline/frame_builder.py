from __future__ import annotations

import cv2
import numpy as np

from squat.domain.types import Point3D, SquatFrame


SIDE_LANDMARKS = {
    "left": (23, 25, 27, 11),
    "right": (24, 26, 28, 12),
}


def choose_side(landmarks: list[dict]) -> str:
    l_vis = (landmarks[11]["visibility"] + landmarks[23]["visibility"] + landmarks[25]["visibility"]) / 3.0
    r_vis = (landmarks[12]["visibility"] + landmarks[24]["visibility"] + landmarks[26]["visibility"]) / 3.0
    return "left" if l_vis >= r_vis else "right"


def depth_point_from_landmark(landmark: dict, depth_map: np.ndarray) -> Point3D:
    h, w = depth_map.shape
    px = int(landmark["x"] * w)
    py = int(landmark["y"] * h)
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))
    return Point3D(
        x=float(px),
        y=float(py),
        z=float(depth_map[py, px]),
        vis=float(landmark.get("visibility", 0.0)),
    )


def world_point_from_landmark(landmark: dict | None) -> Point3D | None:
    if landmark is None:
        return None
    x = landmark.get("x")
    y = landmark.get("y")
    z = landmark.get("z")
    if x is None or y is None or z is None:
        return None
    return Point3D(
        x=float(x),
        y=float(y),
        z=float(z),
        vis=float(landmark.get("visibility", 0.0)),
    )


def build_squat_frame(
    frame_index: int,
    side: str,
    landmarks: list[dict],
    depth_map: np.ndarray,
    frame_bgr: np.ndarray,
    drawn_image: np.ndarray | None,
    store_debug_images: bool,
    store_depth_maps: bool,
    world_landmarks: list[dict] | None = None,
) -> SquatFrame:
    h, w = frame_bgr.shape[:2]
    idxs = SIDE_LANDMARKS[side]
    hip, knee, ankle, shoulder = [depth_point_from_landmark(landmarks[i], depth_map) for i in idxs]

    shank_len_2d = float(np.linalg.norm([ankle.x - knee.x, ankle.y - knee.y]))
    thigh_len_2d = float(np.linalg.norm([knee.x - hip.x, knee.y - hip.y]))

    mp_world_hip = None
    mp_world_knee = None
    mp_world_ankle = None
    if world_landmarks and len(world_landmarks) > max(idxs[:3]):
        mp_world_hip = world_point_from_landmark(world_landmarks[idxs[0]])
        mp_world_knee = world_point_from_landmark(world_landmarks[idxs[1]])
        mp_world_ankle = world_point_from_landmark(world_landmarks[idxs[2]])

    return SquatFrame(
        frame_index=frame_index,
        side=side,
        hip=hip,
        knee=knee,
        ankle=ankle,
        shoulder=shoulder,
        mp_world_hip=mp_world_hip,
        mp_world_knee=mp_world_knee,
        mp_world_ankle=mp_world_ankle,
        shank_len_2d=shank_len_2d,
        thigh_len_2d=thigh_len_2d,
        raw_image=frame_bgr.copy() if store_debug_images else None,
        drawn_image=drawn_image.copy() if (store_debug_images and drawn_image is not None) else None,
        depth_map_half=cv2.resize(depth_map, (w // 2, h // 2)).astype(np.float16) if store_depth_maps else None,
        orig_w=w,
        orig_h=h,
    )