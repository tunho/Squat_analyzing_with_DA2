from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(slots=True)
class Point3D:
    x: float
    y: float
    z: float
    vis: float = 0.0


@dataclass(slots=True)
class SquatFrame:
    frame_index: int
    side: str
    hip: Point3D
    knee: Point3D
    ankle: Point3D
    shoulder: Optional[Point3D] = None
    shank_len_2d: float = 0.0
    thigh_len_2d: float = 0.0
    raw_image: Optional[np.ndarray] = None
    drawn_image: Optional[np.ndarray] = None
    depth_map_half: Optional[np.ndarray] = None
    orig_w: int = 0
    orig_h: int = 0