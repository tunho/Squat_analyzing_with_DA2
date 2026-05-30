from __future__ import annotations

from dataclasses import replace
import numpy as np

from squat.domain.types import Point3D


def point_to_np(p: Point3D) -> np.ndarray:
    return np.array([p.x, p.y, p.z], dtype=float)


def np_to_point(xyz: np.ndarray, ref: Point3D) -> Point3D:
    return Point3D(
        x=float(xyz[0]),
        y=float(xyz[1]),
        z=float(xyz[2]),
        vis=ref.vis,
    )


def safe_unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def ema_smooth_1d(values: list[float], alpha: float = 0.2) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for x in values[1:]:
        out.append(alpha * x + (1.0 - alpha) * out[-1])
    return out


def segment_lengths_from_triplet(hip: Point3D, knee: Point3D, ankle: Point3D) -> tuple[float, float]:
    h = point_to_np(hip)
    k = point_to_np(knee)
    a = point_to_np(ankle)
    thigh_len = float(np.linalg.norm(h - k))
    shank_len = float(np.linalg.norm(a - k))
    return thigh_len, shank_len


def stabilize_segment_lengths(
    hip: Point3D,
    knee: Point3D,
    ankle: Point3D,
    target_thigh_len: float,
    target_shank_len: float,
) -> tuple[Point3D, Point3D, Point3D]:
    h = point_to_np(hip)
    k = point_to_np(knee)
    a = point_to_np(ankle)

    u_thigh = safe_unit(h - k)
    u_shank = safe_unit(a - k)

    h_new = k + u_thigh * target_thigh_len
    a_new = k + u_shank * target_shank_len

    return (
        np_to_point(h_new, hip),
        knee,
        np_to_point(a_new, ankle),
    )

def blend_points(curr: Point3D, prev: Point3D, trust: float) -> Point3D:
    c = point_to_np(curr)
    p = point_to_np(prev)
    out = trust * c + (1.0 - trust) * p
    return np_to_point(out, curr)


def visibility_trust(
    hip: Point3D,
    knee: Point3D,
    ankle: Point3D,
    low: float = 0.3,
    high: float = 0.7,
) -> float:
    min_vis = min(hip.vis, knee.vis, ankle.vis)
    if min_vis <= low:
        return 0.0
    if min_vis >= high:
        return 1.0
    return (min_vis - low) / (high - low)

def angle_between_unit_vectors(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        return 0.0
    dot = float(np.dot(u / nu, v / nv))
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def robust_mad(values: list[float], eps: float = 1e-8) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < eps:
        mad = 1.0
    return med, mad


def interpolate_unit_vectors(u_prev: np.ndarray, u_next: np.ndarray, alpha: float) -> np.ndarray:
    # 단순 선형 보간 후 재정규화
    u = (1.0 - alpha) * u_prev + alpha * u_next
    n = np.linalg.norm(u)
    if n < 1e-8:
        return u_prev.copy()
    return u / n




def project_unit_to_plane(u: np.ndarray, n: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = u - float(np.dot(u, n)) * n
    norm = np.linalg.norm(v)
    if norm < eps:
        return safe_unit(u)
    return v / norm


def average_aligned_normals(normals: list[np.ndarray], eps: float = 1e-8) -> np.ndarray:
    valid = [safe_unit(np.asarray(n, dtype=float)) for n in normals if np.linalg.norm(n) > eps]
    if not valid:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    ref = valid[0]
    aligned: list[np.ndarray] = []
    for n in valid:
        if float(np.dot(n, ref)) < 0.0:
            n = -n
        aligned.append(n)

    mean_n = np.mean(np.stack(aligned, axis=0), axis=0)
    norm = np.linalg.norm(mean_n)
    if norm < eps:
        return ref
    return mean_n / norm


def segment_plane_deviation(u: np.ndarray, n_ref: np.ndarray) -> float:
    return abs(float(np.dot(safe_unit(u), safe_unit(n_ref))))


def construct_inplane_candidates(
    u_fixed_plane: np.ndarray,
    n_ref: np.ndarray,
    target_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    u_fixed_plane = project_unit_to_plane(u_fixed_plane, n_ref)
    theta = np.deg2rad(float(np.clip(target_angle_deg, 0.0, 180.0)))

    v_plane = np.cross(n_ref, u_fixed_plane)
    v_norm = np.linalg.norm(v_plane)
    if v_norm < 1e-8:
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(tmp, n_ref))) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=float)
        v_plane = np.cross(n_ref, tmp)
        v_plane = safe_unit(v_plane)
    else:
        v_plane = v_plane / v_norm

    cand1 = np.cos(theta) * u_fixed_plane + np.sin(theta) * v_plane
    cand2 = np.cos(theta) * u_fixed_plane - np.sin(theta) * v_plane
    return safe_unit(cand1), safe_unit(cand2)

def rebuild_triplet_from_knee_and_directions(
    knee: Point3D,
    u_thigh: np.ndarray,
    u_shank: np.ndarray,
    thigh_len: float,
    shank_len: float,
    ref_hip: Point3D,
    ref_ankle: Point3D,
) -> tuple[Point3D, Point3D, Point3D]:
    k = point_to_np(knee)
    h = k + u_thigh * thigh_len
    a = k + u_shank * shank_len
    return (
        np_to_point(h, ref_hip),
        knee,
        np_to_point(a, ref_ankle),
    )