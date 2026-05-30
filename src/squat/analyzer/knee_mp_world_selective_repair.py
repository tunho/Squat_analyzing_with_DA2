from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from squat.analyzer.knee_base_world import BaseWorldKneeAnalyzer
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import (
    angle_between_unit_vectors,
    interpolate_unit_vectors,
    point_to_np,
    rebuild_triplet_from_knee_and_directions,
    robust_mad,
    safe_unit,
)


@dataclass
class SelectiveRepairFeature:
    frame_index: int
    hip: object
    knee: object
    ankle: object
    raw_angle: float
    min_vis: float
    u_thigh: np.ndarray
    u_shank: np.ndarray
    thigh_len: float
    shank_len: float


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    if window <= 1:
        return values.astype(float).copy()
    pad = window // 2
    padded = np.pad(values.astype(float), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    out = np.convolve(padded, kernel, mode="valid")
    return out[: values.size]


def _gradient(frame_index: np.ndarray, values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros_like(values, dtype=float)
    if values.size == 2:
        dt = max(float(frame_index[1] - frame_index[0]), 1.0)
        g = (values[1] - values[0]) / dt
        return np.array([g, g], dtype=float)
    return np.gradient(values.astype(float), frame_index.astype(float))


def _robust_score_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    med, mad = robust_mad(values.tolist())
    return np.abs(values - med) / mad


def _label_angle_phases(
    frame_index: np.ndarray,
    angles: np.ndarray,
    smooth_window: int,
    top_band_ratio: float,
    bottom_band_ratio: float,
    motion_eps_ratio: float,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    smooth = _rolling_mean(angles, smooth_window)
    p5 = float(np.percentile(smooth, 5))
    p95 = float(np.percentile(smooth, 95))
    robust_range = max(p95 - p5, 1e-6)

    normalized_depth = (p95 - smooth) / robust_range
    normalized_depth = np.clip(normalized_depth, 0.0, 1.0)

    velocity = _gradient(frame_index, smooth)
    motion_eps = motion_eps_ratio * robust_range

    phases: list[str] = []
    last_motion_phase = "descent"
    for depth, vel in zip(normalized_depth, velocity):
        if depth <= top_band_ratio:
            phase = "top"
        elif depth >= bottom_band_ratio:
            phase = "bottom"
        else:
            if vel < -motion_eps:
                phase = "descent"
                last_motion_phase = phase
            elif vel > motion_eps:
                phase = "ascent"
                last_motion_phase = phase
            else:
                phase = last_motion_phase
        phases.append(phase)

    depth_mid = 0.5 * (top_band_ratio + bottom_band_ratio)
    subphases: list[str] = []
    for phase, depth in zip(phases, normalized_depth):
        if phase == "descent":
            subphases.append("descent_early" if depth < depth_mid else "descent_late")
        else:
            subphases.append("")

    return phases, subphases, normalized_depth, velocity


class MPWorldSelectiveRepairSquatAnalyzer(BaseWorldKneeAnalyzer):
    """
    Raw MP world 3D angle를 기본으로 유지하되,
    descent_late의 isolated instability frame만 최소 개입으로 보정한다.

    핵심 아이디어:
    - phase gate: raw angle 기반으로 descent_late만 후보로 본다.
    - anomaly gate: local angle impulse + direction jump/low visibility 중복 증거가 있어야 한다.
    - neighborhood consistency gate: 이웃 두 프레임 자체는 서로 일관적이어야 한다.
      (즉, 현재 프레임만 튄 isolated spike여야 한다.)
    - repair: world thigh/shank unit direction을 이웃 방향 사이로 아주 조금만 당긴다.
      필요한 만큼만(blend 최소) 적용한다.
    """

    def __init__(
        self,
        *args,
        phase_smooth_window: int = 5,
        phase_top_band_ratio: float = 0.15,
        phase_bottom_band_ratio: float = 0.85,
        phase_motion_eps_ratio: float = 0.01,
        anomaly_z_threshold: float = 3.0,
        bridge_consistency_quantile: float = 0.7,
        repair_context_radius: int = 2,
        max_blend: float = 1.0,
        blend_steps: int = 21,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.phase_smooth_window = phase_smooth_window
        self.phase_top_band_ratio = phase_top_band_ratio
        self.phase_bottom_band_ratio = phase_bottom_band_ratio
        self.phase_motion_eps_ratio = phase_motion_eps_ratio
        self.anomaly_z_threshold = anomaly_z_threshold
        self.bridge_consistency_quantile = bridge_consistency_quantile
        self.repair_context_radius = repair_context_radius
        self.max_blend = max_blend
        self.blend_steps = blend_steps
        self._last_extra: dict | None = None

    def _extract_features(self) -> list[SelectiveRepairFeature]:
        feats: list[SelectiveRepairFeature] = []
        for frame in self.history_data:
            triplet = self.get_world_triplet(frame)
            if triplet is None:
                continue
            hip, knee, ankle = triplet

            h = point_to_np(hip)
            k = point_to_np(knee)
            a = point_to_np(ankle)

            thigh_vec = h - k
            shank_vec = a - k
            thigh_len = float(np.linalg.norm(thigh_vec))
            shank_len = float(np.linalg.norm(shank_vec))

            feats.append(
                SelectiveRepairFeature(
                    frame_index=frame.frame_index,
                    hip=hip,
                    knee=knee,
                    ankle=ankle,
                    raw_angle=calculate_knee_angle(hip, knee, ankle),
                    min_vis=float(min(hip.vis, knee.vis, ankle.vis)),
                    u_thigh=safe_unit(thigh_vec),
                    u_shank=safe_unit(shank_vec),
                    thigh_len=thigh_len,
                    shank_len=shank_len,
                )
            )
        return feats

    def _find_context_neighbors(self, stable_mask: np.ndarray, idx: int) -> tuple[int | None, int | None]:
        prev_idx = None
        next_idx = None
        for gap in range(1, self.repair_context_radius + 1):
            j = idx - gap
            if j >= 0 and stable_mask[j]:
                prev_idx = j
                break
        for gap in range(1, self.repair_context_radius + 1):
            j = idx + gap
            if j < stable_mask.size and stable_mask[j]:
                next_idx = j
                break
        return prev_idx, next_idx

    def _blend_toward_target(
        self,
        feat: SelectiveRepairFeature,
        u_thigh_target: np.ndarray,
        u_shank_target: np.ndarray,
        low_angle: float,
        high_angle: float,
    ) -> tuple[float, float]:
        weights = np.linspace(0.0, self.max_blend, self.blend_steps)

        best_dev = float("inf")
        best_weight = 0.0
        best_angle = feat.raw_angle

        for weight in weights:
            u_thigh = safe_unit((1.0 - weight) * feat.u_thigh + weight * u_thigh_target)
            u_shank = safe_unit((1.0 - weight) * feat.u_shank + weight * u_shank_target)

            hip_c, knee_c, ankle_c = rebuild_triplet_from_knee_and_directions(
                knee=feat.knee,
                u_thigh=u_thigh,
                u_shank=u_shank,
                thigh_len=feat.thigh_len,
                shank_len=feat.shank_len,
                ref_hip=feat.hip,
                ref_ankle=feat.ankle,
            )
            angle_c = calculate_knee_angle(hip_c, knee_c, ankle_c)

            if angle_c < low_angle:
                dev = low_angle - angle_c
            elif angle_c > high_angle:
                dev = angle_c - high_angle
            else:
                dev = 0.0

            if dev <= 1e-6:
                return weight, angle_c

            if dev < best_dev - 1e-6 or (abs(dev - best_dev) <= 1e-6 and weight < best_weight):
                best_dev = dev
                best_weight = weight
                best_angle = angle_c

        return best_weight, best_angle

    def build_final_angle_sequence(self) -> list[float]:
        feats = self._extract_features()
        if not feats:
            self._last_extra = {
                "num_valid_frames": 0,
                "num_candidates": 0,
                "num_repaired": 0,
                "repair_rate": 0.0,
            }
            return []

        n = len(feats)
        frame_index = np.asarray([f.frame_index for f in feats], dtype=float)
        raw_angles = np.asarray([f.raw_angle for f in feats], dtype=float)
        min_vis = np.asarray([f.min_vis for f in feats], dtype=float)

        phases, subphases, normalized_depth, raw_vel = _label_angle_phases(
            frame_index=frame_index,
            angles=raw_angles,
            smooth_window=self.phase_smooth_window,
            top_band_ratio=self.phase_top_band_ratio,
            bottom_band_ratio=self.phase_bottom_band_ratio,
            motion_eps_ratio=self.phase_motion_eps_ratio,
        )

        angle_delta = np.abs(np.diff(raw_angles))
        typical_motion = float(np.median(angle_delta)) if angle_delta.size > 0 else 0.0
        local_angle_margin = max(typical_motion, 1.0)

        angle_impulse = np.zeros(n, dtype=float)
        dir_jump_center = np.zeros(n, dtype=float)
        bridge_dir_gap = np.full(n, np.nan, dtype=float)
        bridge_angle_gap = np.full(n, np.nan, dtype=float)
        velocity_flip = np.zeros(n, dtype=bool)

        for i in range(1, n - 1):
            prev_feat = feats[i - 1]
            curr_feat = feats[i]
            next_feat = feats[i + 1]

            dt_full = max(float(next_feat.frame_index - prev_feat.frame_index), 1.0)
            alpha = float((curr_feat.frame_index - prev_feat.frame_index) / dt_full)
            expected_angle = (1.0 - alpha) * prev_feat.raw_angle + alpha * next_feat.raw_angle
            angle_impulse[i] = abs(curr_feat.raw_angle - expected_angle)

            thigh_prev = angle_between_unit_vectors(prev_feat.u_thigh, curr_feat.u_thigh)
            thigh_next = angle_between_unit_vectors(curr_feat.u_thigh, next_feat.u_thigh)
            shank_prev = angle_between_unit_vectors(prev_feat.u_shank, curr_feat.u_shank)
            shank_next = angle_between_unit_vectors(curr_feat.u_shank, next_feat.u_shank)
            dir_jump_center[i] = max(thigh_prev, thigh_next, shank_prev, shank_next)

            bridge_dir_gap[i] = max(
                angle_between_unit_vectors(prev_feat.u_thigh, next_feat.u_thigh),
                angle_between_unit_vectors(prev_feat.u_shank, next_feat.u_shank),
            )
            bridge_angle_gap[i] = abs(prev_feat.raw_angle - next_feat.raw_angle)

            delta_prev = curr_feat.raw_angle - prev_feat.raw_angle
            delta_next = next_feat.raw_angle - curr_feat.raw_angle
            velocity_flip[i] = (delta_prev < 0.0 and delta_next > 0.0) or (delta_prev > 0.0 and delta_next < 0.0)

        impulse_score = _robust_score_array(angle_impulse)
        dir_jump_score = _robust_score_array(dir_jump_center)

        valid_bridge_dir = bridge_dir_gap[~np.isnan(bridge_dir_gap)]
        valid_bridge_angle = bridge_angle_gap[~np.isnan(bridge_angle_gap)]
        bridge_dir_limit = float(np.quantile(valid_bridge_dir, self.bridge_consistency_quantile)) if valid_bridge_dir.size else float("inf")
        bridge_angle_limit = float(np.quantile(valid_bridge_angle, self.bridge_consistency_quantile)) if valid_bridge_angle.size else float("inf")

        vis_floor = float(min(0.5, np.quantile(min_vis, 0.25))) if min_vis.size else 0.5

        candidate_mask = np.zeros(n, dtype=bool)
        candidate_reason_counts = {
            "phase_gate": 0,
            "strong_impulse": 0,
            "strong_dir_jump": 0,
            "low_visibility": 0,
            "velocity_flip": 0,
            "neighborhood_consistent": 0,
        }

        for i in range(1, n - 1):
            phase_gate = subphases[i] == "descent_late"
            if phase_gate:
                candidate_reason_counts["phase_gate"] += 1

            strong_impulse = impulse_score[i] >= self.anomaly_z_threshold and angle_impulse[i] > local_angle_margin
            if strong_impulse:
                candidate_reason_counts["strong_impulse"] += 1

            strong_dir_jump = dir_jump_score[i] >= self.anomaly_z_threshold
            if strong_dir_jump:
                candidate_reason_counts["strong_dir_jump"] += 1

            low_visibility = min_vis[i] < vis_floor
            if low_visibility:
                candidate_reason_counts["low_visibility"] += 1

            if velocity_flip[i]:
                candidate_reason_counts["velocity_flip"] += 1

            neighborhood_consistent = (
                not np.isnan(bridge_dir_gap[i])
                and not np.isnan(bridge_angle_gap[i])
                and bridge_dir_gap[i] <= bridge_dir_limit
                and bridge_angle_gap[i] <= bridge_angle_limit + local_angle_margin
            )
            if neighborhood_consistent:
                candidate_reason_counts["neighborhood_consistent"] += 1

            evidence_gate = strong_dir_jump or low_visibility or velocity_flip[i]

            candidate_mask[i] = phase_gate and strong_impulse and neighborhood_consistent and evidence_gate

        stable_mask = ~candidate_mask
        final_angles: list[float] = []
        repaired_mask = np.zeros(n, dtype=bool)
        repair_weights = np.zeros(n, dtype=float)

        for i, feat in enumerate(feats):
            if not candidate_mask[i]:
                final_angles.append(feat.raw_angle)
                continue

            prev_idx, next_idx = self._find_context_neighbors(stable_mask, i)
            if prev_idx is None or next_idx is None:
                final_angles.append(feat.raw_angle)
                continue

            prev_feat = feats[prev_idx]
            next_feat = feats[next_idx]
            dt_full = max(float(next_feat.frame_index - prev_feat.frame_index), 1.0)
            alpha = float((feat.frame_index - prev_feat.frame_index) / dt_full)
            alpha = float(np.clip(alpha, 0.0, 1.0))

            u_thigh_target = interpolate_unit_vectors(prev_feat.u_thigh, next_feat.u_thigh, alpha)
            u_shank_target = interpolate_unit_vectors(prev_feat.u_shank, next_feat.u_shank, alpha)

            low_angle = min(prev_feat.raw_angle, next_feat.raw_angle) - local_angle_margin
            high_angle = max(prev_feat.raw_angle, next_feat.raw_angle) + local_angle_margin

            weight, angle_repaired = self._blend_toward_target(
                feat=feat,
                u_thigh_target=u_thigh_target,
                u_shank_target=u_shank_target,
                low_angle=low_angle,
                high_angle=high_angle,
            )

            if abs(angle_repaired - feat.raw_angle) > 1e-6:
                repaired_mask[i] = True
                repair_weights[i] = weight
                final_angles.append(angle_repaired)
            else:
                final_angles.append(feat.raw_angle)

        candidate_indices = [int(feats[i].frame_index) for i in range(n) if candidate_mask[i]]
        repaired_indices = [int(feats[i].frame_index) for i in range(n) if repaired_mask[i]]

        self._last_extra = {
            "num_valid_frames": int(n),
            "num_candidates": int(candidate_mask.sum()),
            "num_repaired": int(repaired_mask.sum()),
            "repair_rate": float(repaired_mask.mean()),
            "phase_params": {
                "phase_smooth_window": self.phase_smooth_window,
                "phase_top_band_ratio": self.phase_top_band_ratio,
                "phase_bottom_band_ratio": self.phase_bottom_band_ratio,
                "phase_motion_eps_ratio": self.phase_motion_eps_ratio,
            },
            "repair_params": {
                "anomaly_z_threshold": self.anomaly_z_threshold,
                "bridge_consistency_quantile": self.bridge_consistency_quantile,
                "repair_context_radius": self.repair_context_radius,
                "max_blend": self.max_blend,
                "blend_steps": self.blend_steps,
                "local_angle_margin": local_angle_margin,
                "vis_floor": vis_floor,
                "bridge_dir_limit": bridge_dir_limit,
                "bridge_angle_limit": bridge_angle_limit,
            },
            "candidate_reason_counts": candidate_reason_counts,
            "candidate_frame_indices": candidate_indices,
            "repaired_frame_indices": repaired_indices,
            "mean_repair_weight": float(repair_weights[repaired_mask].mean()) if repaired_mask.any() else 0.0,
            "median_repair_weight": float(np.median(repair_weights[repaired_mask])) if repaired_mask.any() else 0.0,
            "raw_phase_counts": {
                "top": int(sum(p == "top" for p in phases)),
                "descent": int(sum(p == "descent" for p in phases)),
                "bottom": int(sum(p == "bottom" for p in phases)),
                "ascent": int(sum(p == "ascent" for p in phases)),
                "descent_late": int(sum(s == "descent_late" for s in subphases)),
            },
        }
        return final_angles

    def finalize_analysis(self):
        final_angles = self.build_final_angle_sequence()
        final_count = self.recount_from_angles(final_angles)
        return final_count, final_angles, self._last_extra
