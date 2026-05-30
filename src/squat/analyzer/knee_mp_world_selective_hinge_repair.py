from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from squat.analyzer.knee_base_world import BaseWorldKneeAnalyzer
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import (
    angle_between_unit_vectors,
    average_aligned_normals,
    construct_inplane_candidates,
    interpolate_unit_vectors,
    point_to_np,
    project_unit_to_plane,
    rebuild_triplet_from_knee_and_directions,
    safe_unit,
    segment_plane_deviation,
)


@dataclass
class HingeRepairFeature:
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



def _safe_quantile(values: np.ndarray, q: float, default: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))



def _segment_excess(u_prev: np.ndarray, u_curr: np.ndarray, u_next: np.ndarray) -> float:
    prev_jump = angle_between_unit_vectors(u_prev, u_curr)
    next_jump = angle_between_unit_vectors(u_curr, u_next)
    bridge = angle_between_unit_vectors(u_prev, u_next)
    excess = 0.5 * (prev_jump + next_jump) - bridge
    return float(max(excess, 0.0))



def _group_candidate_runs(candidate_mask: np.ndarray, max_hole: int, max_run_len: int) -> list[tuple[int, int]]:
    idxs = np.flatnonzero(candidate_mask)
    if idxs.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = int(idxs[0])
    prev = int(idxs[0])
    for idx in idxs[1:]:
        idx = int(idx)
        if idx - prev <= max_hole + 1:
            prev = idx
            continue
        if prev - start + 1 <= max_run_len:
            runs.append((start, prev))
        start = idx
        prev = idx

    if prev - start + 1 <= max_run_len:
        runs.append((start, prev))
    return runs


class MPWorldSelectiveHingeRepairSquatAnalyzer(BaseWorldKneeAnalyzer):
    """
    Burst candidate는 유지하되, repair operator를 temporal blend에서
    hinge-plane constrained reconstruction으로 교체한 analyzer.

    핵심:
    - 기본 threshold는 현재 H2/H1 완화 실험값으로 설정
    - raw world 3D를 기본으로 유지
    - descent_late short burst만 후보로 선택
    - stable anchor의 thigh-shank plane normal로 local hinge plane 추정
    - 더 불안정한 segment 하나만 plane 안에서 expected angle을 만족하도록 재구성
    - run 전체 objective가 실제로 좋아질 때만 적용
    """

    def __init__(
        self,
        *args,
        phase_smooth_window: int = 5,
        phase_top_band_ratio: float = 0.15,
        phase_bottom_band_ratio: float = 0.85,
        phase_motion_eps_ratio: float = 0.01,
        candidate_impulse_quantile: float = 0.85,
        candidate_segment_quantile: float = 0.90,
        candidate_visibility_quantile: float = 0.20,
        candidate_squat_violation_quantile: float = 0.80,
        candidate_hole_tolerance: int = 1,
        max_run_len: int = 5,
        anchor_search_radius: int = 8,
        anchor_context_radius: int = 2,
        lambda_offplane: float = 1.0,
        max_blend: float = 1.0,
        blend_steps: int = 21,
        run_improvement_margin_deg: float = 0.5,
        envelope_margin_scale: float = 1.5,
        plane_penalty_scale: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.phase_smooth_window = phase_smooth_window
        self.phase_top_band_ratio = phase_top_band_ratio
        self.phase_bottom_band_ratio = phase_bottom_band_ratio
        self.phase_motion_eps_ratio = phase_motion_eps_ratio
        self.candidate_impulse_quantile = candidate_impulse_quantile
        self.candidate_segment_quantile = candidate_segment_quantile
        self.candidate_visibility_quantile = candidate_visibility_quantile
        self.candidate_squat_violation_quantile = candidate_squat_violation_quantile
        self.candidate_hole_tolerance = candidate_hole_tolerance
        self.max_run_len = max_run_len
        self.anchor_search_radius = anchor_search_radius
        self.anchor_context_radius = anchor_context_radius
        self.lambda_offplane = lambda_offplane
        self.max_blend = max_blend
        self.blend_steps = blend_steps
        self.run_improvement_margin_deg = run_improvement_margin_deg
        self.envelope_margin_scale = envelope_margin_scale
        self.plane_penalty_scale = plane_penalty_scale
        self._last_extra: dict | None = None

    def _extract_features(self) -> list[HingeRepairFeature]:
        feats: list[HingeRepairFeature] = []
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
            feats.append(
                HingeRepairFeature(
                    frame_index=frame.frame_index,
                    hip=hip,
                    knee=knee,
                    ankle=ankle,
                    raw_angle=calculate_knee_angle(hip, knee, ankle),
                    min_vis=float(min(hip.vis, knee.vis, ankle.vis)),
                    u_thigh=safe_unit(thigh_vec),
                    u_shank=safe_unit(shank_vec),
                    thigh_len=float(np.linalg.norm(thigh_vec)),
                    shank_len=float(np.linalg.norm(shank_vec)),
                )
            )
        return feats

    def _find_run_anchors(self, candidate_mask: np.ndarray, run_start: int, run_end: int) -> tuple[int | None, int | None]:
        left = None
        right = None
        for gap in range(1, self.anchor_search_radius + 1):
            idx = run_start - gap
            if idx >= 0 and not candidate_mask[idx]:
                left = idx
                break
        for gap in range(1, self.anchor_search_radius + 1):
            idx = run_end + gap
            if idx < candidate_mask.size and not candidate_mask[idx]:
                right = idx
                break
        return left, right

    def _collect_anchor_context(self, left_idx: int, right_idx: int, n: int) -> list[int]:
        indices: list[int] = []
        for base in (left_idx, right_idx):
            start = max(0, base - self.anchor_context_radius)
            end = min(n - 1, base + self.anchor_context_radius)
            for idx in range(start, end + 1):
                if idx not in indices:
                    indices.append(idx)
        return indices

    def _estimate_reference_plane_normal(self, feats: list[HingeRepairFeature], anchor_indices: list[int]) -> np.ndarray:
        normals: list[np.ndarray] = []
        for idx in anchor_indices:
            feat = feats[idx]
            n_i = np.cross(feat.u_thigh, feat.u_shank)
            if np.linalg.norm(n_i) > 1e-8:
                normals.append(n_i)
        return average_aligned_normals(normals)

    def _build_hinge_target_direction(
        self,
        dominant_segment: str,
        feat: HingeRepairFeature,
        expected_angle: float,
        n_ref: np.ndarray,
    ) -> np.ndarray:
        if dominant_segment == "shank":
            u_fixed_plane = project_unit_to_plane(feat.u_thigh, n_ref)
            cand1, cand2 = construct_inplane_candidates(u_fixed_plane, n_ref, expected_angle)
            raw_proj = project_unit_to_plane(feat.u_shank, n_ref)
        else:
            u_fixed_plane = project_unit_to_plane(feat.u_shank, n_ref)
            cand1, cand2 = construct_inplane_candidates(u_fixed_plane, n_ref, expected_angle)
            raw_proj = project_unit_to_plane(feat.u_thigh, n_ref)

        if float(np.dot(cand1, raw_proj)) >= float(np.dot(cand2, raw_proj)):
            return cand1
        return cand2

    def _repair_single_frame_hinge(
        self,
        feat: HingeRepairFeature,
        dominant_segment: str,
        target_dir: np.ndarray,
        expected_angle: float,
        low_angle: float,
        high_angle: float,
        n_ref: np.ndarray,
    ) -> tuple[float, float, float, float]:
        weights = np.linspace(0.0, self.max_blend, self.blend_steps)

        best_weight = 0.0
        best_angle = feat.raw_angle
        if dominant_segment == "thigh":
            best_plane_dev = segment_plane_deviation(feat.u_thigh, n_ref)
        else:
            best_plane_dev = segment_plane_deviation(feat.u_shank, n_ref)
        best_objective = abs(feat.raw_angle - expected_angle) + self.plane_penalty_scale * best_plane_dev

        for weight in weights[1:]:
            if dominant_segment == "thigh":
                u_thigh = interpolate_unit_vectors(feat.u_thigh, target_dir, float(weight))
                u_shank = feat.u_shank
                plane_dev = segment_plane_deviation(u_thigh, n_ref)
            else:
                u_thigh = feat.u_thigh
                u_shank = interpolate_unit_vectors(feat.u_shank, target_dir, float(weight))
                plane_dev = segment_plane_deviation(u_shank, n_ref)

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
            objective = abs(angle_c - expected_angle) + self.plane_penalty_scale * plane_dev

            if angle_c < low_angle:
                objective += 2.0 * (low_angle - angle_c)
            elif angle_c > high_angle:
                objective += 2.0 * (angle_c - high_angle)

            if objective < best_objective - 1e-6 or (abs(objective - best_objective) <= 1e-6 and weight < best_weight):
                best_weight = float(weight)
                best_angle = float(angle_c)
                best_objective = float(objective)
                best_plane_dev = float(plane_dev)

        return best_weight, best_angle, best_objective, best_plane_dev

    def build_final_angle_sequence(self) -> list[float]:
        feats = self._extract_features()
        if not feats:
            self._last_extra = {
                "num_valid_frames": 0,
                "num_candidate_frames": 0,
                "num_candidate_runs": 0,
                "num_repaired_runs": 0,
                "num_repaired_frames": 0,
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
        typical_motion = float(np.median(angle_delta)) if angle_delta.size > 0 else 1.0
        local_margin = max(1.0, self.envelope_margin_scale * typical_motion)

        angle_impulse = np.zeros(n, dtype=float)
        thigh_excess = np.zeros(n, dtype=float)
        shank_excess = np.zeros(n, dtype=float)
        segment_excess = np.zeros(n, dtype=float)
        squat_monotonic_violation = np.zeros(n, dtype=float)
        squat_bottom_entry_violation = np.zeros(n, dtype=float)
        squat_violation = np.zeros(n, dtype=float)
        velocity_flip = np.zeros(n, dtype=bool)

        for i in range(1, n - 1):
            prev_feat = feats[i - 1]
            curr_feat = feats[i]
            next_feat = feats[i + 1]

            dt_full = max(float(next_feat.frame_index - prev_feat.frame_index), 1.0)
            alpha = float((curr_feat.frame_index - prev_feat.frame_index) / dt_full)
            expected_angle = (1.0 - alpha) * prev_feat.raw_angle + alpha * next_feat.raw_angle
            angle_impulse[i] = abs(curr_feat.raw_angle - expected_angle)

            thigh_excess[i] = _segment_excess(prev_feat.u_thigh, curr_feat.u_thigh, next_feat.u_thigh)
            shank_excess[i] = _segment_excess(prev_feat.u_shank, curr_feat.u_shank, next_feat.u_shank)
            segment_excess[i] = max(thigh_excess[i], shank_excess[i])

            delta_prev = curr_feat.raw_angle - prev_feat.raw_angle
            delta_next = next_feat.raw_angle - curr_feat.raw_angle
            velocity_flip[i] = (delta_prev < 0.0 and delta_next > 0.0) or (delta_prev > 0.0 and delta_next < 0.0)

            # Squat-specific structural violations.
            # descent_late에서는 angle이 대체로 계속 감소해야 하므로,
            # local positive bump / sign-flip / saw-tooth entry를 이상 신호로 본다.
            bump_up = max(0.0, curr_feat.raw_angle - min(prev_feat.raw_angle, next_feat.raw_angle))
            sign_flip_mag = 0.0
            if delta_prev * delta_next < 0.0:
                sign_flip_mag = min(abs(delta_prev), abs(delta_next))
            squat_monotonic_violation[i] = max(bump_up, sign_flip_mag)

            if delta_prev < 0.0 and delta_next > 0.0:
                # bottom 진입 직전의 spurious rebound를 더 강하게 본다.
                squat_bottom_entry_violation[i] = abs(delta_next) + 0.5 * abs(delta_prev)
            else:
                squat_bottom_entry_violation[i] = 0.0

            squat_violation[i] = max(squat_monotonic_violation[i], squat_bottom_entry_violation[i])

        descent_late_mask = np.asarray([s == "descent_late" for s in subphases], dtype=bool)
        descent_late_idx = np.flatnonzero(descent_late_mask)

        if descent_late_idx.size == 0:
            self._last_extra = {
                "num_valid_frames": int(n),
                "num_candidate_frames": 0,
                "num_candidate_runs": 0,
                "num_repaired_runs": 0,
                "num_repaired_frames": 0,
                "repair_rate": 0.0,
                "raw_phase_counts": {
                    "top": int(sum(p == "top" for p in phases)),
                    "descent": int(sum(p == "descent" for p in phases)),
                    "bottom": int(sum(p == "bottom" for p in phases)),
                    "ascent": int(sum(p == "ascent" for p in phases)),
                    "descent_late": 0,
                },
            }
            return raw_angles.tolist()

        impulse_thr = _safe_quantile(angle_impulse[descent_late_mask], self.candidate_impulse_quantile, default=np.inf)
        segment_thr = _safe_quantile(segment_excess[descent_late_mask], self.candidate_segment_quantile, default=np.inf)
        vis_thr = _safe_quantile(min_vis[descent_late_mask], self.candidate_visibility_quantile, default=0.0)
        squat_violation_thr = _safe_quantile(
            squat_violation[descent_late_mask], self.candidate_squat_violation_quantile, default=np.inf
        )

        low_vis_mask = min_vis <= vis_thr
        strong_segment_mask = segment_excess >= segment_thr
        strong_squat_violation_mask = squat_violation >= squat_violation_thr
        evidence_mask = strong_segment_mask | low_vis_mask | velocity_flip
        candidate_mask = descent_late_mask & (angle_impulse >= impulse_thr) & strong_squat_violation_mask & evidence_mask

        candidate_runs = _group_candidate_runs(
            candidate_mask=candidate_mask,
            max_hole=self.candidate_hole_tolerance,
            max_run_len=self.max_run_len,
        )

        final_angles = raw_angles.astype(float).copy()
        repaired_mask = np.zeros(n, dtype=bool)
        repaired_weights = np.zeros(n, dtype=float)
        plane_dev_before = np.zeros(n, dtype=float)
        plane_dev_after = np.zeros(n, dtype=float)
        run_records: list[dict] = []

        for run_start, run_end in candidate_runs:
            left_idx, right_idx = self._find_run_anchors(candidate_mask, run_start, run_end)
            if left_idx is None or right_idx is None:
                run_records.append(
                    {
                        "start_frame": int(feats[run_start].frame_index),
                        "end_frame": int(feats[run_end].frame_index),
                        "length": int(run_end - run_start + 1),
                        "status": "skipped_missing_anchor",
                    }
                )
                continue

            anchor_indices = self._collect_anchor_context(left_idx, right_idx, n)
            n_ref = self._estimate_reference_plane_normal(feats, anchor_indices)

            run_slice = slice(run_start, run_end + 1)
            off_thigh = np.array([segment_plane_deviation(feats[i].u_thigh, n_ref) for i in range(run_start, run_end + 1)], dtype=float)
            off_shank = np.array([segment_plane_deviation(feats[i].u_shank, n_ref) for i in range(run_start, run_end + 1)], dtype=float)

            mean_thigh_score = float(np.mean(thigh_excess[run_slice] + self.lambda_offplane * off_thigh))
            mean_shank_score = float(np.mean(shank_excess[run_slice] + self.lambda_offplane * off_shank))
            dominant_segment = "thigh" if mean_thigh_score >= mean_shank_score else "shank"

            left_feat = feats[left_idx]
            right_feat = feats[right_idx]
            low_angle = min(left_feat.raw_angle, right_feat.raw_angle) - local_margin
            high_angle = max(left_feat.raw_angle, right_feat.raw_angle) + local_margin

            raw_objective = 0.0
            repaired_objective = 0.0
            proposed_angles: list[float] = []
            proposed_weights: list[float] = []
            proposed_plane_dev: list[float] = []

            for i in range(run_start, run_end + 1):
                feat = feats[i]
                dt_full = max(float(right_feat.frame_index - left_feat.frame_index), 1.0)
                alpha = float((feat.frame_index - left_feat.frame_index) / dt_full)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                expected_angle = (1.0 - alpha) * left_feat.raw_angle + alpha * right_feat.raw_angle

                if dominant_segment == "thigh":
                    raw_plane_dev = segment_plane_deviation(feat.u_thigh, n_ref)
                else:
                    raw_plane_dev = segment_plane_deviation(feat.u_shank, n_ref)
                plane_dev_before[i] = raw_plane_dev
                raw_obj = abs(feat.raw_angle - expected_angle) + self.plane_penalty_scale * raw_plane_dev
                raw_objective += raw_obj

                target_dir = self._build_hinge_target_direction(
                    dominant_segment=dominant_segment,
                    feat=feat,
                    expected_angle=expected_angle,
                    n_ref=n_ref,
                )

                weight, angle_rep, rep_obj, rep_plane_dev = self._repair_single_frame_hinge(
                    feat=feat,
                    dominant_segment=dominant_segment,
                    target_dir=target_dir,
                    expected_angle=expected_angle,
                    low_angle=low_angle,
                    high_angle=high_angle,
                    n_ref=n_ref,
                )
                repaired_objective += rep_obj
                proposed_angles.append(angle_rep)
                proposed_weights.append(weight)
                proposed_plane_dev.append(rep_plane_dev)

            improvement = raw_objective - repaired_objective
            min_required_gain = self.run_improvement_margin_deg * max(1, run_end - run_start + 1)

            if improvement <= min_required_gain:
                run_records.append(
                    {
                        "start_frame": int(feats[run_start].frame_index),
                        "end_frame": int(feats[run_end].frame_index),
                        "length": int(run_end - run_start + 1),
                        "status": "rejected_small_gain",
                        "dominant_segment": dominant_segment,
                        "raw_objective": float(raw_objective),
                        "repaired_objective": float(repaired_objective),
                        "improvement": float(improvement),
                        "reference_plane_normal": [float(x) for x in n_ref],
                    }
                )
                continue

            for offset, i in enumerate(range(run_start, run_end + 1)):
                final_angles[i] = proposed_angles[offset]
                repaired_mask[i] = abs(proposed_angles[offset] - raw_angles[i]) > 1e-6
                repaired_weights[i] = proposed_weights[offset]
                plane_dev_after[i] = proposed_plane_dev[offset]

            run_records.append(
                {
                    "start_frame": int(feats[run_start].frame_index),
                    "end_frame": int(feats[run_end].frame_index),
                    "length": int(run_end - run_start + 1),
                    "status": "applied",
                    "dominant_segment": dominant_segment,
                    "raw_objective": float(raw_objective),
                    "repaired_objective": float(repaired_objective),
                    "improvement": float(improvement),
                    "mean_weight": float(np.mean(proposed_weights)) if proposed_weights else 0.0,
                    "max_weight": float(np.max(proposed_weights)) if proposed_weights else 0.0,
                    "mean_plane_dev_before": float(np.mean(off_thigh if dominant_segment == "thigh" else off_shank)),
                    "mean_plane_dev_after": float(np.mean(proposed_plane_dev)) if proposed_plane_dev else 0.0,
                    "left_anchor_frame": int(left_feat.frame_index),
                    "right_anchor_frame": int(right_feat.frame_index),
                    "reference_plane_normal": [float(x) for x in n_ref],
                }
            )

        applied_runs = [r for r in run_records if r.get("status") == "applied"]
        candidate_indices = [int(feats[i].frame_index) for i in range(n) if candidate_mask[i]]
        repaired_indices = [int(feats[i].frame_index) for i in range(n) if repaired_mask[i]]

        self._last_extra = {
            "num_valid_frames": int(n),
            "num_candidate_frames": int(candidate_mask.sum()),
            "num_candidate_runs": int(len(candidate_runs)),
            "num_repaired_runs": int(len(applied_runs)),
            "num_repaired_frames": int(repaired_mask.sum()),
            "repair_rate": float(repaired_mask.mean()),
            "phase_params": {
                "phase_smooth_window": self.phase_smooth_window,
                "phase_top_band_ratio": self.phase_top_band_ratio,
                "phase_bottom_band_ratio": self.phase_bottom_band_ratio,
                "phase_motion_eps_ratio": self.phase_motion_eps_ratio,
            },
            "candidate_params": {
                "candidate_impulse_quantile": self.candidate_impulse_quantile,
                "candidate_segment_quantile": self.candidate_segment_quantile,
                "candidate_visibility_quantile": self.candidate_visibility_quantile,
                "candidate_squat_violation_quantile": self.candidate_squat_violation_quantile,
                "candidate_hole_tolerance": self.candidate_hole_tolerance,
                "max_run_len": self.max_run_len,
                "anchor_search_radius": self.anchor_search_radius,
                "anchor_context_radius": self.anchor_context_radius,
                "lambda_offplane": self.lambda_offplane,
                "run_improvement_margin_deg": self.run_improvement_margin_deg,
                "envelope_margin_scale": self.envelope_margin_scale,
                "plane_penalty_scale": self.plane_penalty_scale,
                "local_margin": float(local_margin),
                "impulse_threshold": float(impulse_thr),
                "segment_threshold": float(segment_thr),
                "visibility_threshold": float(vis_thr),
                "squat_violation_threshold": float(squat_violation_thr),
            },
            "candidate_reason_counts": {
                "descent_late_frames": int(descent_late_mask.sum()),
                "angle_impulse_frames": int((descent_late_mask & (angle_impulse >= impulse_thr)).sum()),
                "segment_excess_frames": int((descent_late_mask & strong_segment_mask).sum()),
                "squat_violation_frames": int((descent_late_mask & strong_squat_violation_mask).sum()),
                "monotonic_violation_frames": int((descent_late_mask & (squat_monotonic_violation > 0.0)).sum()),
                "bottom_entry_violation_frames": int((descent_late_mask & (squat_bottom_entry_violation > 0.0)).sum()),
                "low_visibility_frames": int((descent_late_mask & low_vis_mask).sum()),
                "velocity_flip_frames": int((descent_late_mask & velocity_flip).sum()),
            },
            "candidate_frame_indices": candidate_indices,
            "repaired_frame_indices": repaired_indices,
            "run_records": run_records,
            "mean_repair_weight": float(repaired_weights[repaired_mask].mean()) if repaired_mask.any() else 0.0,
            "median_repair_weight": float(np.median(repaired_weights[repaired_mask])) if repaired_mask.any() else 0.0,
            "mean_plane_dev_before": float(plane_dev_before[repaired_mask].mean()) if repaired_mask.any() else 0.0,
            "mean_plane_dev_after": float(plane_dev_after[repaired_mask].mean()) if repaired_mask.any() else 0.0,
            "raw_phase_counts": {
                "top": int(sum(p == "top" for p in phases)),
                "descent": int(sum(p == "descent" for p in phases)),
                "bottom": int(sum(p == "bottom" for p in phases)),
                "ascent": int(sum(p == "ascent" for p in phases)),
                "descent_late": int(sum(s == "descent_late" for s in subphases)),
            },
        }
        return final_angles.tolist()

    def finalize_analysis(self):
        final_angles = self.build_final_angle_sequence()
        final_count = self.recount_from_angles(final_angles)
        return final_count, final_angles, self._last_extra
