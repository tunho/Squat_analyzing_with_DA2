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
    segment_lengths_from_triplet,
)


@dataclass
class WorldFrameFeature:
    frame_index: int
    hip: object
    knee: object
    ankle: object
    u_thigh: np.ndarray
    u_shank: np.ndarray
    thigh_len: float
    shank_len: float
    min_vis: float
    thigh_dir_change: float
    shank_dir_change: float


class MPWorldOutlierCorrectedSquatAnalyzer(BaseWorldKneeAnalyzer):
    def __init__(self, *args, visibility_threshold: float = 0.5, score_threshold: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.visibility_threshold = visibility_threshold
        self.score_threshold = score_threshold

    def _extract_features(self) -> list[WorldFrameFeature]:
        feats = []
        prev_u_thigh = None
        prev_u_shank = None

        for frame in self.history_data:
            triplet = self.get_world_triplet(frame)
            if triplet is None:
                continue
            hip, knee, ankle = triplet

            h = point_to_np(hip)
            k = point_to_np(knee)
            a = point_to_np(ankle)

            v_thigh = h - k
            v_shank = a - k

            u_thigh = safe_unit(v_thigh)
            u_shank = safe_unit(v_shank)

            thigh_len = float(np.linalg.norm(v_thigh))
            shank_len = float(np.linalg.norm(v_shank))
            min_vis = min(hip.vis, knee.vis, ankle.vis)

            if prev_u_thigh is None:
                thigh_dir_change = 0.0
                shank_dir_change = 0.0
            else:
                thigh_dir_change = angle_between_unit_vectors(prev_u_thigh, u_thigh)
                shank_dir_change = angle_between_unit_vectors(prev_u_shank, u_shank)

            feats.append(
                WorldFrameFeature(
                    frame_index=frame.frame_index,
                    hip=hip,
                    knee=knee,
                    ankle=ankle,
                    u_thigh=u_thigh,
                    u_shank=u_shank,
                    thigh_len=thigh_len,
                    shank_len=shank_len,
                    min_vis=min_vis,
                    thigh_dir_change=thigh_dir_change,
                    shank_dir_change=shank_dir_change,
                )
            )

            prev_u_thigh = u_thigh
            prev_u_shank = u_shank

        return feats

    def _detect_outliers(self, feats: list[WorldFrameFeature]) -> list[bool]:
        if not feats:
            return []

        thigh_jumps = [f.thigh_dir_change for f in feats[1:]]
        shank_jumps = [f.shank_dir_change for f in feats[1:]]
        thigh_lens = [f.thigh_len for f in feats]
        shank_lens = [f.shank_len for f in feats]

        thigh_jump_med, thigh_jump_mad = robust_mad(thigh_jumps)
        shank_jump_med, shank_jump_mad = robust_mad(shank_jumps)

        thigh_len_med = float(np.median(thigh_lens))
        shank_len_med = float(np.median(shank_lens))

        thigh_len_dev = [abs(x - thigh_len_med) for x in thigh_lens]
        shank_len_dev = [abs(x - shank_len_med) for x in shank_lens]

        thigh_len_dev_med, thigh_len_dev_mad = robust_mad(thigh_len_dev)
        shank_len_dev_med, shank_len_dev_mad = robust_mad(shank_len_dev)

        outliers = []
        for i, f in enumerate(feats):
            thigh_jump_score = abs(f.thigh_dir_change - thigh_jump_med) / thigh_jump_mad
            shank_jump_score = abs(f.shank_dir_change - shank_jump_med) / shank_jump_mad

            thigh_dev_score = abs(abs(f.thigh_len - thigh_len_med) - thigh_len_dev_med) / thigh_len_dev_mad
            shank_dev_score = abs(abs(f.shank_len - shank_len_med) - shank_len_dev_med) / shank_len_dev_mad

            is_outlier = (
                thigh_jump_score > self.score_threshold
                or shank_jump_score > self.score_threshold
                or thigh_dev_score > self.score_threshold
                or shank_dev_score > self.score_threshold
                or f.min_vis < self.visibility_threshold
            )
            outliers.append(is_outlier)

        return outliers

    def _find_prev_good(self, outliers: list[bool], idx: int) -> int | None:
        for j in range(idx - 1, -1, -1):
            if not outliers[j]:
                return j
        return None

    def _find_next_good(self, outliers: list[bool], idx: int) -> int | None:
        for j in range(idx + 1, len(outliers)):
            if not outliers[j]:
                return j
        return None

    def build_final_angle_sequence(self) -> list[float]:
        feats = self._extract_features()
        if not feats:
            return []

        outliers = self._detect_outliers(feats)

        thigh_len_med = float(np.median([f.thigh_len for f in feats]))
        shank_len_med = float(np.median([f.shank_len for f in feats]))

        final_angles = []

        for i, f in enumerate(feats):
            if not outliers[i]:
                angle = calculate_knee_angle(f.hip, f.knee, f.ankle)
                final_angles.append(angle)
                continue

            prev_idx = self._find_prev_good(outliers, i)
            next_idx = self._find_next_good(outliers, i)

            if prev_idx is None and next_idx is None:
                angle = calculate_knee_angle(f.hip, f.knee, f.ankle)
                final_angles.append(angle)
                continue

            if prev_idx is not None and next_idx is not None:
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                u_thigh_corr = interpolate_unit_vectors(
                    feats[prev_idx].u_thigh,
                    feats[next_idx].u_thigh,
                    alpha,
                )
                u_shank_corr = interpolate_unit_vectors(
                    feats[prev_idx].u_shank,
                    feats[next_idx].u_shank,
                    alpha,
                )
            elif prev_idx is not None:
                u_thigh_corr = feats[prev_idx].u_thigh
                u_shank_corr = feats[prev_idx].u_shank
            else:
                u_thigh_corr = feats[next_idx].u_thigh
                u_shank_corr = feats[next_idx].u_shank

            hip_c, knee_c, ankle_c = rebuild_triplet_from_knee_and_directions(
                knee=f.knee,
                u_thigh=u_thigh_corr,
                u_shank=u_shank_corr,
                thigh_len=thigh_len_med,
                shank_len=shank_len_med,
                ref_hip=f.hip,
                ref_ankle=f.ankle,
            )
            angle = calculate_knee_angle(hip_c, knee_c, ankle_c)
            final_angles.append(angle)

        return final_angles