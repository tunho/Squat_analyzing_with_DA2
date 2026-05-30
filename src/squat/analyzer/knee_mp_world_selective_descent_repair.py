from __future__ import annotations

import numpy as np

from squat.analyzer.knee_base_world import BaseWorldKneeAnalyzer
from squat.geometry.angles import calculate_knee_angle
from squat.geometry.world_knee_ops import (
    point_to_np,
    safe_unit,
    rebuild_triplet_from_knee_and_directions,
    interpolate_unit_vectors,
    project_unit_to_plane,
    average_aligned_normals
)

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

def _gradient(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros_like(values, dtype=float)
    return np.gradient(values.astype(float))

def _label_angle_phases(
    angles: np.ndarray,
    smooth_window: int = 5,
    top_band_ratio: float = 0.15,
    bottom_band_ratio: float = 0.85,
    motion_eps_ratio: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    내부적으로 descent_late 구간을 찾기 위한 라벨링 도우미.
    evaluate_world_vs_gt.py의 phase labeling과 철학을 맞춤.
    """
    if len(angles) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    smooth = _rolling_mean(angles, smooth_window)
    p5 = float(np.percentile(smooth, 5))
    p95 = float(np.percentile(smooth, 95))
    robust_range = max(p95 - p5, 1e-6)

    normalized_depth = (p95 - smooth) / robust_range
    normalized_depth = np.clip(normalized_depth, 0.0, 1.0)

    velocity = _gradient(smooth)
    motion_eps = motion_eps_ratio * robust_range

    phases = []
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
    subphases = []
    for phase, depth in zip(phases, normalized_depth):
        if phase == "descent":
            subphases.append("descent_early" if depth < depth_mid else "descent_late")
        else:
            subphases.append("")

    return np.array(phases), np.array(subphases), normalized_depth, velocity

class MPWorldSelectiveDescentRepairAnalyzer(BaseWorldKneeAnalyzer):
    """
    mp_world_raw를 baseline으로 유지하면서,
    오차가 가장 큰 descent_late 구간 내의 spike/instability만 기하학적으로 고치는 analyzer.
    고정 상수(hardcoded threshold)를 지양하고 현 영상의 분포(Distribution)를 참고합니다.
    """
    
    def __init__(
        self,
        *args,
        phase_smooth_window: int = 5,
        acc_spike_percentile: float = 80.0,
        dir_jump_percentile: float = 80.0,
        max_run_len: int = 10,
        repair_blend_factor: float = 0.8, # 0.8:보정값 중심, 0.2:원본값 중심
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.phase_smooth_window = phase_smooth_window
        self.acc_spike_percentile = acc_spike_percentile
        self.dir_jump_percentile = dir_jump_percentile
        self.max_run_len = max_run_len
        self.repair_blend_factor = repair_blend_factor
        self._last_extra = {}

    def _estimate_global_sagittal_normal(
        self,
        hips: list[np.ndarray],
        knees: list[np.ndarray],
        ankles: list[np.ndarray],
        subphases: np.ndarray,
    ) -> np.ndarray:
        """
        Stage 1: PCA 기반 전역 사지탈 평면(Sagittal Plane)의 법선(Normal) 추출
        가장 불안한 구간(descent_late)을 제외한 나머지 뼈대 벡터들로 PCA를 돌려 평면을 찾는다.
        """
        valid_vecs = []
        for i, sp in enumerate(subphases):
            if sp != "descent_late":
                valid_vecs.append(knees[i] - hips[i])
                valid_vecs.append(ankles[i] - hips[i])
                
        if len(valid_vecs) < 10:
            for i in range(len(hips)):
                valid_vecs.append(knees[i] - hips[i])
                valid_vecs.append(ankles[i] - hips[i])
                
        valid_vecs = np.array(valid_vecs)
        if len(valid_vecs) == 0:
            return np.array([0.0, 0.0, 1.0])
            
        centered = valid_vecs - np.mean(valid_vecs, axis=0)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        return safe_unit(vt[-1])

    def build_final_angle_sequence(self) -> list[float]:
        if len(self.history_data) == 0:
            return []

        raw_angles = []
        u_thighs = []
        u_shanks = []
        bone_lens = [] # (thigh_len, shank_len)
        knees = []
        hips = []
        ankles = []
        
        for frame in self.history_data:
            triplet = self.get_world_triplet(frame)
            if triplet is None:
                # Handle missing frames by repeating last valid to keep indices aligned
                if len(raw_angles) > 0:
                    raw_angles.append(raw_angles[-1])
                    u_thighs.append(u_thighs[-1])
                    u_shanks.append(u_shanks[-1])
                    bone_lens.append(bone_lens[-1])
                    knees.append(knees[-1])
                    hips.append(hips[-1])
                    ankles.append(ankles[-1])
                continue
            hip, knee, ankle = triplet
            
            p_h = point_to_np(hip)
            p_k = point_to_np(knee)
            p_a = point_to_np(ankle)
            
            thigh_vec = p_h - p_k
            shank_vec = p_a - p_k
            thigh_len = float(np.linalg.norm(thigh_vec))
            shank_len = float(np.linalg.norm(shank_vec))
            
            u_t = safe_unit(thigh_vec)
            u_s = safe_unit(shank_vec)
            
            raw_angles.append(calculate_knee_angle(hip, knee, ankle))
            u_thighs.append(u_t)
            u_shanks.append(u_s)
            bone_lens.append((thigh_len, shank_len))
            knees.append(knee)
            hips.append(hip)
            ankles.append(ankle)
            
        n = len(raw_angles)
        if n < 10:
            return raw_angles

        raw_angles = np.array(raw_angles)
        
        # 1. Initial Phase Labeling to find stable frames for PCA
        phases, subphases, _, _ = _label_angle_phases(raw_angles, smooth_window=self.phase_smooth_window)
        
        # =========================================================
        # STAGE 1: Global Sagittal Plane Projection (Fixing the MAE Gap)
        # =========================================================
        hips_np = [point_to_np(h) for h in hips]
        knees_np = [point_to_np(k) for k in knees]
        ankles_np = [point_to_np(a) for a in ankles]
        
        global_normal = self._estimate_global_sagittal_normal(hips_np, knees_np, ankles_np, subphases)
        
        global_angles = []
        for i in range(n):
            u_t_proj = project_unit_to_plane(u_thighs[i], global_normal)
            u_s_proj = project_unit_to_plane(u_shanks[i], global_normal)
            
            h_rep, k_rep, a_rep = rebuild_triplet_from_knee_and_directions(
                knee=knees[i],
                u_thigh=u_t_proj,
                u_shank=u_s_proj,
                thigh_len=bone_lens[i][0],
                shank_len=bone_lens[i][1],
                ref_hip=hips[i],
                ref_ankle=ankles[i]
            )
            
            global_angles.append(calculate_knee_angle(h_rep, k_rep, a_rep))
            
            # Update base data so Stage 2 works on the newly projected clean plane
            u_thighs[i] = u_t_proj
            u_shanks[i] = u_s_proj
            knees[i] = k_rep
            hips[i] = h_rep
            ankles[i] = a_rep
            
        global_angles = np.array(global_angles)
        
        # Re-evaluate phases after Stage 1 (angles are now strictly on-plane)
        phases, subphases, _, _ = _label_angle_phases(global_angles, smooth_window=self.phase_smooth_window)
        
        # =========================================================
        # STAGE 3: Kinematic Hysteresis Correction (Monotonic Gap Repair)
        # =========================================================
        from itertools import groupby
        
        # 무릎 각도는 MediaPipe가 속이고 있을 수 있으므로(Lag/Frozen),
        # 엉덩이(Hip)의 수직 위치를 기준으로 '물리적인 깊이'를 판단합니다.
        hips_y = np.array([h[1] for h in hips_np]) # y is vertical in world
        y_min, y_max = np.min(hips_y), np.max(hips_y)
        y_range = max(y_max - y_min, 1e-6)
        # 엉덩이가 낮을수록 depth는 커짐 (y_max: 서있을 때, y_min: 앉았을 때)
        hip_depth = (y_max - hips_y) / y_range
        
        # 엉덩이가 중간 이상 내려갔는데 하강 중인 구간을 '보정Target'으로 정의 (Descent Late proxy)
        is_descent = (np.array(phases) == "descent")
        target_mask = is_descent & (hip_depth > 0.4)
        
        dl_runs = []
        for k, g in groupby(enumerate(target_mask), key=lambda x: x[1]):
            if k:
                run = list(g)
                dl_runs.append((run[0][0], run[-1][0]))
                
        # =============================================================
        # STAGE 3: Hip Depth-Conditioned Angle Scaling
        # =============================================================
        # MediaPipe의 '바닥 과소평가 오류'는 랜덤하지 않고 체계적(Systematic)입니다.
        # 엉덩이가 깊이 내려갈수록 과소평가 크기가 선형적으로 증가합니다.
        # 이 과소평가 Bias(체계적 편차)를 영상 자체 데이터로 추정하여 프레임별 보정합니다.

        # 1. Stable anchors: Top(서있을때) 와 Bottom(가장 깊이) 구간의 평균 각도 계산
        phases_arr = np.array(phases)
        top_mask = (phases_arr == "top")
        bottom_mask = (phases_arr == "bottom")

        if top_mask.any() and bottom_mask.any():
            angle_at_top = float(np.median(global_angles[top_mask]))
            angle_at_bottom = float(np.median(global_angles[bottom_mask]))
            # 2. angle_range: MediaPipe가 top에서 bottom까지 관찰하는 각도 범위
            mp_angle_range = max(angle_at_top - angle_at_bottom, 1.0)
            # 3. hip_depth_range: MP 추정 각도 범위와 엉덩이 높이 범위의
            #    비율(ratio)을 이용해 예상 '진짜' 각도 범위를 추정
            # - 사용 가능한 정답이 없으므로, 여기서는 Stage1+2 보정 후 내부 분포를 활용
            #   hip_depth 0→1에 따라 각도가 angle_at_top → (angle_at_top - mp_angle_range*1.35) 로
            #   내려가야 한다고 가정 (1.35배 스케일: 35% 하방 추가 보정)
            depth_scale_factor = 1.35  # 공격적인 하방 보정 (1.15 -> 1.35)
        else:
            depth_scale_factor = 1.0
            angle_at_top = float(np.max(global_angles))
            mp_angle_range = 1.0

        # 4. 프레임별 Hip Depth-Conditioned 보정 적용
        bridged_angles = global_angles.copy()
        stage3_repaired_frames = 0

        is_motion = (phases_arr == "descent") | (phases_arr == "bottom") | (phases_arr == "ascent")

        for i in range(n):
            if not is_motion[i]:
                continue

            d = float(hip_depth[i])  # 0(서있음) ~ 1(바닥)
            if d < 0.3:              # 엉덩이가 아직 충분히 내려오지 않은 구간은 건드리지 않음
                continue

            # sigmoid weight: depth가 0.25부터 보정 시작, 0.55 중심, 0.95에서 최대
            w_depth = 1.0 / (1.0 + np.exp(-12.0 * (d - 0.55)))

            a_raw = bridged_angles[i]
            # 이 깊이에서 BaseAngle 대비 'MediaPipe가 예상보다 덜 굽힌 양'을 추정
            # (depth_scale_factor - 1.0)를 깊이에 비례해 누적 보정
            expected_drop = (angle_at_top - a_raw) * (depth_scale_factor - 1.0)
            a_corrected = a_raw - w_depth * expected_drop

            # 보정 적용: 더 깊어진(각도 작아진) 경우만 반영 (과보정 방지)
            if a_corrected < a_raw - 0.5:
                # 기하학적으로 보정: 현재 bone direction을 Slerp로 살짝 더 굽힘
                gap = a_raw - a_corrected
                w_geom = np.clip(gap / 20.0, 0.0, 1.0)  # 상한선을 1.0으로 해제하여 100% 보정 반영

                # anchor end: lookahead에서 최소 각도 지점 찾기
                lookahead_end = min(n, i + 90)
                if lookahead_end > i + 2:
                    future_slice = bridged_angles[i:lookahead_end]
                    p_end = i + int(np.argmin(future_slice))
                else:
                    p_end = min(n - 1, i + 5)

                if p_end > i:
                    alpha = 0.5  # 중간점 방향으로 Slerp
                    u_t_lerp = interpolate_unit_vectors(u_thighs[i], u_thighs[p_end], alpha)
                    u_s_lerp = interpolate_unit_vectors(u_shanks[i], u_shanks[p_end], alpha)

                    u_t_final = interpolate_unit_vectors(u_thighs[i], u_t_lerp, w_geom)
                    u_s_final = interpolate_unit_vectors(u_shanks[i], u_s_lerp, w_geom)

                    h_rep_f, k_rep_f, a_rep_f = rebuild_triplet_from_knee_and_directions(
                        knee=knees[i], u_thigh=u_t_final, u_shank=u_s_final,
                        thigh_len=bone_lens[i][0], shank_len=bone_lens[i][1],
                        ref_hip=hips[i], ref_ankle=ankles[i]
                    )

                    new_angle = calculate_knee_angle(h_rep_f, k_rep_f, a_rep_f)
                    if new_angle < a_raw - 0.5:  # 실제로 각도가 줄어든 경우만 저장
                        bridged_angles[i] = new_angle
                        u_thighs[i] = u_t_final
                        u_shanks[i] = u_s_final
                        knees[i] = k_rep_f
                        hips[i] = h_rep_f
                        ankles[i] = a_rep_f
                        stage3_repaired_frames += 1
                    
        # Re-evaluate phases after Stage 3 (Bottom gap is fixed)
        phases, subphases, _, _ = _label_angle_phases(bridged_angles, smooth_window=self.phase_smooth_window)
        descent_late_mask = (subphases == "descent_late")
        
        # =========================================================
        # STAGE 2: Selective Soft-Blending Repair (Fixing the Jitter/Spikes)
        # =========================================================
        
        # 2. Dynamics Feature (Angle Acceleration as Spike proxy)
        vel = _gradient(bridged_angles)
        acc = np.abs(_gradient(vel))
        
        # 3. Geometric Feature (Unit Vector Directional Jump)
        dir_change = np.zeros(n)
        for i in range(1, n):
            dt = np.arccos(np.clip(np.dot(u_thighs[i], u_thighs[i-1]), -1.0, 1.0))
            ds = np.arccos(np.clip(np.dot(u_shanks[i], u_shanks[i-1]), -1.0, 1.0))
            dir_change[i] = max(dt, ds)
            
        # 4. Data-driven Thresholds (based on descent_late behavior)
        if not np.any(descent_late_mask):
            return raw_angles.tolist()
            
        acc_descent = acc[descent_late_mask]
        dir_descent = dir_change[descent_late_mask]
        
        acc_thr = np.percentile(acc_descent, self.acc_spike_percentile) if len(acc_descent) > 0 else np.inf
        dir_thr = np.percentile(dir_descent, self.dir_jump_percentile) if len(dir_descent) > 0 else np.inf
        
        # Target Candidates
        is_spike = (acc > acc_thr) | (dir_change > dir_thr)
        candidate_mask = descent_late_mask & is_spike
        
        # 5. Extract Runs (Consecutive abnormal frames)
        runs = []
        idxs = np.where(candidate_mask)[0]
        if len(idxs) > 0:
            start = idxs[0]
            prev = idxs[0]
            for idx in idxs[1:]:
                if idx - prev <= 2: # Allow small hole (1 frame)
                    prev = idx
                else:
                    if prev - start + 1 <= self.max_run_len:
                        runs.append((start, prev))
                    start = idx
                    prev = idx
            if prev - start + 1 <= self.max_run_len:
                runs.append((start, prev))
                
        final_angles = bridged_angles.copy()
        repaired_frames = 0
        
        for run_start, run_end in runs:
            # Find Anchors (1-2 frames left and right)
            left_anchor = max(0, run_start - 1)
            right_anchor = min(n - 1, run_end + 1)
            
            # Require clean anchors (not in candidate mask)
            if candidate_mask[left_anchor] or candidate_mask[right_anchor]:
                continue
                
            # Estimate Local Reference Plane
            normals = []
            for i in [left_anchor, right_anchor]:
                n_i = safe_unit(np.cross(u_thighs[i], u_shanks[i]))
                if np.linalg.norm(n_i) > 1e-8:
                    normals.append(n_i)
            if not normals:
                continue
            n_ref = average_aligned_normals(normals)
            
            # Geometric Repair via Slerp & Projection
            dt_full = right_anchor - left_anchor
            if dt_full == 0: dt_full = 1
            
            for i in range(run_start, run_end + 1):
                alpha = (i - left_anchor) / dt_full
                
                # Slerp approximation (interpolate + normalize)
                u_t_interp = interpolate_unit_vectors(u_thighs[left_anchor], u_thighs[right_anchor], alpha)
                u_s_interp = interpolate_unit_vectors(u_shanks[left_anchor], u_shanks[right_anchor], alpha)
                
                # Project back to local hinge plane to enforce geometric stability
                u_t_geom = project_unit_to_plane(u_t_interp, n_ref)
                u_s_geom = project_unit_to_plane(u_s_interp, n_ref)
                
                # Soft Blend: 원본의 위치 특성(MAE)은 어느 정도 유지하면서 Spike(Velocity)만 깎아냄
                u_t_rep = interpolate_unit_vectors(u_thighs[i], u_t_geom, self.repair_blend_factor)
                u_s_rep = interpolate_unit_vectors(u_shanks[i], u_s_geom, self.repair_blend_factor)
                
                # Maintain original bone lengths from the raw triplet
                h_rep, k_rep, a_rep = rebuild_triplet_from_knee_and_directions(
                    knee=knees[i],
                    u_thigh=u_t_rep,
                    u_shank=u_s_rep,
                    thigh_len=bone_lens[i][0],
                    shank_len=bone_lens[i][1],
                    ref_hip=hips[i],
                    ref_ankle=ankles[i]
                )
                
                final_angles[i] = calculate_knee_angle(h_rep, k_rep, a_rep)
                repaired_frames += 1

        # =========================================================
        # STAGE 4: Temporal Lag Correction (Cross-Correlation 기반 위상 정렬)
        # =========================================================
        # MediaPipe는 실제 동작보다 수십 프레임 늦게 무릎 각도를 업데이트(Lag)합니다.
        # 이로 인해 파란 정답선과 주황 예측선 사이에 '시간적 위상 차이'가 발생합니다.
        # 엉덩이 하강 속도(Ground Truth for motion)와 각도 변화 속도를 Cross-Correlation하여
        # 최적 Lag를 자동으로 계산하고 보정합니다.

        # 위상 신호: 엉덩이 수직 속도 (내려갈수록 양수)
        hip_vel = -np.gradient(hips_y)  # 내려가면 +, 올라가면 -
        # 위상 신호: 무릎 각도 변화 속도 (굽혀지면 음수)
        angle_vel = np.gradient(final_angles)

        # 탐색 범위: -60 ~ +60 프레임 (최대 약 2초)
        max_lag_search = 60
        best_lag = 0

        if len(hip_vel) > max_lag_search * 2:
            # hip_vel이 크고(내려가는 구간), angle_vel이 음수(굽혀지는 구간) 인 상관관계 탐색
            # angle_vel에 -1을 곱해서 '굽혀지는 신호' → 양수 방향으로 통일
            sig_a = hip_vel - np.mean(hip_vel)
            sig_b = -(angle_vel - np.mean(angle_vel))

            # 정규화
            norm_a = np.linalg.norm(sig_a)
            norm_b = np.linalg.norm(sig_b)
            if norm_a > 1e-6 and norm_b > 1e-6:
                sig_a /= norm_a
                sig_b /= norm_b

                # Cross-Correlation 계산 (numpy Full mode)
                corr = np.correlate(sig_a, sig_b, mode='full')
                # corr의 center index
                center = len(sig_b) - 1
                # 탐색 범위 내에서 최대 상관 위치 찾기
                search_corr = corr[center - max_lag_search: center + max_lag_search + 1]
                best_lag = int(np.argmax(search_corr)) - max_lag_search
                # best_lag > 0: angle이 hip보다 늦음 → 앞당겨야 함
                # best_lag < 0: angle이 hip보다 빠름 → 뒤로 미뤄야 함

        # Lag 보정 적용 (시퀀스 전체를 best_lag만큼 shift)
        lag_corrected_angles = final_angles.copy()
        if best_lag != 0:
            if best_lag > 0:
                # 앞당기기: 뒤의 값을 앞으로
                lag_corrected_angles[:n - best_lag] = final_angles[best_lag:]
                lag_corrected_angles[n - best_lag:] = final_angles[-1]  # 끝부분 edge fill
            else:
                # 뒤로 밀기: 앞의 값을 뒤로
                shift = -best_lag
                lag_corrected_angles[shift:] = final_angles[:n - shift]
                lag_corrected_angles[:shift] = final_angles[0]  # 앞부분 edge fill

        # =========================================================
        # STAGE 5: Squat Cycle-Level Smooth Trough Enforcement
        # =========================================================
        # 스쿼트 각 사이클의 바닥(trough) 지점을 엉덩이 높이 기준으로 감지하고,
        # 하강 구간에서 "현재 각도 vs 선형 보간된 바닥을 향한 목표 각도" 중 더 작은 쪽을 선택합니다.
        # 이렇게 하면 MediaPipe의 Flat-line 구간에서 자연스럽게 바닥을 향해 각도가 내려갑니다.

        from scipy.signal import find_peaks

        hip_y_smooth = _rolling_mean(hips_y, window=15)
        inv_hip = -hip_y_smooth
        prom_thr = 0.25 * (np.max(inv_hip) - np.min(inv_hip))
        troughs, _ = find_peaks(inv_hip, prominence=prom_thr, distance=20)

        monotonic_angles = np.array(lag_corrected_angles, dtype=float)

        if len(troughs) > 0:
            for trough_idx in troughs:
                # 하강 시작점: trough 이전 최대 hip_y (가장 서있는 지점) 탐색
                search_back = max(0, trough_idx - 150)
                pre_seg = hip_y_smooth[search_back:trough_idx]
                if len(pre_seg) == 0:
                    continue
                local_peak_offset = int(np.argmax(pre_seg))
                desc_start = search_back + local_peak_offset

                # 상승 끝점: trough 이후 최대 hip_y 탐색
                search_fwd = min(n, trough_idx + 150)
                post_seg = hip_y_smooth[trough_idx:search_fwd]
                if len(post_seg) == 0:
                    continue
                local_asc_offset = int(np.argmax(post_seg))
                asc_end = trough_idx + local_asc_offset

                trough_angle = float(monotonic_angles[trough_idx])

                # --- 하강 보정: desc_start → trough_idx ---
                dt_desc = trough_idx - desc_start
                if dt_desc > 2:
                    start_angle = float(monotonic_angles[desc_start])
                    for k in range(dt_desc + 1):
                        alpha = k / dt_desc
                        # 선형 보간: 시작각도 → trough 각도
                        interp_angle = start_angle + alpha * (trough_angle - start_angle)
                        idx = desc_start + k
                        # 현재 각도보다 더 작게(깊게) 내려가는 경우만 적용
                        if interp_angle < monotonic_angles[idx]:
                            monotonic_angles[idx] = interp_angle

                # --- 상승 보정: trough_idx → asc_end ---
                dt_asc = asc_end - trough_idx
                if dt_asc > 2:
                    end_angle = float(monotonic_angles[asc_end])
                    for k in range(dt_asc + 1):
                        alpha = k / dt_asc
                        # 선형 보간: trough 각도 → 끝 각도
                        interp_angle = trough_angle + alpha * (end_angle - trough_angle)
                        idx = trough_idx + k
                        # 현재 각도보다 더 크게(펴지는) 경우만 적용  
                        if interp_angle > monotonic_angles[idx]:
                            monotonic_angles[idx] = interp_angle

        # 최종 가벼운 스무딩 (3프레임): 불연속성 제거
        monotonic_angles = _rolling_mean(monotonic_angles, window=3)


        self._last_extra = {
            "num_valid_frames": n,
            "num_descent_late_frames": int(np.sum(descent_late_mask)),
            "acc_threshold": float(acc_thr),
            "dir_threshold": float(dir_thr),
            "num_candidate_frames": int(np.sum(candidate_mask)),
            "num_repaired_runs": len(runs),
            "num_repaired_frames_stage2": repaired_frames,
            "num_repaired_frames_stage3": stage3_repaired_frames,
            "temporal_lag_frames": int(best_lag),
            "num_squat_cycles_detected": int(len(troughs)),
            "repair_rate": float((repaired_frames+stage3_repaired_frames) / n) if n > 0 else 0.0
        }

        return monotonic_angles.tolist()



    def finalize_analysis(self):
        final_angles = self.build_final_angle_sequence()
        final_count = self.recount_from_angles(final_angles)
        return final_count, final_angles, self._last_extra
