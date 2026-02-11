import cv2
import numpy as np
import mediapipe as mp
import time
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pose_estimation import PoseEstimator

# Inline DepthEstimator class definition to avoid import issues
class DepthEstimator:
    def __init__(self):
        print("Loading Depth Anything V2-Small...")
        self.model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device)
        print("Depth Anything Model Loaded.")

class SquatAnalyzer:
    def __init__(self):
        # MediaPipe Pose 초기화
        # PoseEstimator의 process_frame은 시각화용 더미 객체를 반환하므로,
        # 분석을 위해 extract_keypoints_only를 사용하고 시각화는 별도로 처리함.
        self.pose_estimator = PoseEstimator(static_image_mode=False, model_complexity=1)
        
        # Depth Anything V2 초기화 - ENABLED
        print("Loading Depth Anything V2-Small...")
        self.depth_model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained(self.depth_model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_model_id).to(self.device)
        print("Depth Anything Model Loaded.")

        # 상태 관리 (FSM)
        self.state = "s1" # s1: Standing, s2: Transition, s3: Bottom
        self.state_sequence = [] # 상태 변화 기록용 리스트
        
        self.counter = 0
        self.feedback = ""
        self.veto_reason = None # 자세 불량 시 카운트 방지용 플래그
        self.had_form_issue = False 
        self.issue_types = set() # 어떤 문제들이 있었는지 기록 (BACK, BALANCE, KNEE)
        self.issue_states = set() # 어떤 상태(s2, s3)에서 문제가 발생했는지 기록
        self.has_hit_s3 = False # s3(Bottom) 지점에 도달했었는지 여부 (더 유연한 카운팅용)
        
        # [Academic Standard with Perspective Scaling]
        self.TH_S1_LIMIT = 35.0 # Strict Standing Criteria
        self.TH_S3_MIN = 75.73  # Paper Standard
        self.TH_S3_MAX = 93.89
        
        # Posture Issue Thresholds
        self.TRUNK_LIMIT = 35.0 # Strict Form
        self.ALIGN_LIMIT = 20.0 # Strict Balance
        self.KNEE_LIMIT = 0.10  # Knee Forward
        
        # Side Selection (Auto-assigned per video)
        self.side = None # "left" or "right", initialized to None for auto-selection
        
        # [Equation 1] 1RM Calculation Config
        self.weight = 20.0 # Default weight (Empty Barbell) in kg
        self.one_rm = 0.0
        
        # Calibration Variables
        self.calibration_frames = 0
        self.calib_thigh_sum = 0.0
        self.calib_trunk_sum = 0.0
        self.thigh_bias = 0.0
        self.trunk_bias = 0.0
        self.is_calibrated = False

        # --- RAZS (Robust Adaptive Z-Scaling) Initial State ---
        self.z_scale_param = 1.0  # The "Magic Number" k
        self.base_k = None # [Hybrid RAZS] Learned base scale
        self.razs_stable = False   # Flag if scale has converged somewhat
        self.max_shank_len_2d = 0.0
        self.max_thigh_len_2d = 0.0 # [NEW] Learning User Proportions
        self.max_torso_len_2d = 0.0 # [NEW] Learning User Proportions

    def reset(self):
        """Reset state for batch processing"""
        self.state = "s1"
        self.state_sequence = []
        self.counter = 0
        self.feedback = ""
        self.veto_reason = None
        self.had_form_issue = False
        self.issue_types = set()
        self.issue_states = set()
        self.has_hit_s3 = False
        self.one_rm = 0.0
        self.issue_states = set()
        self.has_hit_s3 = False
        self.one_rm = 0.0
        self.side = None 
        
        # Calibration Variables
        self.calibration_frames = 0
        self.calib_thigh_sum = 0.0
        self.calib_trunk_sum = 0.0
        self.thigh_bias = 0.0
        self.trunk_bias = 0.0
        self.is_calibrated = False
        
        # RAZS State Reset
        self.z_scale_param = 1.0
        self.razs_stable = False
        self.max_shank_len_2d = 0.0 
        self.max_thigh_len_2d = 0.0 
        self.max_torso_len_2d = 0.0 
        
        # [NEW] 2-Pass History storage
        self.history_data = [] # Stores (landmarks, depth_val) for re-evaluation

    @staticmethod
    def calculate_vertical_angle(p1, p2):
        """
        수직선(Gravity Vector, 0, 1)에 대한 p1-p2 벡터의 각도 계산 (2D Only).
        Hip(위) -> Knee(아래) 벡터와 수직선(0, 1: 아래 방향) 사이의 각도를 구함.
        Input: {'x': px, 'y': py, ...}
        """
        delta_x = p2['x'] - p1['x']
        delta_y = p2['y'] - p1['y']
        
        vec_len = np.sqrt(delta_x**2 + delta_y**2)
        if vec_len == 0:
            return 0
            
        # (dx, dy) . (0, 1) = dy
        dot_product = delta_y
        
        cosine_angle = dot_product / vec_len
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def process_video(self, video_path, output_path="output.mp4", depth_estimator=None, show_window=False, override_params=None):
        # Reset per video
        self.reset()
        
        # [2-PASS SUPPORT] Apply override parameters if provided
        if override_params:
            self.max_shank_len_2d = override_params.get('max_shank_len_2d', 0)
            self.max_thigh_len_2d = override_params.get('max_thigh_len_2d', 0)
            self.max_torso_len_2d = override_params.get('max_torso_len_2d', 0)
            if 'base_k' in override_params and override_params['base_k'] is not None:
                self.base_k = override_params['base_k']
                self.z_scale_param = self.base_k
                print(f"  [2-PASS] ACTIVE. Fixed Base K: {self.base_k:.4f}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return 0, False, [], []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Handle corrupted metadata (e.g. 1000.0 fps bug)
        if fps > 100 or fps <= 0:
            print(f"Warning: Suspicious FPS {fps} detected. Forcing to 30.0 for stable output.")
            fps = 30.0
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video Info: {w}x{h} @ {fps}fps")
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        print(f"Processing {video_path}...")
        
        frame_idx = 0
        w_draw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_draw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # 1. Pose Estimation
            # Note: MediaPipe sets input image to read-only. Pass a copy.
            landmarks_list, results = self.pose_estimator.extract_keypoints_only(frame.copy())
            image_draw = frame.copy()
            
            if landmarks_list:
                # 2. Depth Anything Inference (Hybrid Mode: Use for Z-diff only)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                
                # We need depth_estimator!
                if depth_estimator:
                    inputs = depth_estimator.processor(images=pil_image, return_tensors="pt").to(depth_estimator.device)
                    with torch.no_grad():
                        outputs = depth_estimator.depth_model(**inputs)
                        predicted_depth = outputs.predicted_depth

                    # Resize depth map to original resolution
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=pil_image.size[::-1], 
                        mode="bicubic",
                        align_corners=False,
                    )
                    depth_map = prediction.squeeze().cpu().numpy()
                    # [STABILITY] Normalize Depth Map to 0.0 - 1.0 range
                    d_min, d_max = depth_map.min(), depth_map.max()
                    if d_max - d_min > 1e-6:
                        depth_map = (depth_map - d_min) / (d_max - d_min)
                else:
                    # Fallback if no depth_estimator provided (should not happen in batch)
                    depth_map = np.zeros((h_draw, w_draw))

                h_map, w_map = depth_map.shape

                # Helper to extract 3D point (2D Landmark + Depth Z) WITH CONFIDENCE
                def get_3d_point(idx):
                    lm = landmarks_list[idx]
                    # Normalized coordinates (0.0 ~ 1.0)
                    nx = lm['x']
                    ny = lm['y']
                    
                    # Pixel coordinates
                    px = int(nx * w_map)
                    py = int(ny * h_map)
                    
                    # Clamp pixel coordinates
                    px = max(0, min(w_map - 1, px))
                    py = max(0, min(h_map - 1, py))
                    
                    # Get Initial Depth from map
                    z = float(depth_map[py, px])
                    vis = lm.get('visibility', 0.0)
                    
                    return {
                        'x': px, 'y': py,   # Pixel coords for drawing & length calc
                        'nx': nx, 'ny': ny, # Normalized coords for reference
                        'z': z,             # Depth value (0-1 or raw model output)
                        'vis': vis
                    }

                # Auto Side Selection
                if self.side is None:
                     left_vis = (landmarks_list[11]['visibility'] + landmarks_list[23]['visibility'] + landmarks_list[25]['visibility']) / 3
                     right_vis = (landmarks_list[12]['visibility'] + landmarks_list[24]['visibility'] + landmarks_list[26]['visibility']) / 3
                     self.side = "left" if left_vis >= right_vis else "right"
                     print(f"DEBUG: Auto-selected side: {self.side}")

                if self.side == "left":
                    # Indices: Hip 23, Knee 25, Ankle 27, Shoulder 11
                    idx_hip, idx_knee, idx_ankle, idx_shoulder = 23, 25, 27, 11
                else:
                    # Indices: Hip 24, Knee 26, Ankle 28, Shoulder 12
                    idx_hip, idx_knee, idx_ankle, idx_shoulder = 24, 26, 28, 12

                hip = get_3d_point(idx_hip)
                knee = get_3d_point(idx_knee)
                ankle = get_3d_point(idx_ankle)
                shoulder = get_3d_point(idx_shoulder)

                # Confidence Check 
                avg_vis = (hip['vis'] + knee['vis'] + ankle['vis']) / 3
                if avg_vis < 0.1: # Low confidence (Lowered to 0.1)
                    out.write(image_draw)
                    continue

                # --- 3. Angle Calculation (Using Pure 2D) ---
                # Key: We calculate angles using 2D coordinates (h_map/w_map scaling handles pixel aspect)
                # But get_3d_point returns scaled pixels.
                raw_thigh_angle = self.calculate_vertical_angle(hip, knee)
                raw_trunk_angle = self.calculate_vertical_angle(shoulder, hip)
                raw_shank_angle = self.calculate_vertical_angle(knee, ankle) # Shank usually doesn't need bias or different bias


                
                # --- RAZS (Robust Adaptive Z-Scaling) Algorithm ---
                # Goal: Find k such that Thigh_Length_3D ≈ Shank_Length_3D
                # We calculate instantaneous k from current frame, then smooth it.
                
                # 1. Measurements
                # 2D Lengths (Using Pixel Coordinates)
                shank_px_vec = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])
                thigh_px_vec = np.array([knee['x'] - hip['x'], knee['y'] - hip['y']])
                torso_px_vec = np.array([hip['x'] - shoulder['x'], hip['y'] - shoulder['y']])
                
                shank_len_2d = np.linalg.norm(shank_px_vec)
                thigh_len_2d = np.linalg.norm(thigh_px_vec)
                torso_len_2d = np.linalg.norm(torso_px_vec)
                
                # --- [NEW] Depth Anything V2 Inference ---
                # (Already done above, depth_map is ready)
                
                # 4. Extract Z-values from Normalized Depth Map
                # We already extracted Z inside get_3d_point, checking compatibility...
                # get_3d_point used pixel coordinates correctly to fetch Z.
                # So we can skip the redundant get_depth_val function or just use the values we have.
                
                z_hip = hip['z']
                z_knee = knee['z']
                z_ankle = ankle['z']
                z_shoulder = shoulder['z']

                # --- Replace RAZS Z with Depth Anything Z ---
                # Note: Depth Anything output is "Relative Depth" (Disparity), closer is brighter (higher value).
                # But sometimes it's inverse. We need to check the model spec.
                # Usually: High Value = Close, Low Value = Far.
                # We need metric depth? No, relative is fine for angles if scaled.
                # Let's try direct substitution for testing.
                
                # Update landmarks with new Z (Overrides MediaPipe Z)
                # Scale factor might be needed? Let's assume 1.0 for now and see.
                hip['z'] = z_hip
                knee['z'] = z_knee
                ankle['z'] = z_ankle
                shoulder['z'] = z_shoulder
                
                # Skip RAZS Logic (It was overwritten anyway or we just ignore it for angle calc)
                # But we still need Z-Scale Param?
                # For Depth Anything, the Z is already consistent across the body (hopefully).
                
                # 2. Validity Checks (Gating)
                dZ_shank_raw = abs(z_ankle - z_knee)
                dZ_thigh_raw = abs(z_knee - z_hip)
                
                # 2. Validity Checks (Gating)
                # G1. Confidence Gate
                min_conf = min(hip['vis'], knee['vis'], ankle['vis'])
                
                # G2. Pose Gating (Don't update if leg is too straight or too bent - unstable math)
                # Best update range: Mid-squat (knee angle 100 ~ 160 deg).
                # We use raw 2D angle for gating to be safe.
                raw_knee_angle_2d = self.calculate_vertical_angle(hip, knee) # Approximation for gating
                
                valid_update = False
                instant_k = self.z_scale_param # Default to current
                
                # Thigh must be visible enough (len > 10px) to judge ratio
                if min_conf > 0.5 and (thigh_len_2d > 10 and shank_len_2d > 10):
                     # Update Max Reference Lengths (Adaptive Learning)
                     # SAFETY: Only update if Very High Confidence (>0.8)
                     if min_conf > 0.8:
                         # Update Max Shank
                         if shank_len_2d > self.max_shank_len_2d and shank_len_2d < 1.5 * thigh_len_2d:
                             self.max_shank_len_2d = shank_len_2d
                         
                         # [NEW] Update Max Thigh (Critical: Capture Standing Pose)
                         # When standing (s1), thigh length is maximized in projection.
                         # We use this "Best Observed Thigh" as our Ground Truth target.
                         if thigh_len_2d > self.max_thigh_len_2d:
                             self.max_thigh_len_2d = thigh_len_2d
                             
                         # [NEW] Update Max Torso Length (Most Robust Reference)
                         if torso_len_2d > self.max_torso_len_2d:
                             self.max_torso_len_2d = torso_len_2d

                     # G3. Solver
                     # Target: L_thigh_3D = The Best Available Estimate (Torso-Referenced)
                     # Method A: Direct Thigh observation (Best if standing)
                     target_A = self.max_thigh_len_2d
                     
                     # Method B: Torso Ratio (Thigh ≈ 0.75 * Torso) - Robust across poses
                     target_B = 0.75 * self.max_torso_len_2d
                     
                     # Method C: [CRITICAL] Real-time Shank Ratio (Thigh ≈ 1.2 * current Shank)
                     # This is the "God Move" for RAZS. Even if we never saw a standing pose,
                     # we know that if the shank is visible, the thigh should be ~1.2x its length.
                     target_C = 1.2 * shank_len_2d
                     
                     # [CRITICAL LOGIC] Pick the LARGEST credible estimate.
                     best_estimate = max(target_A, target_B, target_C)
                     
                     # Safety fallback if nothing seen yet
                     if best_estimate == 0: best_estimate = shank_len_2d * 1.2
                     
                     target_sq = best_estimate**2
                     thigh_sq = thigh_len_2d**2
                     dz_sq = dZ_thigh_raw**2
                     
                     if dz_sq > 1e-6 and target_sq > thigh_sq:
                         k_sq = (target_sq - thigh_sq) / dz_sq
                         if k_sq > 0:
                             instant_k = np.sqrt(k_sq)
                             # G4. Clamp (Relaxed for Depth Anything 0-1 range vs Pixels)
                             # Depth map is 0~1, Pixels are 0~2000. Ratio can be ~1000.
                             if 0.1 < instant_k < 10000.0:
                                 valid_update = True
                     # 3. Update State (EMA & Step Clamp)
                     if valid_update:
                         # G5. Step Clamp (Relaxed to 10% for faster initial convergence)
                         max_step = self.z_scale_param * 0.50

                         diff = instant_k - self.z_scale_param
                         diff = max(-max_step, min(max_step, diff))
                         target_k = self.z_scale_param + diff
                         
                         # G6. EMA Smoothing (Slightly faster adaptation)
                         alpha = 0.3
                         self.z_scale_param = (1 - alpha) * self.z_scale_param + alpha * target_k
                         
                         self.razs_stable = True
                         # print(f"DEBUG: RAZS Update. k={self.z_scale_param:.3f}")

                # --- Apply RAZS to Angle Calculation ---
                # Now we use self.z_scale_param to scale Z and compute angles.
                
                def get_3d_vec_angle(p_top, p_bottom):
                    # Vector from Top to Bottom
                    vx = p_bottom['x'] - p_top['x']
                    vy = p_bottom['y'] - p_top['y'] # Y is down in image, but Gravity is (0,1,0)
                    vz = (p_bottom['z'] - p_top['z']) * self.z_scale_param
                    
                    v_seg = np.array([vx, vy, vz])
                    norm_seg = np.linalg.norm(v_seg)
                    
                    if norm_seg == 0: return 0.0
                    
                    # Vertical Vector (Gravity) -> Roughly (0, 1, 0) in Image Space if camera is upright
                    v_ref = np.array([0., 1., 0.])
                    
                    cosine = np.dot(v_seg, v_ref) / norm_seg
                    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

                thigh_angle = get_3d_vec_angle(hip, knee)
                trunk_angle = get_3d_vec_angle(shoulder, hip)
                shank_angle = get_3d_vec_angle(knee, ankle)
                
                # 4. FSM Logic with Hysteresis (노이즈 방지)
                current_state_label = self.state # Default to current
                
                # Simple Low-pass Filter for Thigh Angle to reduce flickering
                if not hasattr(self, 'filtered_thigh'): self.filtered_thigh = thigh_angle
                self.filtered_thigh = 0.7 * self.filtered_thigh + 0.3 * thigh_angle
                f_thigh = self.filtered_thigh

                if self.state == "s3":
                    # s3 상태일 때는 임계값보다 5도 더 올라와야 s2로 전환 (히스테리시스)
                    if f_thigh < (self.TH_S3_MIN - 5.0):
                        if f_thigh < self.TH_S1_LIMIT:
                            current_state_label = "s1"
                        else:
                            current_state_label = "s2"
                else:
                    # s3가 아닐 때의 일반적인 전환
                    if f_thigh < self.TH_S1_LIMIT:
                        current_state_label = "s1"
                    elif self.TH_S1_LIMIT <= f_thigh < self.TH_S3_MIN:
                        current_state_label = "s2"
                    elif f_thigh >= self.TH_S3_MIN:
                        current_state_label = "s3"
                
                # --- Veto Logic (Checked during s2 AND s3 for consistency) ---
                if current_state_label in ['s2', 's3']:
                    if current_state_label == 's3':
                        self.has_hit_s3 = True
                    
                    # Rule 1: Trunk Angle (Good Morning)
                    if trunk_angle > self.TRUNK_LIMIT:
                        warning = f"BACK TOO BENT ({trunk_angle:.1f})"
                        self.feedback = warning
                        self.had_form_issue = True
                        self.issue_types.add("BACK_TOO_BENT")
                        self.issue_states.add(current_state_label.upper())
                        
                    # Rule 2: Alignment (Balance)
                    elif abs(trunk_angle - shank_angle) > self.ALIGN_LIMIT:
                        warning = f"UNSTABLE BAL (Diff {abs(trunk_angle - shank_angle):.1f})"
                        self.feedback = warning
                        self.had_form_issue = True
                        self.issue_types.add("UNSTABLE_BAL")
                        self.issue_states.add(current_state_label.upper())
                        
                    # Rule 3: Knee Over Toe (3D Depth Check using Depth Anything Z)
                    else:
                        z_diff = knee['z'] - ankle['z']
                        if z_diff > self.KNEE_LIMIT: 
                             warning = f"KNEE FWD (Z-Diff {z_diff:.2f})"
                             self.feedback = warning
                             self.had_form_issue = True
                             self.issue_types.add("KNEE_FWD")
                             self.issue_states.add(current_state_label.upper())

                # State Transition & Counting (Strict Sequence Logic)
                if self.state != current_state_label:
                    # State Changed
                    self.state_sequence.append(current_state_label)
                    
                    # s1으로 돌아왔을 때, 시퀀스 검사
                    if current_state_label == 's1':
                        # Check strictly for [s1, s2, s3, s2, s1] pattern
                        # We allow repeats like s2, s2 due to noise, but the TRANSITIONS must be correct.
                        # Actually, self.state_sequence only records changes.
                        # So it should look exactly like [... 's1', 's2', 's3', 's2', 's1']
                        
                        # Check for [s1, s2, s3, s2, s1] pattern
                        rep_counted = False
                        if len(self.state_sequence) >= 5:
                             last_5 = self.state_sequence[-5:]
                             if last_5 == ['s1', 's2', 's3', 's2', 's1']:
                                 self.counter += 1
                                 rep_counted = True
                        
                        # Rescue Logic: If first rep starts from s2 (Sequence: s2, s3, s2, s1)
                        if not rep_counted and self.counter == 0 and len(self.state_sequence) >= 4:
                             last_4 = self.state_sequence[-4:]
                             if last_4 == ['s2', 's3', 's2', 's1']:
                                 self.counter += 1
                                 rep_counted = True
                                 print(f"DEBUG: Rescued first rep from s2-start sequence!")

                        if rep_counted:
                             self.one_rm = self.weight * (1 + 0.0333 * self.counter)
                             if self.had_form_issue:
                                 self.feedback = "COUNT (WARNING)!"
                             else:
                                 self.feedback = "COUNT!"

                self.state = current_state_label 
                
                # 5. Visualization (Skeleton)
                self.pose_estimator._draw_landmarks(image_draw, landmarks_list)

                # Dashboard Box - [ENLARGED]
                cv2.rectangle(image_draw, (0,0), (450, 180), (245, 117, 16), -1) 
                
                # Rep Count
                cv2.putText(image_draw, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image_draw, str(self.counter), (10,65), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 3, cv2.LINE_AA)
                
                # State - [BIGGER]
                cv2.putText(image_draw, f'STATE: {self.state}', (190, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3, cv2.LINE_AA)

                # 3-Angle Dashboard -> SIMPLIFIED to Thigh Only (Comparison) - [MUCH BIGGER]
                cv2.putText(image_draw, f'Thigh(3D): {int(thigh_angle)}', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2) # Cyan
                
                # Calculate Naive 2D Angle
                v2d_seg = np.array([knee['x'] - hip['x'], knee['y'] - hip['y']])
                norm2d = np.linalg.norm(v2d_seg)
                angle_2d = 0.0
                if norm2d > 0:
                     cosine2d = np.dot(v2d_seg, np.array([0., 1.])) / norm2d
                     angle_2d = np.degrees(np.arccos(np.clip(cosine2d, -1.0, 1.0)))
                
                cv2.putText(image_draw, f'Thigh(2D): {int(angle_2d)}', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2) # Red

                # Draw angle on image (at Knee) - [Also Bigger]
                knee_px = (int(knee['x']), int(knee['y']))
                cv2.putText(image_draw, str(int(thigh_angle)), 
                            knee_px, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Sequence Debug (Smaller, Bottom)
                seq_str = "->".join(self.state_sequence[-5:])
                cv2.putText(image_draw, f"Seq: {seq_str}", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

                # [VISUALIZATION] Draw Horizontal Limit Line (Parallel Squat Target)
                knee_y = int(knee['y'])
                cv2.line(image_draw, (0, knee_y), (w, knee_y), (0, 255, 255), 3) # Yellow (Thicker)
                
                # [VISUALIZATION] Draw Vertical Reference Line (Gravity Vector)
                hip_x, hip_y = int(hip['x']), int(hip['y'])
                cv2.line(image_draw, (hip_x, hip_y), (hip_x, h), (255, 255, 255), 1, cv2.LINE_AA) # White (Gravity)


                # [VISUALIZATION] Compare 2D vs 3D Angle
                # Calculate Naive 2D Angle
                v2d_seg = np.array([knee['x'] - hip['x'], knee['y'] - hip['y']])
                norm2d = np.linalg.norm(v2d_seg)
                angle_2d = 0.0
                if norm2d > 0:
                     cosine2d = np.dot(v2d_seg, np.array([0., 1.])) / norm2d
                     angle_2d = np.degrees(np.arccos(np.clip(cosine2d, -1.0, 1.0)))

                # Draw Comparison at Hip/Thigh Area
                center_x = int((hip['x'] + knee['x']) / 2) + 40
                center_y = int((hip['y'] + knee['y']) / 2)
                
                # Show Real (3D) in Blue (Cyan)
                cv2.putText(image_draw, f"RAZS 3D: {int(thigh_angle)} deg", (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) # Cyan
                # Show Naive (2D) in Red
                cv2.putText(image_draw, f"Naive 2D: {int(angle_2d)} deg", (center_x, center_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # Red

                # [2-PASS SUPPORT] Store History for Re-Evaluation
                self.history_data.append({
                    'hip': hip, 'knee': knee, 'ankle': ankle, 'shoulder': shoulder,
                    'thigh_len_2d': thigh_len_2d,
                    'shank_len_2d': shank_len_2d,
                    'torso_len_2d': torso_len_2d,
                    'dZ_thigh': dZ_thigh_raw,
                    'dZ_shank': dZ_shank_raw
                })

            else:
                if frame_idx % 30 == 0:
                    print(f"Frame {frame_idx}: No landmarks detected.")

            out.write(image_draw) # 영상 저장
            
            if show_window:
                cv2.imshow('Squat Analysis', image_draw)
                # ESC or q to exit
                if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                    break
        
        print(f"Final Rep Count: {self.counter}")
        print(f"Final Sequence: {self.state_sequence}")
        print(f"Final 1RM Estimate: {self.one_rm:.2f} kg")
        print(f"Video saved to: {output_path}")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return self.counter, self.had_form_issue, list(self.issue_types), list(self.issue_states)

    def finalize_analysis(self):
        """
        [2-Pass Logic] Re-evaluate the entire video using GLOBAL FIXED parameters.
        Step 1: Calculate the 'Best K' (Median) from all valid frames (Standing Poses).
        Step 2: Re-run analysis with this FIXED K. L(Thigh) is now dynamic driven by Depth.
        """
        if not self.history_data:
            return self.counter, list(self.issue_types), {}
            
        # 1. Find Global Maxima across ALL frames
        g_max_thigh = 0.0
        g_max_shank = 0.0
        g_max_torso = 0.0
        
        # [Step 1] Collect Candidate K values
        k_candidates = []
        
        for frame in self.history_data:
            g_max_thigh = max(g_max_thigh, frame['thigh_len_2d'])
            g_max_shank = max(g_max_shank, frame['shank_len_2d'])
            g_max_torso = max(g_max_torso, frame['torso_len_2d'])
            
            # RAZS Solver (Local)
            thigh_len_2d = frame['thigh_len_2d']
            dZ_thigh_raw = frame['dZ_thigh']
            
            # Target: Multi-Factor Estimation (Same as 1-Pass Logic)
            # We want the BEST estimate of the TRUE thigh length.
            # 1. Observed Max Thigh (Direct observation)
            target_A = g_max_thigh
            # 2. Shank Ratio (Thigh ≈ 1.2 * Shank)
            target_B = 1.2 * frame['shank_len_2d']
            # 3. Torso Ratio (Thigh ≈ 0.75 * Torso)
            target_C = 0.75 * g_max_torso
            
            # Pick the LARGEST credible estimate
            target = max(target_A, target_B, target_C)
            
            if target < 10: continue
            
            # Solve for k: target^2 = thigh^2 + (k*dz)^2
            target_sq = target**2
            thigh_sq = thigh_len_2d**2
            dz_sq = dZ_thigh_raw**2
            
            if dz_sq > 1e-6 and target_sq > thigh_sq:
                 k_sq = (target_sq - thigh_sq) / dz_sq
                 if k_sq > 0:
                     k_candidates.append(np.sqrt(k_sq))

            # Determine Global Base K (Median is robust to outliers)
        if k_candidates:
            base_k = float(np.median(k_candidates))
            # Safety checks (Virtually unlimited: 10,000)
            base_k = max(1.0, min(10000.0, base_k))
        else:
            base_k = 1.0 # Default fallback
            
        print(f"DEBUG: 2-Pass Re-Eval. Global K={base_k:.3f}, Max Thigh={g_max_thigh:.1f}")
        
        # Capture History before Rest Wipes it
        local_history = self.history_data
        
        # 2. Reset State Machine for Re-Run
        self.reset()
        # Restore Max Lengths AND Set Base K
        self.max_thigh_len_2d = g_max_thigh
        self.max_shank_len_2d = g_max_shank
        self.max_torso_len_2d = g_max_torso
        self.z_scale_param = base_k # [CRITICAL] Fix K for the entire run
        
        # 3. Virtual Re-Run
        for frame in local_history:
            # Extract stored data
            hip, knee, ankle, shoulder = frame['hip'], frame['knee'], frame['ankle'], frame['shoulder']
            thigh_len_2d = frame['thigh_len_2d']
            shank_len_2d = frame['shank_len_2d']
            torso_len_2d = frame['torso_len_2d']
            dZ_thigh_raw = frame['dZ_thigh']
            
            # --- [NEW] TRUE 2-PASS LOGIC ---
            # We do NOT re-calculate k. We use fixed self.z_scale_param (base_k).
            # This allows Depth (dZ) to directly drive the 3D Length logic.
            # L_thigh_3D = sqrt(2D^2 + (base_k * dZ)^2)
            
            # Calculate Angles (Using Fixed K)
            def get_3d_vec_angle(p_top, p_bottom):
                    vx = p_bottom['x'] - p_top['x']
                    vy = p_bottom['y'] - p_top['y']
                    # Use FIXED global K
                    vz = (p_bottom['z'] - p_top['z']) * self.z_scale_param 
                    v_seg = np.array([vx, vy, vz])
                    norm_seg = np.linalg.norm(v_seg)
                    if norm_seg == 0: return 0.0
                    v_ref = np.array([0., 1., 0.])
                    cosine = np.dot(v_seg, v_ref) / norm_seg
                    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

            # [VISUALIZATION] Compare 2D vs 3D Angle
            # 2D Angle (Naive): Just using x, y
            def get_2d_vec_angle(p_top, p_bottom):
                    vx = p_bottom['x'] - p_top['x']
                    vy = p_bottom['y'] - p_top['y']
                    v_seg = np.array([vx, vy]) # No Z
                    norm_seg = np.linalg.norm(v_seg)
                    if norm_seg == 0: return 0.0
                    v_ref = np.array([0., 1.]) # Vertical Down in 2D
                    cosine = np.dot(v_seg, v_ref) / norm_seg
                    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

            thigh_angle = get_3d_vec_angle(hip, knee)
            thigh_angle_2d = get_2d_vec_angle(hip, knee)
            
            # Draw Comparison Info
            # Top-Left corner info
            
            trunk_angle = get_3d_vec_angle(shoulder, hip)
            shank_angle = get_3d_vec_angle(knee, ankle)
            
            # FSM Update (STRICT Sequence Check for 2nd Pass)
            # IPF Standard Check
            # Tolerance -20px (Strict but allows knee cap)
            vertical_depth_pass = (hip['y'] >= knee['y'] - 20) 
            angle_pass = (thigh_angle >= self.TH_S3_MIN)
            is_deep_enough = angle_pass or vertical_depth_pass

            # Update Label
            if self.state == "s1":
                if thigh_angle > self.TH_S1_LIMIT: current_label = "s2"
                else: current_label = "s1"
            elif self.state == "s2":
                if is_deep_enough: current_label = "s3"
                elif thigh_angle < self.TH_S1_LIMIT: current_label = "s1"
                else: current_label = "s2"
            elif self.state == "s3":
                # Exit Bottom logic (requires significant rise)
                if not is_deep_enough and thigh_angle < self.TH_S3_MIN - 10: current_label = "s2"
                else: current_label = "s3"

            # Form Checks (Non-Blocking for Count)
            if current_label in ['s2', 's3']:
                 if trunk_angle > self.TRUNK_LIMIT: 
                     self.had_form_issue = True
                     self.issue_types.add("BACK_TOO_BENT")
                 elif abs(trunk_angle - shank_angle) > self.ALIGN_LIMIT:
                     self.had_form_issue = True
                     self.issue_types.add("UNSTABLE_BAL")

            # Sequence State Change Tracking
            if current_label != self.state:
                self.state = current_label
                self.state_sequence.append(self.state)
                
                # Count Logic (Strict Pattern)
                if self.state == 's1':
                    rep_counted = False
                    if len(self.state_sequence) >= 5:
                        if self.state_sequence[-5:] == ['s1', 's2', 's3', 's2', 's1']:
                            # [CRITICAL UPDATE] Count regardless of Form Issue
                            # Accuracy goal: Count repetitions correctly first.
                            self.counter += 1
                            rep_counted = True
                    # Rescue (Allow start from s2)
                    if not rep_counted and self.counter == 0 and len(self.state_sequence) >= 4:
                        if self.state_sequence[-4:] == ['s2', 's3', 's2', 's1']:
                            self.counter += 1
                    
                    self.had_form_issue = False # Reset for next rep

        calibration_data = {
            'max_shank_len_2d': self.max_shank_len_2d,
            'max_thigh_len_2d': self.max_thigh_len_2d,
            'max_torso_len_2d': self.max_torso_len_2d,
            'base_k': base_k # Report the fixed K used
        }
        
        print(f"DEBUG: 2-Pass Result. K={base_k:.2f}, Count={self.counter}, Sequence={self.state_sequence}")
        return self.counter, list(self.issue_types), calibration_data


if __name__ == "__main__":
    analyzer = SquatAnalyzer()
    # 테스트 영상 경로
    video_path = "dataset/true/true_14.mp4" 
    analyzer.process_video(video_path)
