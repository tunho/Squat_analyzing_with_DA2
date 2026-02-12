import cv2
import os
import mediapipe as mp
import numpy as np
import time
from pose_estimation import PoseEstimator
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from PIL import Image
from joint_correction import JointCenterCorrector # [NEW] Import Corrector

class SquatAnalyzer:
    def __init__(self):
        # MediaPipe Pose 초기화
        self.pose_estimator = PoseEstimator(static_image_mode=False, model_complexity=1)
        
        # Depth Anything V2 초기화 - ENABLED (For 3D Coordinates)
        print("Loading Depth Anything V2-Small (Knee Mode)...")
        self.depth_model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(self.depth_model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_model_id).to(self.device)
        print("Depth Anything Model Loaded.")
        
        # [NEW] Joint Center Corrector (Geometric Proxy)
        self.corrector = JointCenterCorrector(mode='geometric')

        # 상태 관리 (FSM)
        self.state = "UP" # UP, DOWN
        self.counter = 0
        self.feedback = ""
        
        # [Knee Mode Config]
        self.stand_knee_angle = 170.0 # Default assumption
        self.is_calibrated = False
        
        # Thresholds (Relative)
        self.required_flexion = 60.0 # Must bend at least 60 degrees from standing
        
        # Posture Issue Thresholds
        self.TRUNK_LIMIT = 35.0 
        self.ALIGN_LIMIT = 20.0 
        
        self.side = None
        self.weight = 20.0
        self.one_rm = 0.0

    def reset(self):
        """Reset state for batch processing"""
        self.state = "UP"
        self.counter = 0
        self.feedback = ""
        self.one_rm = 0.0
        self.side = None 
        self.stand_knee_angle = 170.0
        self.is_calibrated = False
        self.z_scale_param = 1.0 # Default
        self.history_data = [] # [NEW]

    def finalize_analysis(self):
        """
        Perform 3-Pass Analysis for Optimal Accuracy.
        Pass 1 (Detection): Already done in process_video (collects history_data).
        Pass 2 (Optimization): Solve for Global K and Stable Radii using whole-video stats.
        Pass 3 (Re-calculation): Re-compute all knee angles using the optimized K and Radii.
        """
        if not self.history_data:
            return self.counter, [], {}
            
        # --- Pass 2: Optimization (Find the Best K and Radii) ---
        
        # 1. 관측된 최대 길이 추출
        obs_max_thigh = max(f['thigh_len_2d'] for f in self.history_data)
        obs_max_shank = max(f['shank_len_2d'] for f in self.history_data)
        
        # 2. [Rigid Body Strategy] 신체 비율 고정 (Anatomical Constraint)
        # 허벅지와 정강이 중 더 길게(신뢰도 높게) 관측된 값을 기준으로 전체 비율을 동기화합니다.
        # 기준: 허벅지 vs 정강이*1.2 중 더 큰 값을 '진짜 허벅지 길이'로 채택.
        
        final_thigh_len = max(obs_max_thigh, obs_max_shank * 1.2)
        final_shank_len = final_thigh_len / 1.2 # 정강이는 무조건 허벅지의 약 0.83배로 고정
        
        # 3. 고정된 반지름 및 두께 차이 산출 (Physical Consistency)
        # 이제 radii와 dr은 완벽한 1.2:1 비율 하에서 계산됩니다.
        stable_radii = self.corrector.get_radii(final_thigh_len, final_shank_len)
        dr_stable = stable_radii['knee'] - stable_radii['hip']
        
        k_candidates = []
        for frame in self.history_data:
            thigh_len = frame['thigh_len_2d']
            
            # Target is now a FIXED constant based on the rigid body model.
            # We trust 'final_thigh_len' is the true 3D length.
            target = final_thigh_len
            
            if target > thigh_len:
                z_needed = np.sqrt(target**2 - thigh_len**2)
                dZ_raw = frame['knee']['z'] - frame['hip']['z']
                
                if abs(dZ_raw) > 1e-6:
                     # Solve for K with stable parameters
                     k1 = (z_needed - dr_stable) / dZ_raw
                     if k1 > 0: k_candidates.append(k1)
                     
                     k2 = (-z_needed - dr_stable) / dZ_raw
                     if k2 > 0: k_candidates.append(k2)

        # Determine Global K
        if k_candidates:
            base_k = float(np.median(k_candidates))
            print(f"DEBUG: Found {len(k_candidates)} K candidates. Median: {base_k:.3f}")
            base_k = max(1.0, min(10000.0, base_k)) # Clamp
        else:
            print("DEBUG: No valid K candidates found. Defaulting to 1.0.")
            base_k = 1.0
            
        print(f"DEBUG: 3-Pass Analysis Complete. Global K={base_k:.3f}, Dr={dr_stable:.1f}")
        print(f"DEBUG: Body Model -> Thigh: {final_thigh_len:.1f}, Shank: {final_shank_len:.1f} (Ratio 1.2:1)")

        # --- Pass 3: Re-calculation (Apply Optimal Parameters) ---
        final_angles = []
        
        for frame in self.history_data:
            # 1. Recover scale using Global K
            hip_z_metric = frame['hip']['z'] * base_k
            knee_z_metric = frame['knee']['z'] * base_k
            ankle_z_metric = frame['ankle']['z'] * base_k
            
            # 2. Apply Center Correction (Radius Push)
            hip_z_center = hip_z_metric + stable_radii['hip']
            knee_z_center = knee_z_metric + stable_radii['knee']
            ankle_z_center = ankle_z_metric + stable_radii['ankle']
            
            # 3. Form 3D Points
            hip_3d = {'x': frame['hip']['x'], 'y': frame['hip']['y'], 'z': hip_z_center}
            knee_3d = {'x': frame['knee']['x'], 'y': frame['knee']['y'], 'z': knee_z_center}
            ankle_3d = {'x': frame['ankle']['x'], 'y': frame['ankle']['y'], 'z': ankle_z_center}
            
            # 4. Calculate Final 3D Angle
            angle_3d = self.calculate_knee_angle(hip_3d, knee_3d, ankle_3d)
            final_angles.append(angle_3d)
            
        return self.counter, final_angles, {'base_k': base_k, 'dr': dr_stable}

        if k_candidates:
            # Robust Median
            base_k = float(np.median(k_candidates))
            print(f"DEBUG: Found {len(k_candidates)} K candidates. Median: {base_k:.3f}")
            base_k = max(1.0, min(10000.0, base_k)) # Clamp
        else:
            print("DEBUG: No valid K candidates found. Defaulting to 1.0.")
            base_k = 1.0
            
        print(f"DEBUG: 2-Pass Knee Re-Eval. Global K={base_k:.3f}")
        
        # Return calibration data for Pass 2 override
        return self.counter, [], {'base_k': base_k}

    @staticmethod
    def calculate_knee_angle(hip, knee, ankle):
        """
        Calculate 3D Knee Angle (Hip-Knee-Ankle).
        Full Extension = 180 degrees.
        """
        # Vector Knee->Hip
        v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y'], hip['z'] - knee['z']])
        # Vector Knee->Ankle
        v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y'], ankle['z'] - knee['z']])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 180.0
            
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def process_video(self, video_path, output_path="output.mp4", depth_estimator=None, show_window=False, override_params=None):
        self.reset()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0, False, [], []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 100 or fps <= 0: fps = 30.0
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        print(f"Processing (Knee Mode) {video_path}...")
        
        frame_idx = 0
        w_draw, h_draw = w, h
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            landmarks_list, results = self.pose_estimator.extract_keypoints_only(frame.copy())
            image_draw = frame.copy()
            
            if landmarks_list:
                # 1. Depth Estimation
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                
                # Depth Anything Logic
                model_source = depth_estimator if depth_estimator else self
                
                if model_source and hasattr(model_source, 'depth_model'):
                    inputs = model_source.processor(images=pil_image, return_tensors="pt").to(model_source.device)
                    with torch.no_grad():
                        outputs = model_source.depth_model(**inputs)
                        predicted_depth = outputs.predicted_depth
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1), size=pil_image.size[::-1], mode="bicubic", align_corners=False,
                    )
                    depth_map = prediction.squeeze().cpu().numpy()
                    d_min, d_max = depth_map.min(), depth_map.max()
                    if d_max - d_min > 1e-6:
                        # Normalize 0..1
                        # INVERTED: 1.0 - norm. So 0=Close (High Orig), 1=Far (Low Orig)
                        # Wait, Depth Anything: High=Close.
                        # We want Z to represent DISTANCE (Low=Close, High=Far) for "z += r" to mean "deeper".
                        # So High Orig (Close) -> Low New (Close).
                        # Low Orig (Far) -> High New (Far).
                        # (val - min)/(max - min) maps min->0, max->1.
                        # We want max->0 (Close), min->1 (Far).
                        # So 1.0 - (val - min)/(range).
                        depth_map = 1.0 - ((depth_map - d_min) / (d_max - d_min))
                        
                        # Scale Z to be comparable to X,Y pixels (0..1000)
                        # We need standard Z (Camera space): Close=Small Z, Far=Large Z?
                        # Actually for angle calc, relative scale matters.
                        # Let's map 0..1 to reasonable pixel-like scale for vector math
                        depth_map = depth_map * 1000.0 # Scale Z to be comparable to X,Y pixels
                else:
                    depth_map = np.zeros((h_draw, w_draw))
                
                h_map, w_map = depth_map.shape

                def get_3d_point(idx):
                    lm = landmarks_list[idx]
                    px = int(lm['x'] * w_map)
                    py = int(lm['y'] * h_map)
                    px = max(0, min(w_map - 1, px))
                    py = max(0, min(h_map - 1, py))
                    z = float(depth_map[py, px])
                    return {'x': px, 'y': py, 'z': z, 'vis': lm.get('visibility', 0.0)}

                # Auto Side Selection 신뢰도 계산
                if self.side is None:
                     left_vis = (landmarks_list[11]['visibility'] + landmarks_list[23]['visibility'] + landmarks_list[25]['visibility']) / 3
                     right_vis = (landmarks_list[12]['visibility'] + landmarks_list[24]['visibility'] + landmarks_list[26]['visibility']) / 3
                     self.side = "left" if left_vis >= right_vis else "right"

                if self.side == "left":
                    idx_hip, idx_knee, idx_ankle, idx_shoulder = 23, 25, 27, 11
                else:
                    idx_hip, idx_knee, idx_ankle, idx_shoulder = 24, 26, 28, 12

                hip = get_3d_point(idx_hip)
                knee = get_3d_point(idx_knee)
                ankle = get_3d_point(idx_ankle)
                shoulder = get_3d_point(idx_shoulder)

                # Confidence Check
                if min(hip['vis'], knee['vis'], ankle['vis']) < 0.1:
                    out.write(image_draw)
                    continue

                # --- RAZS & HISTORY STORAGE ---
                # Calculate Trunk Length for Candidate 3
                trunk_len_2d = np.linalg.norm([hip['x'] - shoulder['x'], hip['y'] - shoulder['y']])
                
                # Store data for 2-Pass Global Optimization
                self.history_data.append({
                     'hip': hip, 'knee': knee, 'ankle': ankle,
                     'shank_len_2d': np.linalg.norm([ankle['x'] - knee['x'], ankle['y'] - knee['y']]),
                     'thigh_len_2d': np.linalg.norm([knee['x'] - hip['x'], knee['y'] - hip['y']]),
                     'trunk_len_2d': trunk_len_2d,
                     'dZ_thigh': abs(hip['z'] - knee['z'])
                })
                
                # Use fixed base_k if available (Pass 2), else default 1.0 (Pass 1)
                # Note: For Pass 1, we use current z_scale_param which adapts.
                # But here we stick to z_scale_param = 1.0 (or base_k) for consistency.
                
                # Apply Scale
                hip_s = hip.copy(); knee_s = knee.copy(); ankle_s = ankle.copy()
                hip_s['z'] *= self.z_scale_param
                knee_s['z'] *= self.z_scale_param
                ankle_s['z'] *= self.z_scale_param

                # --- [NEW] JOINT CENTER CORRECTION ---
                # Prepare metadata for correction (using 2D lengths as proxy for size)
                # We use the raw 2D lengths (before any Z-scaling) to estimate thickness
                correction_meta = {
                    'thigh_len': self.history_data[-1]['thigh_len_2d'],
                    'shank_len': self.history_data[-1]['shank_len_2d'],
                    # Ideally use global max if available (Pass 2)
                    'max_thigh_len': self.history_data[-1]['thigh_len_2d'], 
                    'max_shank_len': self.history_data[-1]['shank_len_2d']
                }
                
                # Bundle joints
                surface_joints = {'hip': hip_s, 'knee': knee_s, 'ankle': ankle_s}
                
                # Get Corrected Joints (Pushed Inwards)
                center_joints = self.corrector.correct(surface_joints, correction_meta)
                
                hip_c = center_joints['hip']
                knee_c = center_joints['knee']
                ankle_c = center_joints['ankle']

                # --- KNEE ANGLE LOGIC (SIMPLE 2-PHASE) ---
                # Use CORRECTED (Center) Joints for Angle Calculation
                knee_angle_3d = self.calculate_knee_angle(hip_c, knee_c, ankle_c)
                
                # Calculate Surface Angle for Comparison
                surface_angle_3d = self.calculate_knee_angle(hip_s, knee_s, ankle_s)
                
                # Simple Thresholds (Knee Joint Angle)
                # TH_DOWN: High flexion (small angle) for bottom
                # TH_UP: Extension (large angle) for standing
                TH_DOWN = 95.0 
                TH_UP = 165.0
                
                # FSM Update (UP/DOWN)
                if knee_angle_3d < TH_DOWN:
                    self.state = "DOWN"
                
                if self.state == "DOWN" and knee_angle_3d > TH_UP:
                    self.counter += 1
                    self.state = "UP"
                             
                # Visualization
                self.pose_estimator._draw_landmarks(image_draw, landmarks_list)
                
                # Dashboard
                cv2.rectangle(image_draw, (0,0), (400, 150), (245, 117, 16), -1) 
                cv2.putText(image_draw, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image_draw, str(self.counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image_draw, f'STATE: {self.state}', (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2, cv2.LINE_AA)
                
                cv2.putText(image_draw, f'Knee 3D (Center): {int(knee_angle_3d)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.putText(image_draw, f'Surface Angle: {int(surface_angle_3d)}', (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(image_draw, f'DOWN < {int(TH_DOWN)} | UP > {int(TH_UP)}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                knee_px = (int(knee['x']), int(knee['y']))
                cv2.putText(image_draw, str(int(knee_angle_3d)), knee_px, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            out.write(image_draw)
            
        cap.release()
        out.release()
        print(f"  > Count: {self.counter}")
        return self.counter, False, [], []


if __name__ == "__main__":
    analyzer = SquatAnalyzer()
    video_path = "dataset/true/true_14.mp4"
    if os.path.exists(video_path):
        analyzer.process_video(video_path, output_path="output_knee_corrected.mp4", show_window=False)
        # Run 2-Pass Check
        analyzer.finalize_analysis()
    else:
        print(f"Video not found: {video_path}")
