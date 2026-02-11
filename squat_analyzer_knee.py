import cv2
import mediapipe as mp
import numpy as np
import time
from pose_estimation import PoseEstimator
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from PIL import Image

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

        # 상태 관리 (FSM)
        self.state = "s1" # s1: Standing, s2: Transition, s3: Bottom
        self.state_sequence = []
        self.counter = 0
        self.feedback = ""
        self.had_form_issue = False 
        self.issue_types = set() 
        self.issue_states = set() 
        
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
        self.state = "s1"
        self.state_sequence = []
        self.counter = 0
        self.feedback = ""
        self.had_form_issue = False
        self.issue_types = set()
        self.issue_states = set()
        self.one_rm = 0.0
        self.side = None 
        self.stand_knee_angle = 170.0
        self.is_calibrated = False
        self.z_scale_param = 1.0 # Default
        self.history_data = [] # [NEW]

    def finalize_analysis(self):
        """
        Compute Global K from history (2-Pass Logic).
        """
        if not self.history_data:
            return self.counter, [], {}
            
        k_candidates = []
        # Find global max lengths first
        g_max_thigh = 0.0
        for frame in self.history_data:
            g_max_thigh = max(g_max_thigh, frame['thigh_len_2d'])
            
        for frame in self.history_data:
            shank_len = frame['shank_len_2d']
            thigh_len = frame['thigh_len_2d']
            dZ = frame['dZ_thigh']
            
            # Target Estimate (3-Way Selection)
            # 1. Observed Max Thigh (Direct)
            target_A = g_max_thigh
            # 2. Shank Ratio
            target_B = 1.2 * shank_len
            # 3. Trunk Ratio (New Candidate)
            target_C = 0.75 * frame['trunk_len_2d']
            
            target = max(target_A, target_B, target_C)
            
            if dZ > 1e-6 and target > thigh_len:
                k_sq = (target**2 - thigh_len**2) / (dZ**2)
                if k_sq > 0:
                    k_candidates.append(np.sqrt(k_sq))
                    
        if k_candidates:
            base_k = float(np.median(k_candidates))
            base_k = max(1.0, min(10000.0, base_k)) # Clamp
        else:
            base_k = 1.0
            
        print(f"DEBUG: 2-Pass Knee Re-Eval. Global K={base_k:.3f}")
        
        # Return calibration data for Pass 2 override
        return self.counter, [], {'base_k': base_k}
        # For Trunk Angle (2D)
        delta_x = p2['x'] - p1['x']
        delta_y = p2['y'] - p1['y']
        vec_len = np.sqrt(delta_x**2 + delta_y**2)
        if vec_len == 0: return 0
        dot_product = delta_y
        cosine_angle = dot_product / vec_len
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

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
        
        import os
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
                if depth_estimator:
                    inputs = depth_estimator.processor(images=pil_image, return_tensors="pt").to(depth_estimator.device)
                    with torch.no_grad():
                        outputs = depth_estimator.depth_model(**inputs)
                        predicted_depth = outputs.predicted_depth
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1), size=pil_image.size[::-1], mode="bicubic", align_corners=False,
                    )
                    depth_map = prediction.squeeze().cpu().numpy()
                    d_min, d_max = depth_map.min(), depth_map.max()
                    if d_max - d_min > 1e-6:
                        depth_map = (depth_map - d_min) / (d_max - d_min)
                        # Invert map if needed (Usually Depth Anything: Close=High)
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

                # Auto Side Selection
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

                # --- KNEE ANGLE LOGIC (SIMPLE 2-PHASE) ---
                knee_angle_3d = self.calculate_knee_angle(hip_s, knee_s, ankle_s)
                
                # Simple Thresholds
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
                
                cv2.putText(image_draw, f'Knee 3D: {int(knee_angle_3d)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.putText(image_draw, f'DOWN < {int(TH_DOWN)} | UP > {int(TH_UP)}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                knee_px = (int(knee['x']), int(knee['y']))
                cv2.putText(image_draw, str(int(knee_angle_3d)), knee_px, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            out.write(image_draw)
            
        cap.release()
        out.release()
        print(f"  > Count: {self.counter}")
        return self.counter, False, [], []

    def finalize_analysis(self):
        # Stub for compatibility with process_dataset.py
        return self.counter, [], {}
