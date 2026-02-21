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
from config import MODEL_CONFIG, SQUAT_THRESHOLDS # [REF] Central Config

class SquatAnalyzer:
    def __init__(self):
        # MediaPipe Pose 초기화
        self.pose_estimator = PoseEstimator(static_image_mode=False, model_complexity=MODEL_CONFIG['POSE_MODEL_COMPLEXITY'])
        
        # Depth Anything V2 초기화 - ENABLED (For 3D Coordinates)
        print("Loading Depth Anything V2-Small (Knee Mode)...")
        self.depth_model_id = MODEL_CONFIG['DEPTH_MODEL_ID']
        self.device = MODEL_CONFIG['DEVICE']
        self.processor = AutoImageProcessor.from_pretrained(self.depth_model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_model_id).to(self.device)
        print("Depth Anything Model Loaded.")
        
        # [NEW] Joint Center Corrector (Geometric Proxy)
        self.corrector = JointCenterCorrector(mode='geometric')

        # 상태 관리 (FSM)
        self.state = "UP" # UP, DOWN
        self.counter = 0
        self.feedback = ""
        
        self.side = None

    def reset(self):
        """Reset state for batch processing"""
        self.state = "UP"
        self.counter = 0
        self.feedback = ""
        self.side = None 
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
        
        # 3. 선형 회귀를 통한 Global K 산출 (정강이-Shank- 기반)
        # 엉덩이의 동적 두께 변화로 인한 오차를 피하기 위해, 두께 왜곡이 거의 없는 무릎-발목(Shank) 데이터로 K를 구합니다.
        stable_radii = self.corrector.get_radii(final_thigh_len, final_shank_len) 

        X_data = [] # dZ_raw (정강이의 순수 표면 깊이 격차, 단위: 상대값)
        Y_data = [] # Z_needed_bone (피타고라스가 요구하는 순수 내부 뼈 깊이 격차, 단위: 픽셀)
        
        for frame in self.history_data:
            shank_len = frame['shank_len_2d']
            target_shank = final_shank_len
            
            # 발목-무릎에 Z축 변위가 발생한 프레임만 사용
            if target_shank > shank_len:
                # 1. 뼈 속 기준 절대 깊이 도출
                z_needed_bone_shank = np.sqrt(target_shank**2 - shank_len**2)
                dZ_raw_shank = frame['ankle']['z'] - frame['knee']['z']
                
                if abs(dZ_raw_shank) > 1e-6:
                    # 방향 맞추기 (dZ_raw 방향에 맞게 Z_needed 부호 결정)
                    signed_z_needed_bone = z_needed_bone_shank if dZ_raw_shank > 0 else -z_needed_bone_shank
                    
                    # 2. 픽셀 오차(dr)를 미리 빼지 않습니다!
                    # X는 순수 표면 뎁스, Y는 순수 뼈대 뎁스. (Y = K * X + C)
                    # 선형 회귀가 X를 픽셀화시키는 K와, 표면-뼈 사이의 평균 두께 오차 C를 스스로 찾아내게 합니다.
                    X_data.append(dZ_raw_shank)
                    Y_data.append(signed_z_needed_bone)

        # 선형 회귀 (Linear Regression) 적용
        if len(X_data) > 2:
            # 1차 방정식 피팅 (Z_bone = K * dZ_surface + C)
            coefficients = np.polyfit(X_data, Y_data, 1)
            base_k = float(coefficients[0])  
            dynamic_dr_shank = float(coefficients[1]) # 회귀가 증명한 이 영상의 정강이 두께 총 오차
            
            # K가 음수이면(카메라 방향 역전) 절대값으로 보정
            if base_k <= 0:
                 base_k = abs(base_k)
                 if base_k == 0: base_k = 1.0
                     
            print(f"DEBUG: Shank-Based Regression fit with {len(X_data)} pairs.")
            print(f"DEBUG: Pure Scale Regression K = {base_k:.3f}, Dynamic C (dr_shank) = {dynamic_dr_shank:.1f}")
        else:
            base_k = 1.0
            print("DEBUG: Not enough valid shank data for regression. Defaulting K to 1.0.")

        print(f"DEBUG: 3-Pass Analysis Complete. Global K={base_k:.3f}")
        print(f"DEBUG: Body Model -> Thigh: {final_thigh_len:.1f}, Shank: {final_shank_len:.1f}")

        # --- Pass 3: Re-calculation (Knee Anchoring & Rigid Hip) ---
        final_angles = []
        
        for frame in self.history_data:
            # 1. 무릎(Knee) 중심 좌표 계산 (글로벌 앵커 포인트)
            knee_z_metric = frame['knee']['z'] * base_k
            knee_z_center = knee_z_metric + stable_radii['knee']
            knee_3d = {'x': frame['knee']['x'], 'y': frame['knee']['y'], 'z': knee_z_center}
            
            # 2. 발목(Ankle) 강체 역전개 (무릎 앵커 기준)
            shank_len = frame['shank_len_2d']
            if final_shank_len > shank_len:
                 z_needed_shank = np.sqrt(final_shank_len**2 - shank_len**2)
            else:
                 z_needed_shank = 0.0
            direction_shank = 1 if frame['ankle']['z'] > frame['knee']['z'] else -1
            ankle_z_center = knee_z_center + (direction_shank * z_needed_shank)
            ankle_3d = {'x': frame['ankle']['x'], 'y': frame['ankle']['y'], 'z': ankle_z_center}
            
            # 3. 골반(Hip) 강체 역전개 (무릎 앵커 기준)
            thigh_len = frame['thigh_len_2d']
            
            if final_thigh_len > thigh_len:
                 z_needed_thigh = np.sqrt(final_thigh_len**2 - thigh_len**2)
            else:
                 z_needed_thigh = 0.0
                 
            direction_hip = 1 if frame['hip']['z'] > frame['knee']['z'] else -1
            hip_z_center = knee_z_center + (direction_hip * z_needed_thigh)
            hip_3d = {'x': frame['hip']['x'], 'y': frame['hip']['y'], 'z': hip_z_center}
            
            # 4. Calculate Final 3D Angle
            angle_3d = self.calculate_knee_angle(hip_3d, knee_3d, ankle_3d)
            final_angles.append(angle_3d)
            
        return self.counter, final_angles, {'base_k': base_k, 'dynamic_dr_shank': dynamic_dr_shank if 'dynamic_dr_shank' in locals() else 0.0}

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
                        depth_map = 1.0 - ((depth_map - d_min) / (d_max - d_min))
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

                # --- HISTORY STORAGE ---
                # Store data for 2-Pass Global Optimization
                self.history_data.append({
                     'hip': hip, 'knee': knee, 'ankle': ankle,
                     'shank_len_2d': np.linalg.norm([ankle['x'] - knee['x'], ankle['y'] - knee['y']]),
                     'thigh_len_2d': np.linalg.norm([knee['x'] - hip['x'], knee['y'] - hip['y']]),
                     'raw_image': frame.copy(),
                     'drawn_image': image_draw.copy() # Store the MediaPipe rendered frame
                })
                
                # Apply Base Z Scale (1.0 for Pass 1 live preview)
                hip_s = hip.copy(); knee_s = knee.copy(); ankle_s = ankle.copy()

                # --- [NEW] JOINT CENTER CORRECTION ---
                correction_meta = {
                    'thigh_len': self.history_data[-1]['thigh_len_2d'],
                    'shank_len': self.history_data[-1]['shank_len_2d'],
                    'max_thigh_len': self.history_data[-1]['thigh_len_2d'], 
                    'max_shank_len': self.history_data[-1]['shank_len_2d']
                }
                
                surface_joints = {'hip': hip_s, 'knee': knee_s, 'ankle': ankle_s}
                center_joints = self.corrector.correct(surface_joints, correction_meta)
                
                hip_c = center_joints['hip']
                knee_c = center_joints['knee']
                ankle_c = center_joints['ankle']

                # --- KNEE ANGLE LOGIC (SIMPLE 2-PHASE) ---
                knee_angle_3d = self.calculate_knee_angle(hip_c, knee_c, ankle_c)
                surface_angle_3d = self.calculate_knee_angle(hip_s, knee_s, ankle_s)
                
                TH_DOWN = SQUAT_THRESHOLDS['KNEE_ANGLE_DOWN']
                TH_UP = SQUAT_THRESHOLDS['KNEE_ANGLE_UP']
                
                if knee_angle_3d < TH_DOWN:
                    self.state = "DOWN"
                
                if self.state == "DOWN" and knee_angle_3d > TH_UP:
                    self.counter += 1
                    self.state = "UP"
                             
                # Visualization
                self.pose_estimator._draw_landmarks(image_draw, landmarks_list)
                
                # Update the stored drawn_image with the landmarks
                self.history_data[-1]['drawn_image'] = image_draw.copy()
                
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
        print(f"  > Count: {self.counter}, Side: {self.side.upper() if self.side else 'UNKNOWN'}")
        return self.counter, False, [], []

    def create_multiview_video(self, output_path, base_k, dr, stable_radii):
        """
        Generates a Multi-View Video (Original | 45-deg | 90-deg Side).
        Visualizes the CORRECTED 3D skeleton from different angles.
        [UPDATED] Centers the skeleton on the HIP to prevent it from going out of frame.
        """
        if not self.history_data:
            print("No history data to visualize.")
            return

        print(f"Generating Multi-View Video with Auto-Centering to {output_path}...")
        
        # Video Writer Setup
        sample_frame = self.history_data[0]['raw_image']
        h, w = sample_frame.shape[:2]
        
        # Output width will be 3x original (3 panels)
        out_w = w * 3
        out_h = h
        
        # Scale factor for drawing (optional, to make sure it fits)
        DRAW_SCALE = 0.8
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (out_w, out_h))
        
        # Helper: Project 3D point to 2D canvas with rotation and centering
        # center_pt ensures rotation happens around a specific 3D point (e.g., hip)
        def project_point(pt3d, angle_deg, offset_x, pivot_3d):
            # 1. Translate point so pivot is at (0,0,0)
            x_rel = (pt3d['x'] - pivot_3d['x']) * DRAW_SCALE
            y_rel = (pt3d['y'] - pivot_3d['y']) * DRAW_SCALE
            z_rel = (pt3d['z'] - pivot_3d['z']) * DRAW_SCALE # Z is depth relative to pivot
            
            # 2. Rotate around Y-axis
            rad = np.radians(angle_deg)
            # x' = x cos - z sin
            # z' = x sin + z cos
            x_rot = x_rel * np.cos(rad) - z_rel * np.sin(rad)
            # y' = y (unchanged)
            
            # 3. Map back to screen (Ankle-Centered Logic)
            # We want the ankle (pivot) to be at the bottom-center of the panel
            cx = w // 2
            cy = h - 50 # 50 pixels from bottom
            
            final_x = int(x_rot + cx) + offset_x
            final_y = int(y_rel + cy) 
            
            return (final_x, final_y)

        # Draw Skeleton Helper
        def draw_skeleton(img, joints, angle, offset_x, pivot_name='ankle'):
            # Pivot point for rotation (default: ankle for grounded look)
            pivot_pt = joints[pivot_name]
            
            # Connections
            connections = [
                ('shoulder', 'hip'),
                ('hip', 'knee'),
                ('knee', 'ankle')
            ]
            
            # Project all joints first
            proj_joints = {}
            for name, pt in joints.items():
                proj_joints[name] = project_point(pt, angle, offset_x, pivot_pt)
                
            # Draw Ground Line (Reference)
            # Draw a horizontal line at ankle height to represent the floor
            floor_y = proj_joints['ankle'][1]
            cv2.line(img, (offset_x + 20, floor_y), (offset_x + w - 20, floor_y), (100, 100, 100), 2)
            
            # Draw Lines (Bones)
            for start, end in connections:
                 if start in proj_joints and end in proj_joints:
                     pt1 = proj_joints[start]
                     pt2 = proj_joints[end]
                     cv2.line(img, pt1, pt2, (255, 255, 255), 4) # White Bone
            
            # Draw Points (Joints)
            for name, pt in proj_joints.items():
                color = (0, 255, 255) # Yellow
                if name == 'knee': color = (0, 0, 255) # Red Knee
                if name == 'hip': color = (255, 0, 0) # Blue Hip
                cv2.circle(img, pt, 8, color, -1)

        obs_max_thigh = max(f['thigh_len_2d'] for f in self.history_data)
        obs_max_shank = max(f['shank_len_2d'] for f in self.history_data)
        final_thigh_len = max(obs_max_thigh, obs_max_shank * 1.2)

        # Frame Loop
        for frame in self.history_data:
            # 1. 무릎(Knee) 글로벌 앵커 포인트
            knee_z_metric = frame['knee']['z'] * base_k
            knee_z_center = knee_z_metric + stable_radii['knee']
            
            # 2. 길이 제약을 이용한 양방향 역전개 (Hip & Ankle)
            thigh_len = frame['thigh_len_2d']
            shank_len = frame['shank_len_2d']
            
            if final_thigh_len > thigh_len:
                 z_needed_thigh = np.sqrt(final_thigh_len**2 - thigh_len**2)
            else:
                 z_needed_thigh = 0.0
                 
            if final_shank_len > shank_len:
                 z_needed_shank = np.sqrt(final_shank_len**2 - shank_len**2)
            else:
                 z_needed_shank = 0.0
                 
            direction_hip = 1 if frame['hip']['z'] > frame['knee']['z'] else -1
            direction_shank = 1 if frame['ankle']['z'] > frame['knee']['z'] else -1
            
            hip_z_center = knee_z_center + (direction_hip * z_needed_thigh)
            ankle_z_center = knee_z_center + (direction_shank * z_needed_shank)
            
            # Use hip z for shoulder approx if not tracked perfectly
            shoulder_z = hip_z_center 
            
            # Create 3D Joint Dict
            joints_3d = {
                'hip': {'x': frame['hip']['x'], 'y': frame['hip']['y'], 'z': hip_z_center},
                'knee': {'x': frame['knee']['x'], 'y': frame['knee']['y'], 'z': knee_z_center},
                'ankle': {'x': frame['ankle']['x'], 'y': frame['ankle']['y'], 'z': ankle_z_center},
            }
            # Add shoulder if available
            if 'shoulder' in frame:
                 joints_3d['shoulder'] = {'x': frame['shoulder']['x'], 'y': frame['shoulder']['y'], 'z': shoulder_z}

            # 2. Prepare Canvas
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            
            # Panel 1: Original (RGB)
            if frame['raw_image'] is not None:
                canvas[:h, :w] = frame['raw_image']
            
            # Panel 2: 45 Degree View
            # Background
            cv2.rectangle(canvas, (w, 0), (2*w, h), (40, 40, 40), -1) # Dark Gray
            draw_skeleton(canvas, joints_3d, 45, w)
            cv2.putText(canvas, "45-Deg View (Ankle-Centered)", (w+20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Panel 3: 90 Degree Side View
            cv2.rectangle(canvas, (2*w, 0), (3*w, h), (20, 20, 20), -1) # Almost Black
            draw_skeleton(canvas, joints_3d, 90, 2*w)
            cv2.putText(canvas, "90-Deg Side View", (2*w+20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            out.write(canvas)
            
        out.release()
        print(f"Multi-View Video Generated: {output_path}")

if __name__ == "__main__":
    analyzer = SquatAnalyzer()
    video_path = "dataset/true/true_14.mp4"
    if os.path.exists(video_path):
        analyzer.process_video(video_path, output_path="output_knee_corrected.mp4", show_window=False)
        # Run 2-Pass Check
        analyzer.finalize_analysis()
    else:
        print(f"Video not found: {video_path}")
