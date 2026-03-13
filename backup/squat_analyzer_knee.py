import cv2
import os
import mediapipe as mp
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from src.pose_estimation import PoseEstimator
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from PIL import Image
from src.config import MODEL_CONFIG, SQUAT_THRESHOLDS, OUTPUT_CONFIG


@dataclass
class SquatFrame:
    """Dataclass to hold data for a single frame of squat analysis."""
    frame_index: int
    side: str
    hip: Dict[str, float]
    knee: Dict[str, float]
    ankle: Dict[str, float]
    shoulder: Optional[Dict[str, float]] = None
    shank_len_2d: float = 0.0
    thigh_len_2d: float = 0.0
    raw_image: Optional[np.ndarray] = None
    drawn_image: Optional[np.ndarray] = None
    depth_map_half: Optional[np.ndarray] = None
    orig_w: int = 0
    orig_h: int = 0


def fit_shank_regression(
    raw_dz_shank_list: list[float],
    thigh_len_2d_list: list[float],
    target_shank_dz_list: list[float],
    obs_max_thigh: float,
    obs_max_shank: float,
) -> tuple[dict, dict]:
    """
    Fit additive and multiplicative models.
    Additive: pred_dz = a_add * raw_dz + c_add * thigh_len_2d + b_add
    Multiplicative: k = a_k + c_k * thigh_len_2d, pred_dz = raw_dz * k

    Returns:
        params: dict with a_add, c_add, b_add, a_k, c_k, final_thigh_len, final_shank_len
    """
    final_thigh_len = max(obs_max_thigh, obs_max_shank * 1.2)
    final_shank_len = final_thigh_len / 1.2

    a_add, c_add, b_add = 1.0, 0.0, 0.0
    a_k, c_k = 1.0, 0.0

    if len(raw_dz_shank_list) > 2:
        X = np.array(raw_dz_shank_list)
        D = np.array(thigh_len_2d_list)
        Y = np.array(target_shank_dz_list)

        # Additive: [raw_dz, thigh_len_2d, 1] -> target_dz
        A_add = np.column_stack([X, D, np.ones_like(X)])
        coef_add, _, _, _ = np.linalg.lstsq(A_add, Y, rcond=None)
        a_add = float(coef_add[0])
        c_add = float(coef_add[1])
        b_add = float(coef_add[2])
        if a_add <= 0:
            a_add = abs(a_add) if a_add != 0 else 1.0

        # Multiplicative: k_i = target_dz_i / raw_dz_i, exclude |raw_dz| <= 1e-6
        valid_mask = np.abs(X) > 1e-6 # 수치적 폭주 방지
        k_list = Y[valid_mask] / X[valid_mask]
        D_valid = D[valid_mask]
        if len(k_list) > 2:
            coef_k = np.polyfit(D_valid, k_list, 1)  # k = c_k * D + a_k
            a_k = float(coef_k[1])
            c_k = float(coef_k[0])
            if a_k <= 0:
                a_k = abs(a_k) if a_k != 0 else 1.0

    params = {
        "a_add": a_add,
        "c_add": c_add,
        "b_add": b_add,
        "a_k": a_k,
        "c_k": c_k,
        "final_thigh_len": final_thigh_len,
        "final_shank_len": final_shank_len,
    }
    rigid = {"final_thigh_len": final_thigh_len, "final_shank_len": final_shank_len}
    return params, rigid


def predict_shank_dz(raw_dz_shank: float, thigh_len_2d: float, params: dict, formula: str) -> float:
    """Predict shank dz: additive (a*dz + c*dist + b) or multiplicative (raw_dz * k, k = a_k + c_k * thigh)."""
    if formula == "additive":
        return params["a_add"] * raw_dz_shank + params["c_add"] * thigh_len_2d + params["b_add"]
    # multiplicative
    pred_k = params["a_k"] + params["c_k"] * thigh_len_2d
    return raw_dz_shank * pred_k


def predict_shank_dz_with_k(raw_dz_shank: float, thigh_len_2d: float, params: dict) -> tuple[float, float]:
    """Return (pred_dz, pred_k) for multiplicative model. Used for debug/sanity."""
    pred_k = params["a_k"] + params["c_k"] * thigh_len_2d
    pred_dz = raw_dz_shank * pred_k
    return pred_dz, pred_k


class SquatAnalyzer:
    def __init__(self):
        # MediaPipe Pose 초기화
        self.pose_estimator = PoseEstimator(
            static_image_mode=False, 
            model_complexity=MODEL_CONFIG['POSE_MODEL_COMPLEXITY']
        )
        
        # Depth Anything V2 초기화
        self._init_depth_model()

        # 상태 관리 (FSM)
        self.reset()

    def _init_depth_model(self):
        """Initialize Depth Anything V2 model and processor."""
        print("Loading Depth Anything V2-Small (Knee Mode)...")
        self.depth_model_id = MODEL_CONFIG['DEPTH_MODEL_ID']
        self.device = MODEL_CONFIG['DEVICE']
        self.processor = AutoImageProcessor.from_pretrained(self.depth_model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_model_id).to(self.device)
        print("Depth Anything Model Loaded.")

    def reset(self):
        """Reset state for batch processing"""
        self.state = "UP" # UP, DOWN
        self.counter = 0
        self.feedback = ""
        self.side = None
        self.history_data: List[SquatFrame] = []

    def finalize_analysis(self):
        """
        Pass 2: Fit additive and multiplicative models.
        Pass 3: Compute knee angles for both models.
        Returns (counter, final_angles_additive, final_angles_multiplicative, params).
        """
        if not self.history_data:
            return self.counter, [], [], {}

        obs_max_thigh = max(f.thigh_len_2d for f in self.history_data)
        obs_max_shank = max(f.shank_len_2d for f in self.history_data)
        target_shank = max(obs_max_thigh, obs_max_shank * 1.2) / 1.2

        raw_dz_list: list[float] = []
        thigh_list: list[float] = []
        target_list: list[float] = []
        for frame in self.history_data:
            shank_len = frame.shank_len_2d
            
            if target_shank > shank_len:
                z_needed = np.sqrt(target_shank**2 - shank_len**2)
                dz = frame.ankle['z'] - frame.knee['z']
                if abs(dz) > 1e-6:
                    signed = z_needed if dz > 0 else -z_needed
                    raw_dz_list.append(dz)
                    thigh_list.append(frame.thigh_len_2d)
                    target_list.append(signed)

        params, _ = fit_shank_regression(raw_dz_list, thigh_list, target_list, obs_max_thigh, obs_max_shank)
        final_thigh_len = params["final_thigh_len"]
        final_shank_len = params["final_shank_len"]

        if len(raw_dz_list) > 2 and os.environ.get("SQUAT_DEBUG"):
            print(f"DEBUG: Shank Regression fit with {len(raw_dz_list)} pairs.")
            print(f"DEBUG: additive:     a_add={params['a_add']:.3f}, c_add={params['c_add']:.4f}, b_add={params['b_add']:.1f}")
            print(f"DEBUG: multiplicative: k = a_k + c_k*dist, a_k={params['a_k']:.3f}, c_k={params['c_k']:.4f}")
        
        final_angles_additive = []
        final_angles_multiplicative = []
        pred_k_all: list[float] = []

        for idx, frame in enumerate(self.history_data):
            thigh_len_2d = frame.thigh_len_2d
            raw_dz_thigh = frame.hip['z'] - frame.knee['z']
            raw_dz_shank = frame.ankle['z'] - frame.knee['z']
            
            pred_dz_additive = predict_shank_dz(raw_dz_shank, thigh_len_2d, params, "additive")
            pred_dz_mult, pred_k = predict_shank_dz_with_k(raw_dz_shank, thigh_len_2d, params)
            pred_k_all.append(pred_k)

            z_needed_thigh = np.sqrt(max(0, final_thigh_len**2 - thigh_len_2d**2))
            direction_hip = 1 if raw_dz_thigh >= 0 else -1
            thigh_dz = direction_hip * z_needed_thigh

            knee_3d = {'x': frame.knee['x'], 'y': frame.knee['y'], 'z': 0.0}
            hip_3d = {'x': frame.hip['x'], 'y': frame.hip['y'], 'z': thigh_dz}

            ankle_add = {'x': frame.ankle['x'], 'y': frame.ankle['y'], 'z': pred_dz_additive}
            ankle_mult = {'x': frame.ankle['x'], 'y': frame.ankle['y'], 'z': pred_dz_mult}
            final_angles_additive.append(self.calculate_knee_angle(hip_3d, knee_3d, ankle_add))
            final_angles_multiplicative.append(self.calculate_knee_angle(hip_3d, knee_3d, ankle_mult))

        return self.counter, final_angles_additive, final_angles_multiplicative, params

    @staticmethod
    def calculate_knee_angle(hip, knee, ankle):
        """Calculate 3D Knee Angle (Hip-Knee-Ankle)."""
        v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y'], hip['z'] - knee['z']])
        v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y'], ankle['z'] - knee['z']])
        
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 180.0
            
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def _get_depth_map(self, frame: np.ndarray, depth_estimator: Optional[Any] = None) -> np.ndarray:
        """Estimate depth map."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        model_source = depth_estimator if depth_estimator else self
        
        if model_source and hasattr(model_source, 'depth_model'):
            inputs = model_source.processor(images=pil_image, return_tensors="pt").to(model_source.device)
            with torch.no_grad():
                outputs = model_source.depth_model(**inputs)
            prediction = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1), size=pil_image.size[::-1], mode="bicubic", align_corners=False,
            )
            depth_map = prediction.squeeze().cpu().numpy()
            d_min, d_max = depth_map.min(), depth_map.max()
            if d_max - d_min > 1e-6:
                depth_map = 1.0 - ((depth_map - d_min) / (d_max - d_min))
                depth_map *= 1000.0
            return depth_map
        return np.zeros(frame.shape[:2])

    def _get_3d_point(self, landmark: Dict[str, float], depth_map: np.ndarray) -> Dict[str, float]:
        h, w = depth_map.shape
        px, py = int(landmark['x'] * w), int(landmark['y'] * h)
        px, py = max(0, min(w - 1, px)), max(0, min(h - 1, py))
        return {'x': px, 'y': py, 'z': float(depth_map[py, px]), 'vis': landmark.get('visibility', 0.0)}

    def _update_state(self, angle: float):
        if angle < SQUAT_THRESHOLDS['KNEE_ANGLE_DOWN']: self.state = "DOWN"
        if self.state == "DOWN" and angle > SQUAT_THRESHOLDS['KNEE_ANGLE_UP']:
            self.counter += 1
            self.state = "UP"

    def _draw_dashboard(self, image: np.ndarray, angle: float):
        cv2.rectangle(image, (0,0), (400, 150), (245, 117, 16), -1) 
        cv2.putText(image, f'REPS: {self.counter}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.putText(image, f'STATE: {self.state}', (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(image, f'Knee 3D: {int(angle)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    def process_video(
        self, video_path: str, output_path: str = "output.mp4", depth_estimator: Any = None,
        max_frames: int = None, start_frame: int = None, end_frame: int = None, 
        force_side: str = None, save_video: bool = True, store_debug_images: bool = True, 
        store_depth_maps: bool = False
    ) -> Tuple[int, bool, List, List]:
        self.reset()
        if force_side: self.side = force_side
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 0, False, [], []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 100 or fps <= 0: fps = OUTPUT_CONFIG.get('VIDEO_FPS', 30.0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = None
        if save_video:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        print(f"Processing {video_path}...")
        frame_idx = 0
        while cap.isOpened():
            if max_frames and frame_idx >= max_frames: break
            success, frame = cap.read()
            if not success: break
            
            curr_idx, frame_idx = frame_idx, frame_idx + 1
            if start_frame and curr_idx < start_frame: continue
            if end_frame and curr_idx > end_frame: break

            landmarks, _ = self.pose_estimator.extract_keypoints_only(frame.copy())
            if landmarks:
                depth_map = self._get_depth_map(frame, depth_estimator)
                if self.side is None:
                    l_vis = (landmarks[11]['visibility'] + landmarks[23]['visibility'] + landmarks[25]['visibility']) / 3
                    r_vis = (landmarks[12]['visibility'] + landmarks[24]['visibility'] + landmarks[26]['visibility']) / 3
                    self.side = "left" if l_vis >= r_vis else "right"

                idxs = {"left": (23,25,27,11), "right": (24,26,28,12)}[self.side]
                hip, knee, ankle, shoulder = [self._get_3d_point(landmarks[i], depth_map) for i in idxs]

                if min(hip['vis'], knee['vis'], ankle['vis']) > 0.1:
                    image_draw = frame.copy()
                    angle = self.calculate_knee_angle(hip, knee, ankle)
                    self._update_state(angle)
                    self.pose_estimator._draw_landmarks(image_draw, landmarks)
                    self._draw_dashboard(image_draw, angle)
                    
                    self.history_data.append(SquatFrame(
                        frame_index=curr_idx, side=self.side, hip=hip, knee=knee, ankle=ankle, shoulder=shoulder,
                        shank_len_2d=np.linalg.norm([ankle['x']-knee['x'], ankle['y']-knee['y']]),
                        thigh_len_2d=np.linalg.norm([knee['x']-hip['x'], knee['y']-hip['y']]),
                        raw_image=frame.copy() if store_debug_images else None,
                        drawn_image=image_draw.copy() if store_debug_images else None,
                        depth_map_half=cv2.resize(depth_map, (w//2, h//2)).astype(np.float16) if store_depth_maps else None,
                        orig_w=w, orig_h=h
                    ))
                    if out: out.write(image_draw)
                else:
                    if out: out.write(frame)
            else:
                if out: out.write(frame)
            
        cap.release()
        if out: out.release()
        return self.counter, False, [], []

    def create_multiview_video(self, output_path, base_k, dr, stable_radii):
        if not self.history_data: return
        print(f"Generating Multi-View Video to {output_path}...")
        h, w = self.history_data[0].orig_h, self.history_data[0].orig_w
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w * 3, h))
        
        def project(pt, angle, offset, pivot):
            x, y, z = (pt['x']-pivot['x'])*0.8, (pt['y']-pivot['y'])*0.8, (pt['z']-pivot['z'])*0.8
            rad = np.radians(angle)
            return int(x*np.cos(rad) - z*np.sin(rad) + w//2) + offset, int(y + h - 50)

        t_max = max(f.thigh_len_2d for f in self.history_data)
        s_max = max(f.shank_len_2d for f in self.history_data)
        f_thigh = max(t_max, s_max * 1.2)
        f_shank = f_thigh / 1.2

        for f in self.history_data:
            kz = f.knee['z'] * base_k + stable_radii['knee']
            tz = np.sqrt(max(0, f_thigh**2 - f.thigh_len_2d**2))
            sz = np.sqrt(max(0, f_shank**2 - f.shank_len_2d**2))
            
            joints = {
                'hip': {'x': f.hip['x'], 'y': f.hip['y'], 'z': kz + (1 if f.hip['z'] > f.knee['z'] else -1)*tz},
                'knee': {'x': f.knee['x'], 'y': f.knee['y'], 'z': kz},
                'ankle': {'x': f.ankle['x'], 'y': f.ankle['y'], 'z': kz + (1 if f.ankle['z'] > f.knee['z'] else -1)*sz},
                'shoulder': {'x': f.shoulder['x'], 'y': f.shoulder['y'], 'z': kz + (1 if f.hip['z'] > f.knee['z'] else -1)*tz} if f.shoulder else None
            }
            
            canvas = np.zeros((h, w*3, 3), dtype=np.uint8)
            if f.raw_image is not None: canvas[:h, :w] = f.raw_image
            for i, ang in enumerate([45, 90]):
                off = (i+1)*w
                cv2.rectangle(canvas, (off, 0), (off+w, h), (40-i*20, 40-i*20, 40-i*20), -1)
                piv = joints['ankle']
                proj = {k: project(v, ang, off, piv) for k, v in joints.items() if v}
                cv2.line(canvas, (off+20, proj['ankle'][1]), (off+w-20, proj['ankle'][1]), (100,100,100), 2)
                for s, e in [('shoulder','hip'), ('hip','knee'), ('knee','ankle')]:
                    if s in proj and e in proj: cv2.line(canvas, proj[s], proj[e], (255,255,255), 4)
                for k, p in proj.items():
                    cv2.circle(canvas, p, 8, (0,0,255) if k=='knee' else (255,0,0) if k=='hip' else (0,255,255), -1)
            out.write(canvas)
        out.release()

if __name__ == "__main__":
    analyzer = SquatAnalyzer()
    video_path = "dataset/true/true_14.mp4"
    if os.path.exists(video_path):
        analyzer.process_video(video_path)
        analyzer.finalize_analysis()
