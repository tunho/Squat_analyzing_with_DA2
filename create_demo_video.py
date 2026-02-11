import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pose_estimation import PoseEstimator

def create_demo(video_path, output_name):
    print(f"=== Creating Skeleton + Depth Demo: {video_path} ===")
    
    # Standard MediaPipe Connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (13, 23), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (14, 24),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
    ]

    # 1. Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(depth_model_id)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id).to(device)
    pose_estimator = PoseEstimator(static_image_mode=False, model_complexity=1)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Video Writer (Side-by-Side: Skeleton + Depth Map)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, fps, (w * 2, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        # Inference
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2D Pose - Get landmarks and raw result for drawing
        landmarks, result = pose_estimator.extract_keypoints_only(frame.copy())
        
    # RAZS State for this video
    max_shank_len_2d = 0.0
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, _ = pose_estimator.extract_keypoints_only(frame.copy())
        
    # RAZS State for this video
    max_shank_len_2d = 0.0
    selected_side = None # Lock the side once detected
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, _ = pose_estimator.extract_keypoints_only(frame.copy())
        
        # Depth Map
        inputs = processor(images=Image.fromarray(image_rgb), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        )
        depth_map = prediction.squeeze().cpu().numpy()
        
        # Draw Skeleton
        if landmarks:
            pts = []
            for lm in landmarks:
                px, py = int(lm['x'] * w), int(lm['y'] * h)
                pts.append((px, py))
            for i, j in connections:
                if i < len(pts) and j < len(pts):
                    cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2)

            # --- SMART RAZS LOGIC ---
            try:
                # 1. Lock Side once 
                if selected_side is None:
                    l_vis = landmarks[23]['visibility'] + landmarks[25]['visibility']
                    r_vis = landmarks[24]['visibility'] + landmarks[26]['visibility']
                    if l_vis > 0.5 or r_vis > 0.5: # Wait for a clear detection
                        selected_side = "left" if l_vis >= r_vis else "right"
                        print(f"[{output_name}] Side Locked: {selected_side}")

                if selected_side:
                    h_idx, k_idx, a_idx = (23, 25, 27) if selected_side == "left" else (24, 26, 28)

                    def get_3d_val(idx):
                        lm = landmarks[idx]
                        px, py = int(lm['x']*w), int(lm['y']*h)
                        pz = depth_map[max(0,min(h-1,py)), max(0,min(w-1,px))]
                        return px, py, pz, lm['x'], lm['y']

                    h_x, h_y, h_z, h_nx, h_ny = get_3d_val(h_idx)
                    k_x, k_y, k_z, k_nx, k_ny = get_3d_val(k_idx)
                    a_x, a_y, a_z, a_nx, a_ny = get_3d_val(a_idx)
                    
                    # 1. Update Max Shank Length (Calibration)
                    curr_shank_2d = np.sqrt((a_x - k_x)**2 + (a_y - k_y)**2)
                    if curr_shank_2d > max_shank_len_2d:
                        max_shank_len_2d = curr_shank_2d
                    
                    # 2. Basic 2D Angle
                    dx_px, dy_px = k_x - h_x, k_y - h_y
                    len_2d = np.sqrt(dx_px**2 + dy_px**2)
                    angle_2d = np.degrees(np.arccos(np.clip(dy_px/len_2d, -1, 1))) if len_2d > 0 else 0
                    
                    # 3. RAZS K-Factor Estimation
                    k_razs = 1.0 # Default
                    raw_dz_shank = (a_z - k_z)
                    dz_shank_sq = raw_dz_shank**2
                    
                    if dz_shank_sq > 1e-6:
                        target_sq = max_shank_len_2d**2
                        shank_2d_sq = curr_shank_2d**2
                        if target_sq > shank_2d_sq:
                            k_sq = (target_sq - shank_2d_sq) / dz_shank_sq
                            k_razs = np.sqrt(k_sq)
                    
                    k_razs = max(0.1, min(2000.0, k_razs))
                    
                    # 4. Final 3D Angle with RAZS
                    dz_thigh = (k_z - h_z) * k_razs
                    len_3d = np.sqrt(dx_px**2 + dy_px**2 + dz_thigh**2)
                    angle_3d = np.degrees(np.arccos(np.clip(dy_px/len_3d, -1, 1))) if len_3d > 0 else 0
                    
                    # Draw Info
                    cv2.line(frame, (h_x, h_y), (k_x, k_y), (0, 255, 0), 4)
                    cv2.putText(frame, f"SIDE: {selected_side.upper()}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"RAZS K: {k_razs:.1f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"2D Angle: {int(angle_2d)}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.putText(frame, f"3D Angle: {int(angle_3d)}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            except: pass

        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        combined = np.hstack((frame, depth_color))
        out.write(combined)
        
        if frame_idx % 30 == 0: print(f"Processing Frame {frame_idx} (K={k_razs if 'k_razs' in locals() else 0:.1f})...")

    cap.release()
    out.release()
    print(f"Demo Saved: {output_name}")

    cap.release()
    out.release()
    print(f"Demo Saved: {output_name}")

if __name__ == "__main__":
    # Create Real RAZS Comparison Demos
    create_demo("dataset/false/false_1.mp4", "demo_razs_false_1.mp4")
    create_demo("dataset/true/true_28.mp4", "demo_razs_true_28.mp4")
    create_demo("dataset/true/true_39.mp4", "demo_razs_true_39.mp4")
