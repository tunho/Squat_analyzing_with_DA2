import cv2
import torch
import numpy as np
import os
import glob
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pose_estimation import PoseEstimator

def generate_batch_demos():
    print("=== Batch Demo Generation Started ===")
    
    # 1. Setup Folders
    output_dir = "demo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"Loading Depth Anything on {device}...")
    processor = AutoImageProcessor.from_pretrained(depth_model_id)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id).to(device)
    
    print("Loading MediaPipe...")
    pose_estimator = PoseEstimator(static_image_mode=False, model_complexity=1)
    
    # 3. Connections
    connections = [
        (11, 12), (11, 23), (12, 24), (23, 24), # Torso
        (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
    ]

    # 4. Get All Videos
    video_files = glob.glob("dataset/**/*.mp4", recursive=True)
    print(f"Found {len(video_files)} videos.")

    for i, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        save_path = os.path.join(output_dir, f"demo_{filename}")
        
        print(f"[{i+1}/{len(video_files)}] Processing {filename}...")
        
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Resize for speed if too large (Optional, but recommended for demo gen speed)
        scale = 1.0
        if w > 1280: scale = 0.5
        w_draw, h_draw = int(w * scale), int(h * scale)

        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_draw * 2, h_draw))
        
        # RAZS State (Reset per video)
        max_shank_len_2d = 0.0
        max_thigh_len_2d = 0.0
        z_scale_param = 1.0
        selected_side = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if scale != 1.0:
                frame = cv2.resize(frame, (w_draw, h_draw))
            
            # Inference
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks, _ = pose_estimator.extract_keypoints_only(frame.copy())
            
            # Depth Map
            inputs = processor(images=Image.fromarray(image_rgb), return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = depth_model(**inputs)
                raw_depth = outputs.predicted_depth.squeeze().cpu().numpy()
            
            # Normalize Depth (0-1)
            d_min, d_max = raw_depth.min(), raw_depth.max()
            if d_max - d_min > 1e-6:
                depth_map = (raw_depth - d_min) / (d_max - d_min)
            else:
                depth_map = np.zeros_like(raw_depth)
            
            # Resize Depth to Frame
            depth_map_resized = cv2.resize(depth_map, (w_draw, h_draw))
            
            # Viz Depth
            depth_viz = (depth_map_resized * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)
            
            # Helper
            def get_pt(idx):
                lm = landmarks[idx]
                px, py = int(lm['x']*w_draw), int(lm['y']*h_draw)
                px = max(0, min(w_draw-1, px))
                py = max(0, min(h_draw-1, py))
                pz = depth_map_resized[py, px]
                return px, py, pz, lm['x'], lm['y']

            if landmarks:
                # Auto Side Selection (Once)
                if selected_side is None:
                    l_vis = landmarks[23]['visibility'] + landmarks[25]['visibility']
                    r_vis = landmarks[24]['visibility'] + landmarks[26]['visibility']
                    if l_vis > 0.5 or r_vis > 0.5:
                         selected_side = "left" if l_vis >= r_vis else "right"
                
                side = selected_side if selected_side else "left" # Default
                h_idx, k_idx, a_idx = (23, 25, 27) if side == "left" else (24, 26, 28)
                
                try:
                    h_x, h_y, h_z, h_nx, h_ny = get_pt(h_idx)
                    k_x, k_y, k_z, k_nx, k_ny = get_pt(k_idx)
                    a_x, a_y, a_z, a_nx, a_ny = get_pt(a_idx)
                    
                    # 1. Update Max Lengths
                    curr_shank = np.sqrt((a_x - k_x)**2 + (a_y - k_y)**2)
                    curr_thigh = np.sqrt((k_x - h_x)**2 + (k_y - h_y)**2)
                    
                    if curr_shank > max_shank_len_2d: max_shank_len_2d = curr_shank
                    if curr_thigh > max_thigh_len_2d: max_thigh_len_2d = curr_thigh
                    
                    # 2. RAZS Calculation
                    target_estimate = max_shank_len_2d * 1.2 # Fallback
                    # If standing (confident), use Max Thigh as truth
                    if max_thigh_len_2d > 0: target_estimate = max(target_estimate, max_thigh_len_2d)
                    
                    dZ_raw = (a_z - k_z)
                    
                    instant_k = z_scale_param
                    if abs(dZ_raw) > 1e-6 and target_estimate > curr_shank:
                         k_sq = (target_estimate**2 - curr_shank**2) / (dZ_raw**2)
                         if k_sq > 0: instant_k = np.sqrt(k_sq)
                    
                    # Smoothing
                    z_scale_param = 0.9 * z_scale_param + 0.1 * instant_k
                    
                    # 3. Angles
                    # 2D
                    dx, dy = k_x - h_x, k_y - h_y
                    l2 = np.sqrt(dx**2 + dy**2)
                    ang2 = np.degrees(np.arccos(dy/l2)) if l2 > 0 else 0
                    
                    # 3D
                    dz = (k_z - h_z) * z_scale_param
                    l3 = np.sqrt(dx**2 + dy**2 + dz**2)
                    ang3 = np.degrees(np.arccos(dy/l3)) if l3 > 0 else 0
                    
                    # Draw
                    cv2.putText(frame, f"Side: {side.upper()}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.putText(frame, f"2D: {int(ang2)} deg", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    cv2.putText(frame, f"3D: {int(ang3)} deg", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    
                    cv2.line(frame, (h_x, h_y), (k_x, k_y), (0, 255, 0), 3)
                    for pt in [(h_x, h_y), (k_x, k_y), (a_x, a_y)]:
                        cv2.circle(frame, pt, 5, (0, 255, 255), -1)

                except Exception as e: pass

            combined = np.hstack((frame, depth_color))
            out.write(combined)
        
        cap.release()
        out.release()

if __name__ == "__main__":
    generate_batch_demos()
