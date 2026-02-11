import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pose_estimation import PoseEstimator

def main():
    print("=== 1. Loading Models ===")
    
    # Depth Anything V2 Load
    print("Loading Depth Anything V2-Small...")
    depth_model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    processor = AutoImageProcessor.from_pretrained(depth_model_id)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id).to(device)
    
    # MediaPipe Load
    print("Loading MediaPipe Pose...")
    pose_estimator = PoseEstimator(static_image_mode=False, model_complexity=1)
    
    print("=== Models Loaded ===")

    # Helper: 2D Angle Calculation (Original)
    def calculate_vertical_angle_2d(p1, p2):
        # p1: Hip, p2: Knee
        # vertical is (0, 1)
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        vec_len = np.sqrt(dx**2 + dy**2)
        if vec_len == 0: return 0
        return np.degrees(np.arccos(np.clip(dy / vec_len, -1.0, 1.0)))

    # Helper: 3D Projected Angle Calculation (New)
    def calculate_projected_angle_3d(p1, p2):
        # p1: Hip, p2: Knee
        # 3D Vector Length (True Length of Thigh)
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        # Scaling Z: Depth Anything outputs raw disparity/depth. 
        # For this test, we assume direct usage to see stability.
        # We scale Z by 50 to make it comparable to pixel coords 
        dz = (p2['z'] - p1['z']) * 50.0
        
        length_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if length_3d == 0: return 0
        
        # Projected Angle: arccos(delta_y / length_3d)
        # This projects the 3D bone onto the vertical axis
        # effectively giving "Side View Angle" where Z-variation (and X-variation) is handled 
        # by the length denominator.
        return np.degrees(np.arccos(np.clip(dy / length_3d, -1.0, 1.0)))

    # 2. Video Source
    video_path = "dataset/false/false_74.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error reading {video_path}")
        return

    # Video Info
    w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing {video_path} ({w_video}x{h_video} @ {fps}fps)")
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        if frame_idx % 5 != 0:
            continue

        # 3. MediaPipe 2D Inference
        # Note: MediaPipe sets the input image to read-only. We pass a copy to keep 'frame' writable.
        landmarks = pose_estimator.extract_keypoints_only(frame.copy())
        
        if not landmarks:
            print(f"Frame {frame_idx}: No landmarks detected.")
            continue

        # 4. Depth Anything Inference
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Prepare inputs
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original image size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1], # PIL size is (W, H)
            mode="bicubic",
            align_corners=False,
        )
        
        depth_map = prediction.squeeze().cpu().numpy()
        
        # 5. Fusion
        h_map, w_map = depth_map.shape
        
        def get_3d_point(idx):
            lm = landmarks[idx]
            px = int(lm['x'] * w_map)
            py = int(lm['y'] * h_map)
            px = max(0, min(w_map - 1, px))
            py = max(0, min(h_map - 1, py))
            z_depth = depth_map[py, px]
            
            return {"id": idx, "x": px, "y": py, "z": float(z_depth)}

        hip = get_3d_point(23)
        knee = get_3d_point(25)
        
        # 6. Angle Comparison
        angle_2d = calculate_vertical_angle_2d(hip, knee)
        angle_3d_proj = calculate_projected_angle_3d(hip, knee)
        
        print(f"Frame {frame_idx}:")
        print(f"  [2D Angle]: {angle_2d:.2f}")
        print(f"  [3D Proj ]: {angle_3d_proj:.2f}")
        diff = abs(angle_2d - angle_3d_proj)
        print(f"  > Diff    : {diff:.2f}")
        # print("-" * 30)
        
        # Visualization
        cv2.putText(frame, f"2D: {int(angle_2d)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"3D: {int(angle_3d_proj)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Bridge 2D-3D Debug", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Processing Complete.")

if __name__ == "__main__":
    main()
