import cv2
import mediapipe as mp
import json
import numpy as np
import os
from scipy.spatial import distance

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def run_test():
    subject = "s03"
    exercise = "squat"
    camera = "50591643"
    
    video_path = f"/home/lee/exe_est/dataset/raw/fit3d/fit3d_train/train/{subject}/videos/{camera}/{exercise}.mp4"
    json_path = f"/home/lee/exe_est/dataset/raw/fit3d/fit3d_train/train/{subject}/joints3d_25/{exercise}.json"
    
    if not os.path.exists(video_path) or not os.path.exists(json_path):
        print("Missing files")
        return

    # Load GT
    with open(json_path, 'r') as f:
        gt_data = json.load(f)['joints3d_25']
    
    print(f"Total GT frames: {len(gt_data)}")

    # 프로젝트 루트 경로 설정
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from squat.pose_estimation import PoseEstimator
    from squat.pipeline.frame_processor import _extract_world_landmarks

    pose_estimator = PoseEstimator(static_image_mode=False)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total frames: {total_frames}")

    mp_angles_l = []
    mp_angles_r = []
    gt_angles_1 = []
    gt_angles_2 = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(gt_data):
            break
            
        # Run MediaPipe
        extracted = pose_estimator.extract_keypoints_only(frame)
        landmarks, res_obj = (extracted if isinstance(extracted, tuple) else (extracted, None))
        
        if landmarks:
            world_lms = _extract_world_landmarks(res_obj)
            if world_lms:
                # Left Leg
                ml_hip = [world_lms[23]['x'], world_lms[23]['y'], world_lms[23]['z']]
                ml_knee = [world_lms[25]['x'], world_lms[25]['y'], world_lms[25]['z']]
                ml_ankle = [world_lms[27]['x'], world_lms[27]['y'], world_lms[27]['z']]
                ml_angle = calculate_angle(ml_hip, ml_knee, ml_ankle)
                
                # Right Leg
                mr_hip = [world_lms[24]['x'], world_lms[24]['y'], world_lms[24]['z']]
                mr_knee = [world_lms[26]['x'], world_lms[26]['y'], world_lms[26]['z']]
                mr_ankle = [world_lms[28]['x'], world_lms[28]['y'], world_lms[28]['z']]
                mr_angle = calculate_angle(mr_hip, mr_knee, mr_ankle)

                # Ground Truth side 1 (Joints 1, 2, 3)
                g1_hip = gt_data[frame_idx][1]
                g1_knee = gt_data[frame_idx][2]
                g1_ankle = gt_data[frame_idx][3]
                g1_angle = calculate_angle(g1_hip, g1_knee, g1_ankle)
                
                # Ground Truth side 2 (Joints 4, 5, 6)
                g2_hip = gt_data[frame_idx][4]
                g2_knee = gt_data[frame_idx][5]
                g2_ankle = gt_data[frame_idx][6]
                g2_angle = calculate_angle(g2_hip, g2_knee, g2_ankle)
                
                mp_angles_l.append(ml_angle)
                mp_angles_r.append(mr_angle)
                gt_angles_1.append(g1_angle)
                gt_angles_2.append(g2_angle)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    pose_estimator.close()
    
    if len(mp_angles_l) == 0:
        print("No pose detected")
        return

    mp_l = np.array(mp_angles_l)
    mp_r = np.array(mp_angles_r)
    gt_1 = np.array(gt_angles_1)
    gt_2 = np.array(gt_angles_2)
    
    mae_l1 = np.mean(np.abs(mp_l - gt_1))
    mae_l2 = np.mean(np.abs(mp_l - gt_2))
    mae_r1 = np.mean(np.abs(mp_r - gt_1))
    mae_r2 = np.mean(np.abs(mp_r - gt_2))
    
    print(f"\n--- Fit3D Test Results ({subject}, {exercise}, {camera}) ---")
    print(f"MAE (MP-L, GT-1): {mae_l1:.2f}")
    print(f"MAE (MP-L, GT-2): {mae_l2:.2f}")
    print(f"MAE (MP-R, GT-1): {mae_r1:.2f}")
    print(f"MAE (MP-R, GT-2): {mae_r2:.2f}")
    
    # Identify which index belongs to which leg
    if mae_l2 < mae_l1:
        print("\nSide Mapping Suggestion: GT Side 2 is Left Leg, GT Side 1 is Right Leg")
    else:
        print("\nSide Mapping Suggestion: GT Side 1 is Left Leg, GT Side 2 is Right Leg")
    
    print(f"Best MAE: {min(mae_l1, mae_l2, mae_r1, mae_r2):.2f} degrees")

if __name__ == "__main__":
    run_test()
