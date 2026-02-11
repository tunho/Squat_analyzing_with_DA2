import cv2
import os
from pose_estimation import PoseEstimator
from pathlib import Path

def visualize_sample(video_path, output_dir="visualization"):
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    estimator = PoseEstimator(static_image_mode=False, model_complexity=2) # 고정밀 모드
    
    # 첫 번째 유효한 프레임 찾기
    success, frame = cap.read()
    if not success:
        print("Error: Could not read video")
        return
        
    # 중간 프레임으로 이동 (동작을 하고 있을 확률이 높음)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    success, frame = cap.read()
    
    results, output_image = estimator.process_frame(frame)
    
    output_path = os.path.join(output_dir, "skeleton_sample.jpg")
    cv2.imwrite(output_path, output_image)
    print(f"Visualization saved to: {output_path}")
    
    cap.release()

if __name__ == "__main__":
    sample_video = "/home/lee/exe_est/dataset/true/true_14.mp4"
    visualize_sample(sample_video)
