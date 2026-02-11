import cv2
from pose_estimation import PoseEstimator

estimator = PoseEstimator(static_image_mode=False, model_complexity=2)

video_path = "/home/lee/exe_est/dataset/true/true_14.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open")
else:
    print(f"Opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    ret, frame = cap.read()
    print(f"First frame read: {ret}")
    if ret:
        print(f"Frame shape: {frame.shape}")
cap.release()
