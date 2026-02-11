import cv2

video_path = "/home/lee/exe_est/test/true/true_46.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open video")
else:
    ret, frame = cap.read()
    if ret:
        output_path = "/home/lee/exe_est/debug_frame.jpg"
        cv2.imwrite(output_path, frame)
        print(f"Saved frame to {output_path} with shape {frame.shape}")
    else:
        print("Failed to read first frame")
cap.release()
