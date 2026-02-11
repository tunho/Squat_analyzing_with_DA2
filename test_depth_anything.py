import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def test_depth():
    # 1. 모델 로드 (Depth Anything V2 Small - 가볍고 빠름)
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"Loading model: {model_id}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    
    # 2. 영상 로드 및 프레임 추출
    video_path = "test/true/true_46.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    cap.release()
    
    # BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    
    # 3. 깊이 추정
    print("Inference...")
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
        
    # 4. 결과 후처리 (시각화용)
    # Resize to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=pil_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    
    output = prediction.squeeze().cpu().numpy()
    
    # Normalize to 0-255 for saving
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth_image = Image.fromarray(formatted)
    
    depth_image.save("debug_depth.png")
    print("Depth map saved as debug_depth.png")

if __name__ == "__main__":
    test_depth()
