"""
데이터셋에서 MediaPipe Pose로 키포인트만 추출하여 JSON으로 저장합니다.
이미지 폴더 또는 영상 폴더를 입력으로 받습니다.
추후 스켈레톤/각도 계산 시 이 JSON을 읽어 사용할 수 있습니다.
"""

import argparse
import json
import os
from pathlib import Path

import cv2

from pose_estimation import PoseEstimator

# 지원 확장자
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_from_image(estimator, image_path, output_path):
    """단일 이미지에서 키포인트 추출 후 JSON 저장."""
    image = cv2.imread(str(image_path))
    if image is None:
        return False, "이미지 로드 실패"
    h, w, _ = image.shape
    landmarks = estimator.extract_keypoints_only(image)
    data = {
        "source": str(image_path),
        "image_width": w,
        "image_height": h,
        "frame_index": 0,
        "landmarks": landmarks,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True, len(landmarks) if landmarks else 0


def extract_from_video(estimator, video_path, output_path, frame_skip=1):
    """영상의 각 프레임에서 키포인트 추출 후 한 JSON에 저장."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "영상 열기 실패"

    # 영상 정보 가져오기
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval_ms = 1000.0 / fps

    frames_data = []
    frame_index = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        if frame_index % frame_skip != 0:
            frame_index += 1
            continue
        
        timestamp_ms = int(frame_index * frame_interval_ms)
        landmarks = estimator.extract_keypoints_only(image, frame_timestamp_ms=timestamp_ms)
        frames_data.append({
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "landmarks": landmarks,
        })
        frame_index += 1
    cap.release()

    data = {
        "source": str(video_path),
        "image_width": w,
        "image_height": h,
        "frame_skip": frame_skip,
        "num_frames": len(frames_data),
        "frames": frames_data,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True, len(frames_data)


def run(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 모드: 입력 디렉터리 내 이미지 파일만 처리
    if args.mode == "images":
        image_paths = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(input_dir.glob(f"*{ext}"))
        image_paths = sorted(image_paths)

        if not image_paths:
            print(f"이미지를 찾을 수 없습니다: {input_dir}")
            return

        estimator = PoseEstimator(static_image_mode=True, model_complexity=args.model_complexity)
        for i, path in enumerate(image_paths):
            out_path = output_dir / f"{path.stem}.json"
            ok, msg = extract_from_image(estimator, path, out_path)
            if ok:
                print(f"[{i+1}/{len(image_paths)}] {path.name} -> {out_path.name} (landmarks: {msg})")
            else:
                print(f"[{i+1}/{len(image_paths)}] {path.name} 실패: {msg}")
        return

    # 영상 모드: 입력 디렉터리 내 영상 파일만 처리
    if args.mode == "videos":
        video_paths = []
        for ext in VIDEO_EXTENSIONS:
            video_paths.extend(input_dir.glob(f"*{ext}"))
        video_paths = sorted(video_paths)

        if not video_paths:
            print(f"영상을 찾을 수 없습니다: {input_dir}")
            return

        estimator = PoseEstimator(static_image_mode=False, model_complexity=args.model_complexity)
        for i, path in enumerate(video_paths):
            out_path = output_dir / f"{path.stem}.json"
            ok, msg = extract_from_video(
                estimator, path, out_path, frame_skip=args.frame_skip
            )
            if ok:
                print(f"[{i+1}/{len(video_paths)}] {path.name} -> {out_path.name} (frames: {msg})")
            else:
                print(f"[{i+1}/{len(video_paths)}] {path.name} 실패: {msg}")
        return

    print("--mode는 'images' 또는 'videos' 중 하나를 지정하세요.")
    return


def main():
    parser = argparse.ArgumentParser(description="데이터셋에서 키포인트만 추출하여 JSON 저장")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="이미지 또는 영상이 있는 폴더 경로")
    parser.add_argument("--output_dir", "-o", type=str, default="keypoints", help="키포인트 JSON을 저장할 폴더 (기본: keypoints)")
    parser.add_argument("--mode", "-m", type=str, choices=["images", "videos"], required=True,
                        help="입력 데이터 타입: images 또는 videos")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="영상 모드에서 N프레임마다 추출 (기본: 1, 모두 추출)")
    parser.add_argument("--model_complexity", type=int, choices=[0, 1, 2], default=1,
                        help="MediaPipe 모델 복잡도 (0: 빠름, 2: 정확)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
