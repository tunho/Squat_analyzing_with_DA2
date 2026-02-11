"""
MediaPipe 0.10+ Tasks API 기반 포즈 추정 (BlazePose).
모델은 .task 파일을 사용하며, 없으면 자동 다운로드합니다.
"""

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# MediaPipe 0.10 Tasks API
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.core import base_options as base_options_lib

# 프로젝트 루트 기준 모델 경로
_MODEL_DIR = Path(__file__).resolve().parent / "models"
_POSE_MODEL_FILENAME = "pose_landmarker_lite.task"
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _get_model_path():
    """모델 파일 경로 반환. 없으면 다운로드."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / _POSE_MODEL_FILENAME
    if not path.is_file():
        print(f"모델 다운로드 중: {_POSE_MODEL_URL}")
        urllib.request.urlretrieve(_POSE_MODEL_URL, path)
        print(f"저장됨: {path}")
    return str(path)


class PoseEstimator:
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_path=None,
        **kwargs,
    ):
        """
        MediaPipe Pose (BlazePose) 초기화 — Tasks API 사용.

        Args:
            static_image_mode: 이미지 처리용 True, 비디오용 False
            model_complexity: 호환용 무시 (0.10에서는 모델 파일로 구분)
            min_detection_confidence: 검출 임계값
            min_tracking_confidence: 추적 임계값
            model_path: .task 모델 경로. None이면 기본 모델 자동 다운로드
        """
        self._static_image_mode = static_image_mode
        model_path = model_path or _get_model_path()
        base_options = base_options_lib.BaseOptions(model_asset_path=model_path)
        run_mode = VisionTaskRunningMode.IMAGE if static_image_mode else VisionTaskRunningMode.VIDEO
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=run_mode,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process_frame(self, image):
        """
        이미지 프레임에서 포즈 추출 (시각화용: 랜드마크 그린 이미지 반환).
        """
        landmarks_list = self.extract_keypoints_only(image)
        image_draw = image.copy()
        if landmarks_list:
            self._draw_landmarks(image_draw, landmarks_list)
        return type("Results", (), {"pose_landmarks": None, "_landmarks_list": landmarks_list})(), image_draw

    def _draw_landmarks(self, image, landmarks_list):
        """랜드마크를 이미지에 그리기 (간단한 원 + 선)."""
        # POSE_CONNECTIONS 유사 (MediaPipe 33점)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (13, 23), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (14, 24),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
            (24, 26), (26, 28), (28, 30), (30, 32),
        ]
        h, w = image.shape[:2]
        pts = []
        for lm in landmarks_list:
            x = int(lm["x"] * w)
            y = int(lm["y"] * h)
            pts.append((x, y))
        for i, j in connections:
            if i < len(pts) and j < len(pts):
                cv2.line(image, pts[i], pts[j], (0, 255, 0), 1)
        for (x, y) in pts:
            cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

    def get_landmarks_as_list(self, results):
        """process_frame 반환값에서 랜드마크 리스트 추출."""
        if hasattr(results, "_landmarks_list"):
            return results._landmarks_list
        return None

    def set_frame_timestamp_ms(self, timestamp_ms):
        """영상 모드에서 다음 프레임 타임스탬프 설정 (ms)."""
        self._frame_timestamp_ms = timestamp_ms

    def extract_keypoints_only(self, image, frame_timestamp_ms=None):
        """
        이미지에서 키포인트만 추출 (시각화 없음). 데이터셋 배치 처리용.
        
        Args:
            image (numpy.ndarray): OpenCV BGR 이미지
            frame_timestamp_ms (int, optional): 비디오 모드일 때 현재 프레임의 타임스탬프
            
        Returns:
            list | None: 랜드마크 리스트 또는 검출 실패 시 None
        """
        if image is None or image.size == 0:
            return None
        
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        if self._static_image_mode:
            result = self.landmarker.detect(mp_image)
        else:
            ts = frame_timestamp_ms if frame_timestamp_ms is not None else self._frame_timestamp_ms
            result = self.landmarker.detect_for_video(mp_image, ts)
            self._frame_timestamp_ms = ts + 33  # ~30fps 가정
        if not result.pose_landmarks:
            return None, result # Return None for landmarks, but the result object
        landmarks = []
        for idx, lm in enumerate(result.pose_landmarks[0]):
            vis = lm.visibility if lm.visibility is not None else (lm.presence or 0.0)
            landmarks.append({
                "id": idx,
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z) if lm.z is not None else 0.0,
                "visibility": float(vis) if vis is not None else 0.0,
            })
        return landmarks, result # Return both landmarks and the result object

    def close(self):
        if hasattr(self.landmarker, "close"):
            self.landmarker.close()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    estimator = PoseEstimator(static_image_mode=False)
    print("q를 누르면 종료됩니다.")
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            results, output_image = estimator.process_frame(image)
            cv2.imshow("MediaPipe Pose Demo", output_image)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    finally:
        estimator.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
