from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np
# torch·transformers·PIL 은 선택적 depth 모델에만 필요 → 사용 시점에 지연 import
# (코어 무릎각 파이프라인은 이 무거운 의존성 없이 동작)

from squat.config import MODEL_CONFIG


@dataclass(slots=True)
class DepthAnythingEstimator:
    model_id: str = MODEL_CONFIG["DEPTH_MODEL_ID"]
    device: str = MODEL_CONFIG["DEVICE"]
    processor: Any = field(init=False)
    model: Any = field(init=False)

    def __post_init__(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device)

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        prediction = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        depth_map = prediction.squeeze().cpu().numpy()

        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        if d_max - d_min > 1e-6:
            depth_map = 1.0 - ((depth_map - d_min) / (d_max - d_min))
            depth_map *= 1000.0

        return depth_map


@dataclass(slots=True)
class NullDepthEstimator:
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        return np.zeros((h, w), dtype=np.float32)