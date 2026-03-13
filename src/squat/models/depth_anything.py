from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from squat.config import MODEL_CONFIG


@dataclass(slots=True)
class DepthAnythingEstimator:
    model_id: str = MODEL_CONFIG["DEPTH_MODEL_ID"]
    device: str = MODEL_CONFIG["DEVICE"]
    processor: Any = field(init=False)
    model: Any = field(init=False)

    def __post_init__(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device)

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
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