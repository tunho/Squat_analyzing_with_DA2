from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from squat.config import MODEL_CONFIG
from squat.domain.types import SquatFrame
from squat.io.video import create_video_writer, open_video_capture, read_video_info
from squat.models.depth_anything import DepthAnythingEstimator
from squat.pipeline.frame_processor import process_frame
from squat.pose_estimation import PoseEstimator
from squat.state.squat_counter import SquatCounter
from squat.visualization.dashboard import draw_dashboard


class BaseSquatAnalyzer(ABC):
    def __init__(
        self,
        pose_estimator: PoseEstimator | None = None,
        depth_estimator: DepthAnythingEstimator | None = None,
        visibility_threshold: float = 0.1,
    ) -> None:
        self.pose_estimator = pose_estimator or PoseEstimator(
            static_image_mode=False,
            model_complexity=MODEL_CONFIG["POSE_MODEL_COMPLEXITY"],
        )
        self.depth_estimator = depth_estimator or DepthAnythingEstimator()
        self.visibility_threshold = visibility_threshold
        self.reset()

    def reset(self) -> None:
        self.counter = SquatCounter()
        self.side: str | None = None
        self.history_data: list[SquatFrame] = []

    @abstractmethod
    def finalize_analysis(self) -> Any:
        raise NotImplementedError

    def process_video(
        self,
        video_path: str,
        output_path: str | None = None,
        max_frames: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        force_side: str | None = None,
        save_video: bool = True,
        store_debug_images: bool = True,
        store_depth_maps: bool = False,
    ) -> dict[str, Any]:
        self.reset()
        if force_side is not None:
            self.side = force_side

        cap = open_video_capture(video_path)
        fps, width, height = read_video_info(cap)
        out = create_video_writer(
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
            save_video=save_video,
        )

        frame_idx = 0

        try:
            while cap.isOpened():
                if max_frames is not None and frame_idx >= max_frames:
                    break

                success, frame = cap.read()
                if not success:
                    break

                curr_idx = frame_idx
                frame_idx += 1

                if start_frame is not None and curr_idx < start_frame:
                    continue
                if end_frame is not None and curr_idx > end_frame:
                    break

                processed = process_frame(
                    frame=frame,
                    frame_index=curr_idx,
                    side=self.side,
                    pose_estimator=self.pose_estimator,
                    depth_estimator=self.depth_estimator,
                    store_debug_images=store_debug_images,
                    store_depth_maps=store_depth_maps,
                    visibility_threshold=self.visibility_threshold,
                )
                self.side = processed.resolved_side

                output_frame = processed.output_frame
                if processed.squat_frame is not None:
                    frame_angle = self.compute_frame_angle(processed.squat_frame)
                    if frame_angle is None:
                        frame_angle = processed.raw_angle

                    if frame_angle is not None:
                        self.counter.update(frame_angle)
                        output_frame = draw_dashboard(
                            output_frame,
                            self.counter.count,
                            self.counter.state,
                            frame_angle,
                        )


                    processed.squat_frame.drawn_image = output_frame.copy() if store_debug_images else None
                    self.history_data.append(processed.squat_frame)

                if out is not None:
                    out.write(output_frame)

        finally:
            cap.release()
            if out is not None:
                out.release()

        return {
            "count": self.counter.count,
            "side": self.side,
            "num_processed_frames": len(self.history_data),
            "video_path": video_path,
            "output_path": output_path,
            "fps": fps,
            "width": width,
            "height": height,
        }

    def analyze_video(
        self,
        video_path: str,
        output_path: str | None = None,
        max_frames: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        force_side: str | None = None,
        save_video: bool = True,
        store_debug_images: bool = True,
        store_depth_maps: bool = False,
    ) -> Any:
        self.process_video(
            video_path=video_path,
            output_path=output_path,
            max_frames=max_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            force_side=force_side,
            save_video=save_video,
            store_debug_images=store_debug_images,
            store_depth_maps=store_depth_maps,
        )
        return self.finalize_analysis()
    
    def compute_frame_angle(self, squat_frame: SquatFrame) -> float | None:
        return None
