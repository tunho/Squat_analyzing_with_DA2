from __future__ import annotations

import argparse
import json
from pathlib import Path

from squat.analyzer import (
    MPWorldLenStableSquatAnalyzer,
    MPWorldOutlierCorrectedSquatAnalyzer,
    MPWorldRawSquatAnalyzer,
    MPWorldSelectiveBurstRepairSquatAnalyzer,
    MPWorldSelectiveHingeRepairSquatAnalyzer,
    MPWorldSelectiveRepairSquatAnalyzer,
    MPWorldSmoothSquatAnalyzer,
    MPWorldVisibilityFilteredSquatAnalyzer,
)


def build_analyzer(name: str):
    if name == "mp_world_raw":
        return MPWorldRawSquatAnalyzer()
    if name == "mp_world_smooth":
        return MPWorldSmoothSquatAnalyzer(smooth_alpha=0.2)
    if name == "mp_world_len_stable":
        return MPWorldLenStableSquatAnalyzer()
    if name == "mp_world_visibility_filtered":
        return MPWorldVisibilityFilteredSquatAnalyzer()
    if name == "mp_world_outlier_corrected":
        return MPWorldOutlierCorrectedSquatAnalyzer()
    if name == "mp_world_selective_repair":
        return MPWorldSelectiveRepairSquatAnalyzer()
    if name == "mp_world_selective_burst_repair":
        return MPWorldSelectiveBurstRepairSquatAnalyzer()
    if name == "mp_world_selective_hinge_repair":
        return MPWorldSelectiveHingeRepairSquatAnalyzer()
    raise ValueError(f"Unknown algorithm: {name}")


def run_one(
    algorithm: str,
    video_path: str,
    output_dir: Path,
    force_side: str | None,
    start_frame: int | None,
    end_frame: int | None,
    max_frames: int | None,
    save_video: bool,
    store_debug_images: bool,
    store_depth_maps: bool,
) -> dict:
    analyzer = build_analyzer(algorithm)

    algo_dir = output_dir / algorithm
    algo_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = str(algo_dir / f"{algorithm}_output.mp4") if save_video else None

    video_info = analyzer.process_video(
        video_path=video_path,
        output_path=output_video_path,
        max_frames=max_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        force_side=force_side,
        save_video=save_video,
        store_debug_images=store_debug_images,
        store_depth_maps=store_depth_maps,
    )

    final_count, final_angles, extra = analyzer.finalize_analysis()

    result = {
        "algorithm": algorithm,
        "video_path": video_path,
        "side": video_info["side"],
        "num_processed_frames": video_info["num_processed_frames"],
        "process_video_count": video_info["count"],
        "final_count": final_count,
        "num_final_angles": len(final_angles),
        "first_10_angles": final_angles[:10],
        "last_10_angles": final_angles[-10:],
        "output_video_path": output_video_path,
        "extra": extra,
    }

    with open(algo_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Algorithm           : {algorithm}")
    print(f"Side                : {result['side']}")
    print(f"Processed frames    : {result['num_processed_frames']}")
    print(f"process_video count : {result['process_video_count']}")
    print(f"Final count         : {result['final_count']}")
    print(f"Final angle frames  : {result['num_final_angles']}")
    print(f"Output video        : {result['output_video_path']}")
    print(f"Result json         : {algo_dir / 'result.json'}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "mp_world_raw",
            "mp_world_smooth",
            "mp_world_len_stable",
            "mp_world_visibility_filtered",
            "mp_world_outlier_corrected",
            "mp_world_selective_repair",
            "mp_world_selective_burst_repair",
            "mp_world_selective_hinge_repair",
        ],
        help="Algorithms to run",
    )
    parser.add_argument("--output-dir", default="outputs/world_baselines")
    parser.add_argument("--force-side", choices=["left", "right"], default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-debug-images", action="store_true")
    parser.add_argument("--store-depth-maps", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for algorithm in args.algorithms:
        result = run_one(
            algorithm=algorithm,
            video_path=args.video,
            output_dir=output_dir,
            force_side=args.force_side,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_frames=args.max_frames,
            save_video=args.save_video,
            store_debug_images=not args.no_debug_images,
            store_depth_maps=args.store_depth_maps,
        )
        summary.append(result)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
