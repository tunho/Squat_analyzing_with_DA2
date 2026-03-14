from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class SideSpec:
    def __init__(self, hip_idx: int, knee_idx: int, ankle_idx: int) -> None:
        self.hip_idx = hip_idx
        self.knee_idx = knee_idx
        self.ankle_idx = ankle_idx


def build_command(
    primary_algorithm: str,
    secondary_algorithm: str,
    side: str,
    hip_idx: int,
    knee_idx: int,
    ankle_idx: int,
    video: str,
    gt_npy: str,
    start_frame: int,
    end_frame: int,
    rolling_window: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "squat.tools.compare_dual_algorithms_vs_gt",
        "--primary-algorithm",
        primary_algorithm,
        "--secondary-algorithm",
        secondary_algorithm,
        "--video",
        video,
        "--gt-npy",
        gt_npy,
        "--hip-idx",
        str(hip_idx),
        "--knee-idx",
        str(knee_idx),
        "--ankle-idx",
        str(ankle_idx),
        "--force-side",
        side,
        "--start-frame",
        str(start_frame),
        "--end-frame",
        str(end_frame),
        "--rolling-window",
        str(rolling_window),
        "--output-dir",
        str(output_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run chunked dual-algorithm comparison against GT."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt-npy", required=True)

    parser.add_argument("--primary-algorithm", required=True)
    parser.add_argument("--secondary-algorithm", required=True)

    parser.add_argument("--sides", default="left,right")
    parser.add_argument("--left-hip-idx", type=int, default=16)
    parser.add_argument("--left-knee-idx", type=int, default=17)
    parser.add_argument("--left-ankle-idx", type=int, default=18)
    parser.add_argument("--right-hip-idx", type=int, default=21)
    parser.add_argument("--right-knee-idx", type=int, default=22)
    parser.add_argument("--right-ankle-idx", type=int, default=23)

    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--last-frame", type=int, required=True)
    parser.add_argument("--rolling-window", type=int, default=15)
    parser.add_argument("--base-output-dir", default="outputs/chunked_dual_compare")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    sides = parse_comma_list(args.sides)
    if not sides:
        raise ValueError("At least one side is required.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.last_frame < args.start_frame:
        raise ValueError("--last-frame must be >= --start-frame.")

    side_specs = {
        "left": SideSpec(args.left_hip_idx, args.left_knee_idx, args.left_ankle_idx),
        "right": SideSpec(args.right_hip_idx, args.right_knee_idx, args.right_ankle_idx),
    }

    base_output_dir = Path(args.base_output_dir)
    failures: list[tuple[str, int, int, int]] = []
    total_runs = 0

    current = args.start_frame
    while current <= args.last_frame:
        end_frame = min(current + args.chunk_size - 1, args.last_frame)

        for side in sides:
            if side not in side_specs:
                raise ValueError(f"Unsupported side={side!r}. Use left and/or right.")
            spec = side_specs[side]

            output_dir = (
                base_output_dir
                / f"{args.primary_algorithm}_vs_{args.secondary_algorithm}"
                / side
                / f"{current:06d}_{end_frame:06d}"
            )

            cmd = build_command(
                primary_algorithm=args.primary_algorithm,
                secondary_algorithm=args.secondary_algorithm,
                side=side,
                hip_idx=spec.hip_idx,
                knee_idx=spec.knee_idx,
                ankle_idx=spec.ankle_idx,
                video=args.video,
                gt_npy=args.gt_npy,
                start_frame=current,
                end_frame=end_frame,
                rolling_window=args.rolling_window,
                output_dir=output_dir,
            )

            total_runs += 1
            print("=" * 100)
            print(
                f"[{total_runs}] side={side} "
                f"frames={current}-{end_frame} "
                f"{args.primary_algorithm} vs {args.secondary_algorithm}"
            )
            print(f"output_dir={output_dir}")
            print(" ".join(cmd))
            print("=" * 100)

            result = subprocess.run(cmd)
            if result.returncode != 0:
                failures.append((side, current, end_frame, result.returncode))
                if args.stop_on_error:
                    raise SystemExit(result.returncode)

        current += args.chunk_size

    print("\nFinished chunked dual comparison.")
    print(f"Total runs    : {total_runs}")
    print(f"Failure count : {len(failures)}")
    if failures:
        print("Failed runs:")
        for side, start_frame, end_frame, returncode in failures:
            print(
                f"  side={side} frames={start_frame}-{end_frame} "
                f"returncode={returncode}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()