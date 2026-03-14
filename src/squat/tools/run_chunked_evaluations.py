from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def build_command(
    algorithm: str,
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
        '-m',
        'squat.tools.evaluate_algorithm_vs_gt2d_injection',
        '--algorithm',
        algorithm,
        '--video',
        video,
        '--gt-npy',
        gt_npy,
        '--hip-idx',
        str(hip_idx),
        '--knee-idx',
        str(knee_idx),
        '--ankle-idx',
        str(ankle_idx),
        '--force-side',
        side,
        '--start-frame',
        str(start_frame),
        '--end-frame',
        str(end_frame),
        '--rolling-window',
        str(rolling_window),
        '--output-dir',
        str(output_dir),
    ]


class SideSpec:
    def __init__(self, hip_idx: int, knee_idx: int, ankle_idx: int) -> None:
        self.hip_idx = hip_idx
        self.knee_idx = knee_idx
        self.ankle_idx = ankle_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run evaluation for multiple algorithms, sides, and frame chunks.'
    )
    parser.add_argument('--video', required=True)
    parser.add_argument('--gt-npy', required=True)
    parser.add_argument('--algorithms', default='relative_linear,relative_add,relative_mul')
    parser.add_argument('--sides', default='left,right')
    parser.add_argument('--left-hip-idx', type=int, default=16)
    parser.add_argument('--left-knee-idx', type=int, default=17)
    parser.add_argument('--left-ankle-idx', type=int, default=18)
    parser.add_argument('--right-hip-idx', type=int, default=21)
    parser.add_argument('--right-knee-idx', type=int, default=22)
    parser.add_argument('--right-ankle-idx', type=int, default=23)
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--last-frame', type=int, required=True)
    parser.add_argument('--rolling-window', type=int, default=15)
    parser.add_argument('--base-output-dir', default='outputs/chunked_eval')
    parser.add_argument('--stop-on-error', action='store_true')
    args = parser.parse_args()

    algorithms = parse_comma_list(args.algorithms)
    sides = parse_comma_list(args.sides)
    if not algorithms:
        raise ValueError('At least one algorithm is required.')
    if not sides:
        raise ValueError('At least one side is required.')
    if args.chunk_size <= 0:
        raise ValueError('--chunk-size must be positive.')
    if args.last_frame < args.start_frame:
        raise ValueError('--last-frame must be >= --start-frame.')

    side_specs = {
        'left': SideSpec(args.left_hip_idx, args.left_knee_idx, args.left_ankle_idx),
        'right': SideSpec(args.right_hip_idx, args.right_knee_idx, args.right_ankle_idx),
    }

    base_output_dir = Path(args.base_output_dir)
    failures: list[tuple[str, str, int, int, int]] = []
    total_runs = 0

    for algorithm in algorithms:
        current = args.start_frame
        while current <= args.last_frame:
            end_frame = min(current + args.chunk_size - 1, args.last_frame)

            for side in sides:
                if side not in side_specs:
                    raise ValueError(f'Unsupported side={side!r}. Use left and/or right.')
                spec = side_specs[side]
                output_dir = base_output_dir / algorithm / side / f'{current:06d}_{end_frame:06d}'
                cmd = build_command(
                    algorithm=algorithm,
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
                print('=' * 100)
                print(f'[{total_runs}] algorithm={algorithm} side={side} frames={current}-{end_frame}')
                print(f'output_dir={output_dir}')
                print(' '.join(cmd))
                print('=' * 100)

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    failures.append((algorithm, side, current, end_frame, result.returncode))
                    if args.stop_on_error:
                        raise SystemExit(result.returncode)

            current += args.chunk_size

    print('\nFinished chunked evaluation.')
    print(f'Total runs    : {total_runs}')
    print(f'Failure count : {len(failures)}')
    if failures:
        print('Failed runs:')
        for algorithm, side, start_frame, end_frame, returncode in failures:
            print(
                f'  algorithm={algorithm} side={side} '
                f'frames={start_frame}-{end_frame} returncode={returncode}'
            )
        raise SystemExit(1)


if __name__ == '__main__':
    main()
