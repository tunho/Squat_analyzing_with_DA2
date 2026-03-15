from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_pairwise_command(
    pair_runner: str,
    primary_algorithm: str,
    secondary_algorithm: str,
    video: str,
    gt_npy: str,
    sides: str,
    left_hip_idx: int,
    left_knee_idx: int,
    left_ankle_idx: int,
    right_hip_idx: int,
    right_knee_idx: int,
    right_ankle_idx: int,
    chunk_size: int,
    start_frame: int,
    last_frame: int,
    rolling_window: int,
    base_output_dir: Path,
    stop_on_error: bool,
    min_visibility: float | None,
    top_percent: float | None,
    trim_low: float | None,
    trim_high: float | None,
) -> list[str]:
    cmd = [
        sys.executable,
        pair_runner,
        "--video",
        video,
        "--gt-npy",
        gt_npy,
        "--primary-algorithm",
        primary_algorithm,
        "--secondary-algorithm",
        secondary_algorithm,
        "--sides",
        sides,
        "--left-hip-idx",
        str(left_hip_idx),
        "--left-knee-idx",
        str(left_knee_idx),
        "--left-ankle-idx",
        str(left_ankle_idx),
        "--right-hip-idx",
        str(right_hip_idx),
        "--right-knee-idx",
        str(right_knee_idx),
        "--right-ankle-idx",
        str(right_ankle_idx),
        "--chunk-size",
        str(chunk_size),
        "--start-frame",
        str(start_frame),
        "--last-frame",
        str(last_frame),
        "--rolling-window",
        str(rolling_window),
        "--base-output-dir",
        str(base_output_dir),
    ]
    if stop_on_error:
        cmd.append("--stop-on-error")
    if min_visibility is not None:
        cmd.extend(["--min-visibility", str(min_visibility)])
    if top_percent is not None:
        cmd.extend(["--top-percent", str(top_percent)])
    if trim_low is not None:
        cmd.extend(["--trim-low", str(trim_low)])
    if trim_high is not None:
        cmd.extend(["--trim-high", str(trim_high)])
    return cmd


def parse_chunk_from_dirname(name: str) -> tuple[int | None, int | None]:
    parts = name.split("_")
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def collect_pairwise_rows(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(base_dir.rglob("*_summary.json")):
        rel = path.relative_to(base_dir)
        parts = rel.parts
        if len(parts) < 4:
            continue
        pair_name = parts[0]
        side = parts[1]
        chunk_dir = parts[2]
        start_frame, end_frame = parse_chunk_from_dirname(chunk_dir)
        if "_vs_" in pair_name:
            primary_algorithm, secondary_algorithm = pair_name.split("_vs_", 1)
        else:
            primary_algorithm, secondary_algorithm = pair_name, ""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "pair": pair_name,
                "primary_algorithm": data.get("primary_algorithm", primary_algorithm),
                "secondary_algorithm": data.get("secondary_algorithm", secondary_algorithm),
                "side": side,
                "chunk_dir": chunk_dir,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "aligned_frames": data.get("aligned_frames"),
                "primary_mae": data.get("primary_mae"),
                "secondary_mae": data.get("secondary_mae"),
                "primary_gt2d_mae": data.get("primary_gt2d_mae"),
                "secondary_gt2d_mae": data.get("secondary_gt2d_mae"),
                "mp_world_mae": data.get("mp_world_mae"),
                "summary_json": str(path),
                "angle_plot_path": data.get("angle_plot_path"),
                "error_plot_path": data.get("error_plot_path"),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["primary_algorithm"]),
            str(row["secondary_algorithm"]),
            str(row["side"]),
            row["start_frame"] if row["start_frame"] is not None else 10**18,
            row["end_frame"] if row["end_frame"] is not None else 10**18,
        )
    )
    return rows


def write_pairwise_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair",
        "primary_algorithm",
        "secondary_algorithm",
        "side",
        "chunk_dir",
        "start_frame",
        "end_frame",
        "aligned_frames",
        "primary_mae",
        "secondary_mae",
        "primary_gt2d_mae",
        "secondary_gt2d_mae",
        "mp_world_mae",
        "summary_json",
        "angle_plot_path",
        "error_plot_path",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_console_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("No pairwise summary rows found.")
        return
    print(f"Total pairwise rows: {len(rows)}")
    print("-" * 140)
    for row in rows:
        p_mae = row["primary_mae"]
        s_mae = row["secondary_mae"]
        m_mae = row["mp_world_mae"]
        p_text = "None" if p_mae is None else f"{float(p_mae):.6f}"
        s_text = "None" if s_mae is None else f"{float(s_mae):.6f}"
        m_text = "None" if m_mae is None else f"{float(m_mae):.6f}"
        print(
            f"{row['primary_algorithm']:>16} vs {row['secondary_algorithm']:<16} | "
            f"{row['side']:<5} | {row['chunk_dir']} | "
            f"primary_mae={p_text} | secondary_mae={s_text} | mp_world_mae={m_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run all pairwise chunked dual comparisons for a list of algorithms, "
            "then collect all *_summary.json files into one CSV."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt-npy", required=True)
    parser.add_argument(
        "--algorithms",
        required=True,
        help="Comma-separated algorithm names, e.g. relative_linear,relative_add,model_a,model_b,mp_world_linear",
    )
    parser.add_argument("--sides", default="left,right")
    parser.add_argument("--left-hip-idx", type=int, default=16)
    parser.add_argument("--left-knee-idx", type=int, default=17)
    parser.add_argument("--left-ankle-idx", type=int, default=18)
    parser.add_argument("--right-hip-idx", type=int, default=21)
    parser.add_argument("--right-knee-idx", type=int, default=22)
    parser.add_argument("--right-ankle-idx", type=int, default=23)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--last-frame", type=int, required=True)
    parser.add_argument("--rolling-window", type=int, default=15)
    parser.add_argument("--min-visibility", type=float, default=None)
    parser.add_argument("--top-percent", type=float, default=None)
    parser.add_argument("--trim-low", type=float, default=None)
    parser.add_argument("--trim-high", type=float, default=None)
    parser.add_argument(
        "--pair-runner",
        default="/mnt/data/run_chunked_dual_compare.py",
        help="Path to run_chunked_dual_compare.py",
    )
    parser.add_argument(
        "--base-output-dir",
        default="outputs/all_pairwise_chunked_compare",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional output CSV path. Defaults to <base-output-dir>/pairwise_chunked_summary.csv",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    algorithms = parse_comma_list(args.algorithms)
    if len(algorithms) < 2:
        raise ValueError("At least two algorithms are required for pairwise comparison.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.last_frame < args.start_frame:
        raise ValueError("--last-frame must be >= --start-frame.")

    pair_runner = Path(args.pair_runner)
    if not pair_runner.exists():
        raise FileNotFoundError(f"Pair runner not found: {pair_runner}")

    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv) if args.summary_csv else base_output_dir / "pairwise_chunked_summary.csv"

    pairs = list(itertools.combinations(algorithms, 2))
    failures: list[tuple[str, str, int]] = []
    total_runs = 0

    print("=" * 100)
    print(f"Algorithms ({len(algorithms)}): {', '.join(algorithms)}")
    print(f"Pair count      : {len(pairs)}")
    print(f"Chunk size      : {args.chunk_size}")
    print(f"Frame range     : {args.start_frame}-{args.last_frame}")
    print(f"Base output dir : {base_output_dir}")
    print("=" * 100)

    for primary_algorithm, secondary_algorithm in pairs:
        cmd = build_pairwise_command(
            pair_runner=str(pair_runner),
            primary_algorithm=primary_algorithm,
            secondary_algorithm=secondary_algorithm,
            video=args.video,
            gt_npy=args.gt_npy,
            sides=args.sides,
            left_hip_idx=args.left_hip_idx,
            left_knee_idx=args.left_knee_idx,
            left_ankle_idx=args.left_ankle_idx,
            right_hip_idx=args.right_hip_idx,
            right_knee_idx=args.right_knee_idx,
            right_ankle_idx=args.right_ankle_idx,
            chunk_size=args.chunk_size,
            start_frame=args.start_frame,
            last_frame=args.last_frame,
            rolling_window=args.rolling_window,
            base_output_dir=base_output_dir,
            stop_on_error=args.stop_on_error,
            min_visibility=args.min_visibility,
            top_percent=args.top_percent,
            trim_low=args.trim_low,
            trim_high=args.trim_high,
        )
        total_runs += 1
        print("=" * 100)
        print(f"[{total_runs}/{len(pairs)}] {primary_algorithm} vs {secondary_algorithm}")
        print(" ".join(cmd))
        print("=" * 100)
        if args.dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((primary_algorithm, secondary_algorithm, result.returncode))
            if args.stop_on_error:
                raise SystemExit(result.returncode)

    if args.dry_run:
        print("Dry run complete. No subprocesses were executed.")
        return

    rows = collect_pairwise_rows(base_output_dir)
    write_pairwise_csv(rows, summary_csv)
    print_console_summary(rows)
    print("-" * 100)
    print(f"Pairwise summary CSV saved to: {summary_csv}")
    print(f"Finished pairwise chunked comparison. Pair runs: {total_runs}, failures: {len(failures)}")
    if failures:
        print("Failed pair runs:")
        for primary_algorithm, secondary_algorithm, returncode in failures:
            print(f"  {primary_algorithm} vs {secondary_algorithm} returncode={returncode}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
