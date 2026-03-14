from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_chunk_from_dirname(name: str) -> tuple[int | None, int | None]:
    parts = name.split('_')
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def find_summary_files(base_dir: Path) -> list[Path]:
    return sorted(base_dir.rglob('*_summary.json'))



def load_rows(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in find_summary_files(base_dir):
        rel = path.relative_to(base_dir)
        parts = rel.parts
        if len(parts) < 4:
            continue

        algorithm = parts[0]
        side = parts[1]
        chunk_dir = parts[2]
        start_frame, end_frame = parse_chunk_from_dirname(chunk_dir)

        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        rows.append(
            {
                'algorithm': algorithm,
                'side': side,
                'chunk_dir': chunk_dir,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'aligned_frames': data.get('aligned_frames'),
                'pred_mae': data.get('pred_mae'),
                'gt2d_injected_mae': data.get('gt2d_injected_mae'),
                'mp_world_mae': data.get('mp_world_mae'),
                'mp_world_aligned_frames': data.get('mp_world_aligned_frames'),
                'summary_json': str(path),
                'angle_plot_path': data.get('angle_plot_path'),
                'error_plot_path': data.get('error_plot_path'),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row['algorithm']),
            str(row['side']),
            row['start_frame'] if row['start_frame'] is not None else 10**18,
            row['end_frame'] if row['end_frame'] is not None else 10**18,
        )
    )
    return rows

def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'algorithm',
        'side',
        'chunk_dir',
        'start_frame',
        'end_frame',
        'aligned_frames',
        'pred_mae',
        'gt2d_injected_mae',
        'mp_world_mae',
        'mp_world_aligned_frames',
        'summary_json',
        'angle_plot_path',
        'error_plot_path',
    ]
    with output_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_console_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        print('No summary rows found.')
        return

    print(f'Total rows: {len(rows)}')
    print('-' * 130)
    for row in rows:
        pred_mae = row['pred_mae']
        gt2d_mae = row['gt2d_injected_mae']
        mp_world_mae = row['mp_world_mae']

        pred_text = 'None' if pred_mae is None else f'{float(pred_mae):.6f}'
        gt2d_text = 'None' if gt2d_mae is None else f'{float(gt2d_mae):.6f}'
        mp_world_text = 'None' if mp_world_mae is None else f'{float(mp_world_mae):.6f}'

        print(
            f"{row['algorithm']:>16} | {row['side']:<5} | "
            f"{row['chunk_dir']} | pred_mae={pred_text} | "
            f"gt2d_mae={gt2d_text} | mp_world_mae={mp_world_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Collect chunked evaluation summary JSON files into a single CSV.'
    )
    parser.add_argument('--base-output-dir', default='outputs/chunked_eval')
    parser.add_argument('--output-csv', default='outputs/chunked_eval/chunked_eval_summary.csv')
    args = parser.parse_args()

    base_dir = Path(args.base_output_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f'Base output directory does not exist: {base_dir}')

    rows = load_rows(base_dir)
    write_csv(rows, Path(args.output_csv))
    print_console_summary(rows)
    print('-' * 100)
    print(f'CSV saved to: {args.output_csv}')


if __name__ == '__main__':
    main()
