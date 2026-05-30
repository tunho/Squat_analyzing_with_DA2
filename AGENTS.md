# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python research/prototype codebase for squat pose analysis and knee-angle evaluation. Core reusable code lives under `src/squat/`: analyzers in `analyzer/`, frame pipelines in `pipeline/`, geometry helpers in `geometry/`, I/O utilities in `io/`, metrics in `metrics/`, visualizations in `visualization/`, and model wrappers/assets in `models/`. Top-level scripts in `src/` run experiments, training, evaluation, and plotting, for example `src/run_world_baselines.py` and `src/evaluate_world_vs_gt.py`. Draft papers, ID lists, and experiment notes are kept at the repository root; `backup/` contains older reference code.

## Build, Test, and Development Commands

Create an isolated environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run scripts with `src` on `PYTHONPATH` so imports such as `from squat.analyzer import ...` resolve:

```bash
PYTHONPATH=src python src/run_world_baselines.py --video path/to/video.mp4 --max-frames 300
PYTHONPATH=src python src/evaluate_world_vs_gt.py --video path/to/video.mp4 --gt path/to/gt.csv --gt-format table
```

Use `python -m compileall src -q` as a quick syntax/import-path sanity check after edits.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type hints for new public functions, and `pathlib.Path` for filesystem paths. Keep analyzer classes named by method and role, such as `MPWorldRawSquatAnalyzer`, and keep script filenames lowercase with descriptive underscores, such as `plot_algorithm_comparison.py`. Prefer small helper functions in `src/squat/*` over duplicating logic across experiment scripts.

## Testing Guidelines

No formal test suite is currently present. For changes to analyzers, geometry, or pipeline behavior, add focused tests under `tests/` using `pytest` and name files `test_<module>.py`. At minimum, run `python -m compileall src -q` and one representative evaluation command with a short frame range before submitting changes.

## Commit & Pull Request Guidelines

Recent commits use short descriptive summaries in either Korean or English, often imperative or result-focused, for example `Add PM_008 evaluation scripts` or `Refactor: Switch to Surface Point Regression Model`. Keep commits scoped to one experiment or behavior change. Pull requests should include the purpose, commands run, data/video assumptions, output paths, and screenshots or plots when visualization output changes. Note any large generated artifacts separately and avoid committing temporary outputs.

## Security & Configuration Tips

Do not commit private videos, ground-truth datasets, model checkpoints, or large generated result folders unless they are intentionally part of the research artifact. Keep machine-specific paths out of scripts; expose them as CLI arguments instead.
