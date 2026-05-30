# Legacy Experiment Area

This directory is reserved for exploratory scripts and outputs that are not part of the official paper pipeline.

## Official Paper Code

The paper pipeline is fixed to the v6 single residual model:

- `scripts/paper_extract.py`
- `scripts/paper_train_loso.py`
- `scripts/paper_evaluate.py`
- `scripts/paper_make_tables.py`

## Archived Source Scripts

The previous top-level experiment scripts have been moved to:

```text
experiments/legacy/src_scripts/
```

This includes:

- `src/train_squat_corrector.py`
- `src/train_squat_corrector_v3_advanced.py`
- `src/train_squat_corrector_v4_rich.py`
- `src/train_squat_corrector_v5_paper.py`
- `src/train_squat_corrector_v6_cnn.py`
- `src/train_squat_corrector_v7_hybrid_transformer.py`
- `src/train_universal_corrector_v7.py`
- `src/extract_universal_features.py`
- `src/extract_universal_features_final.py`
- `src/extract_v15_both_legs.py`
- `src/evaluate_*.py`
- `src/plot_*.py`
- `src/run_world_baselines.py`

## Output Archives

The old output folders remain under `outputs/` because that directory is ignored and contains large/generated artifacts. Treat these as audit material, not paper reproduction outputs:

- non-v6 folders under `outputs/ml_correction/`

Keep legacy files available for audit, but do not cite them as paper reproduction code.
