# Revision Evaluation Package

This package is split by workflow responsibility rather than by reviewer request.

## Entry Points

- `python -m src.revision_evaluation`: runs the suite from `configs/revision_evaluation_all_models_with_nns.yaml`.
- `python -m src.revision_evaluation path/to/config.yaml`: runs the suite from an explicit config.
- `tabular.py`: builds sampled tabular baselines, CatBoost ablations, interpretability tables, and placeholder rows for experiments handled by later stages.
- `sensitivity_experiments.py`: runs label-construction and lead-time sensitivity experiments on sampled feature matrices.
- `neural_training.py` and `neural_metrics.py`: train neural models, run best-architecture input-branch ablations, and import their metrics into cross-model tables.
- `era5_source_comparison.py`: builds the ERA5-available source-comparison matrix and metrics.
- `probability_overlays.py`: creates full-grid prediction overlay plots from suite config.
- `experiment_library.py`: converts raw run outputs into the reader-facing `experiments/*` tree.

## Shared Infrastructure

- `config.py`, `workflow.py`, `stages.py`: declarative suite config and ordered in-process stage execution.
- `full_grid_evaluation.py`, `deployment_grid.py`, `calibration.py`, `probability_metrics.py`: calibrated deployment-grid evaluation helpers.
- `artifacts.py`: small helpers for writing JSON, text, YAML, and pruning empty directories.

## Adding A New Experiment

1. Add runner code to the domain module that owns the data contract.
2. Write raw CSV outputs at the result root or under a clear shared subdirectory.
3. Add the raw table name to `experiment_library.load_all_tables()` when the organizer must archive it.
4. Add one focused `Builder.build_*` section in `experiment_library.py` for the reader-facing tables.
5. Wire the runner into `stages.py` and `workflow.py` only if it is a normal suite stage.

Keep neural, ERA5/source, full-grid probability, and sampled sensitivity work in separate modules.
