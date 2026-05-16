# Revision Experiments Complete

This directory has been reorganized into a clean experiment library.

- `experiments/`: reader-facing experiment folders with descriptions, analysis, narrow CSV tables, plots, and per-experiment artifacts when those outputs exist.
- `shared_artifacts/`: reusable raw sources, configs, logs, models, predictions, target caches, neural data, and original mixed plot files.
- `experiments/04_primary_full_grid_calibrated/`: calibrated full-grid prediction study with prevalence, contrast, risk concentration, count correction, and spatial-scale tables when available.
- `experiments/05_legacy_sampled_case_control/`: old undersampled-negative diagnostics retained for backwards comparison.
All remaining CSV files are compact presentation tables; wide raw tables were archived as JSONL.GZ plus schema JSON files under `shared_artifacts/`.

Start with [`experiments/index.md`](experiments/index.md).
