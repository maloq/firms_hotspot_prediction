# Revision Experiments Complete

This directory has been reorganized into a clean experiment library.

- `experiments/`: reader-facing experiment folders with descriptions, analysis, narrow CSV tables, PNG plots, PDF plots, and per-experiment artifacts.
- `shared_artifacts/`: reusable raw sources, configs, logs, models, predictions, target caches, neural data, and original mixed plot files.

No `.tex` files are kept in this result tree. All remaining CSV files are capped at six columns; wide raw tables were archived as JSONL.GZ plus schema JSON files under `shared_artifacts/`.

Start with [`experiments/index.md`](experiments/index.md).
