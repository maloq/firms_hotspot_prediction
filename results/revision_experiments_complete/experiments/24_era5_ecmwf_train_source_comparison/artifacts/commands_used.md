# Commands Used

Working directory: `/home/infres/vmorozov/Misc`

Git commit: `3a48ef85483049608b88aa3e16c77f43ba2c3606`

Environment: `pointnet`, Python `3.12.13`

```bash
conda run -n pointnet python scripts/prepare_missing_era5_2000_2008.py --force
conda run -n pointnet python scripts/run_era5_ecmwf_source_comparison.py --iterations 450 --task-type CPU --training-mode robust --selection-metric validation_f1 --min-tree-count 20
```

The initial F1-early-stopping rerun produced a one-tree ECMWF model. The final robust run uses Logloss early stopping plus a full-iteration candidate per source, then selects by validation F1 only. Thresholds are also selected on validation only.
