# Analysis

- Highest full-grid AP: CatBoost with PR-AUC=0.0014.
- Experiment 1 makes the base-rate denominator explicit before interpreting any threshold metric.
- Experiment 2 contrasts sampled and deployment-grid metrics; sampled PR-AUC/F1 should not be reported as deployment F1.
- Experiment 3 reports whether high-score cells concentrate risk using Recall@top q%, Lift@q, and AP lift.
- Experiment 4 compares raw and calibrated E/O ratios to quantify overprediction correction.
- Experiment 5 tests whether localization improves when exact 0.1 degree cells are relaxed to neighborhoods or coarser grids.
