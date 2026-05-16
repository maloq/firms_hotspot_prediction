# Neural Feature Importance

## Purpose
Measures PR-AUC drops after permuting each input of the best global neural architecture.

## Source Tables
- `neural_feature_importance.csv`

## Notes
- The selected neural architecture is the highest global PR-AUC row in the main model comparison.
- Permutation importance is model reliance on the sampled test tensor, not causal attribution.
