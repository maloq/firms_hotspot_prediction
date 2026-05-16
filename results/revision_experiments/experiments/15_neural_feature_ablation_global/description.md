# Neural Feature Ablation Global

## Purpose
Ablates dynamic, static, and categorical input branches for the configured best neural architecture on the global combined test set.

## Source Tables
- `neural_feature_ablation.csv`

## Notes
- The ablation keeps the neural architecture fixed and zeroes one or more input branches in the prepared NN tensors.
- Delta columns are full neural feature set minus variant, so positive values mean the ablated variant is lower.
