# Reviewer Response Insertions

## Reviewer 1: Novelty / Methodological Innovation

We added a reviewer-focused revision experiment package comparing linear, fire-weather-only, weather-only, Random Forest, and fused CatBoost models under a fixed chronological split. These experiments clarify that the proposed contribution is not a single classifier choice, but a data-fusion pipeline that combines meteorological history with ecological, topographic, and anthropogenic context.

## Reviewer 2: Ablation And Feature Importance Requests

We added CatBoost feature-source ablations and grouped permutation importance. The ablations report absolute metrics and drops relative to the full fused model, while grouped permutation measures the PR-AUC/F1 decrease after disrupting each feature group. We explicitly interpret these as model attributions rather than causal effects.

## Reviewer 2: Lead-Time Justification

The current saved training matrix contains 30-day aggregate historical features and does not preserve forecast lead-time metadata. We therefore report lead-time sensitivity as unsupported by the present pipeline and identify forecast-lead-specific feature generation as the next required experiment.

## Reviewer 3: Dataset Statistics Request

We added dataset statistics for global train, validation, and test periods and for each test region, including grid resolution, positive/negative counts, unique grid cells, positive rate, negative sampling status, land/water masking, and target-labeling rules.

## Reviewer 3: Embedding/Fusion Reproducibility Request

The repository contains neural checkpoints but not the prepared NN `.npz` dataset required to reproduce embedding/fusion ablations. We therefore completed the more stable CatBoost data-fusion ablations and report the neural embedding ablation as blocked until the NN input dataset is regenerated.

## Reviewer 3: Morphological Expansion / Grid-Size Concern

We now report that the target configuration uses 0.1-degree spatial coarsening, not a 5 km grid. The no-dilation experiment requires target-cache regeneration because the current feature matrix was already built after positive expansion; this limitation is reported explicitly rather than inferred from the existing labels.
