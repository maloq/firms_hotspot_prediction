# Organized Revision Experiments

Each experiment folder contains:

- `description.md`: what the experiment studies.
- `analysis.md`: compact automatic interpretation.
- `tables/`: readable CSV tables capped at six columns.
- `plots/png/`: PNG plots for that experiment.
- `plots/pdf/`: PDF plots for that experiment.
- `artifacts/`: raw JSONL sources, schemas, manifests, and symlinks to reusable outputs.

## Experiments
- [Run Metadata And Repository Audit](00_run_metadata_and_audit/description.md): Collects the repository audit, command log, environment/config references, feature schemas, and run manifest used by the complete revision workflow.
- [Dataset Statistics By Split](01_dataset_statistics_splits/description.md): Summarizes sample counts, class balance, spatial support, labeling rule, and grid resolution for the train/validation/test splits.
- [Dataset Statistics By Region](02_dataset_statistics_by_region/description.md): Separates regional dataset statistics from model metrics so spatial coverage can be read independently.
- [Dataset Statistics By Year](03_dataset_statistics_by_year/description.md): Keeps annual class-balance and sample-count reporting separate from regional and split summaries.
- [Main Model Comparison Global](04_main_model_comparison_global/description.md): Compares the main models on the global combined test set while keeping regional and yearly views in their own folders.
- [Main Model Comparison By Region](05_main_model_comparison_by_region/description.md): Separates the regional model comparison from the global table to make spatial robustness easier to inspect.
- [Main Model Comparison By Year](06_main_model_comparison_by_year/description.md): Reports annual model metrics for 2021-2025 and keeps temporal stability separate from the regional comparison.
- [Feature Ablation Global](07_feature_ablation_global/description.md): Measures how CatBoost performance changes when feature sources are removed or restricted on the global combined test set.
- [Feature Ablation By Region](08_feature_ablation_by_region/description.md): Keeps spatial ablation effects separate from the global ablation table.
- [Feature Ablation By Year](09_feature_ablation_by_year/description.md): Reports annual ablation effects for 2021-2025 apart from the combined and regional-only summaries.
- [Neural Embedding Fusion Global](10_neural_embedding_fusion_global/description.md): Compares temporal-only, static-only, concatenation, one-hot, learned embedding, full fusion, and gated neural variants globally.
- [Neural Embedding Fusion By Region](11_neural_embedding_fusion_by_region/description.md): Separates spatial neural fusion behavior from the global neural comparison.
- [Neural Embedding Fusion By Year](12_neural_embedding_fusion_by_year/description.md): Reports annual neural ablation metrics for 2021-2025 separately from regional and global summaries.
- [Label Sensitivity Global](13_label_sensitivity_global/description.md): Compares the main labels, no dilation, stricter MODIS thresholds, alternative negative ratio, and no historical-fire feature variants globally.
- [Label Sensitivity By Region](14_label_sensitivity_by_region/description.md): Separates regional robustness of target-construction variants from the global label-sensitivity result.
- [Label Sensitivity By Year](15_label_sensitivity_by_year/description.md): Reports annual target-construction sensitivity metrics for 2021-2025 separately from regional and global summaries.
- [Lead-Time Sensitivity Global](16_lead_time_sensitivity_global/description.md): Compares CatBoost performance at 7-day, 14-day, and 30-day horizons on the global combined test set.
- [Lead-Time Sensitivity By Region](17_lead_time_sensitivity_by_region/description.md): Separates regional horizon sensitivity from the global lead-time comparison.
- [Lead-Time Sensitivity By Year](18_lead_time_sensitivity_by_year/description.md): Reports annual horizon sensitivity for 2021-2025 separately from regional and global lead-time summaries.
- [Input Source Comparison Global](19_input_source_comparison_global/description.md): Compares ERA5 and SEAS5/ECMWF source settings, including retrospective upper bound, operational setting, domain shift, and mixed training.
- [Native CatBoost Feature Importance](20_feature_importance_native/description.md): Ranks individual features by CatBoost native importance for the best full model.
- [Grouped Permutation Importance](21_grouped_permutation_importance/description.md): Measures group-level performance drops after permuting feature sources.
- [SHAP Importance](22_shap_importance/description.md): Summarizes TreeSHAP mean absolute contributions for the best CatBoost model where feasible.
- [Failures And Limitations](23_failures_and_limitations/description.md): Contains only true remaining blockers and limitations after attempted rebuilds and reruns.
- [ERA5 vs ECMWF Training Source Comparison](24_era5_ecmwf_train_source_comparison/description.md): Compares ERA5, ECMWF, and duplicated ERA5+ECMWF training rows on the same ECMWF validation/test footprint.
