# Failed Or Skipped Experiments

## Neural embedding/fusion ablations

- Failure reason: No prepared NN `.npz` dataset was present in the expected metadata directory.
- Affects main paper claims: No; CatBoost/data-fusion ablations cover the primary reviewer request.
- Suggested next action: Run `make_nn_train_data.py` for the reviewer split, then train/evaluate the listed NN variants.

## No morphological expansion / no dilation

- Failure reason: Current feature parquet was already built from expanded targets; no no-dilation target cache was found.
- Affects main paper claims: Potentially relevant to target-construction sensitivity; reported as a limitation.
- Suggested next action: Add a target_config flag to bypass `expand_positive_points`, rebuild target caches, and rerun CatBoost on the rebuilt feature matrix.

## Stricter MODIS confidence/brightness thresholds

- Failure reason: Requires target and feature regeneration; no stricter-threshold cache was present.
- Affects main paper claims: Limited; main performance and feature ablation claims are still based on the stated target config.
- Suggested next action: Create a stricter target_config and rebuild per-country feature parquet files.

## Lead-time sensitivity

- Failure reason: No lead-time-specific features or forecast-initialization metadata are present in the saved training matrix.
- Affects main paper claims: No direct effect on retrospective ignition discrimination; affects operational lead-time claims.
- Suggested next action: Generate feature matrices by forecast lead time from forecast/hindcast inputs.

## ERA5 -> ERA5

- Failure reason: Raw ERA5 files are readable at /home/ids/vmorozov/era5, but no precomputed ERA5-derived feature parquet with the same schema/date/grid rows as the ECMWF feature table was found. Building it requires a full climate-feature regeneration pass, not a lightweight in-run adapter.
- Affects main paper claims: No for model/ablation claims; yes for the requested weather-source comparison.
- Suggested next action: Generate an ERA5-derived feature parquet on the same target rows and common feature subset, then rerun this runner with an ERA5 features path.

## ERA5 -> SEAS5/ECMWF

- Failure reason: Raw ERA5 files are readable at /home/ids/vmorozov/era5, but no precomputed ERA5-derived feature parquet with the same schema/date/grid rows as the ECMWF feature table was found. Building it requires a full climate-feature regeneration pass, not a lightweight in-run adapter.
- Affects main paper claims: No for model/ablation claims; yes for the requested weather-source comparison.
- Suggested next action: Generate an ERA5-derived feature parquet on the same target rows and common feature subset, then rerun this runner with an ERA5 features path.

## ERA5 + SEAS5/ECMWF -> SEAS5/ECMWF

- Failure reason: Raw ERA5 files are readable at /home/ids/vmorozov/era5, but no precomputed ERA5-derived feature parquet with the same schema/date/grid rows as the ECMWF feature table was found. Building it requires a full climate-feature regeneration pass, not a lightweight in-run adapter.
- Affects main paper claims: No for model/ablation claims; yes for the requested weather-source comparison.
- Suggested next action: Generate an ERA5-derived feature parquet on the same target rows and common feature subset, then rerun this runner with an ERA5 features path.
