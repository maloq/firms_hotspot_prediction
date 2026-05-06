# Failures And Remaining Blockers

## Full ERA5 vs SEAS5 CatBoost source matrix

- Reason: Processed ERA5 zarr coverage is 2009-2024 and spatially narrower than the 2001-2025 SEAS5 feature matrix; direct raw 2025 GRIB inspection did not complete fast enough for safe full parity reconstruction in this run.
- Attempted fixes: Verified raw GRIB tree, verified processed ERA5 zarr schema, patched loader for ERA5 `time` coordinate, wrote common schema files.
- Affects paper claims: Affects ERA5/SEAS5 source-comparison claims; does not affect main SEAS5 operational model/ablation claims.
- Next action: Batch-convert raw ERA5 2000-2025 GRIBs to the repo zarr schema over the full domain, then rerun source comparison.
