# Failures And Remaining Blockers

No failed ERA5 input-source experiment remains after the available-domain rerun.

## ERA5 Available-Domain Source Comparison

- Status: completed for SEAS5/ECMWF -> SEAS5/ECMWF, ERA5 -> ERA5, ERA5 -> SEAS5/ECMWF, ERA5 + SEAS5/ECMWF -> SEAS5/ECMWF, and configured best-neural source rows.
- Limitation: metrics are legacy sampled/case-control diagnostics on rows covered by processed ERA5 zarr data and common years; they are not primary calibrated deployment-grid probability metrics.
- Coverage: processed ERA5 zarrs currently support the available-domain run through 2024. Exact full-domain/full-2025 parity remains a data-coverage limitation, not a model-code failure.
