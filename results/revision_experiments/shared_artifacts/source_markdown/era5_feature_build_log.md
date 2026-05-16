# ERA5 Feature Build Log

- ERA5 zarr root: `/home/ids/vmorozov/data/climate_data/climate_features/ERA5`.
- Common ERA5 years used: `2000-2024`.
- ERA5-covered comparison domain: lat `40-75`, lon `19-180`.
- The source-comparison rerun uses only sampled feature-table rows inside this ERA5-covered domain and common-year window.
- Metrics are legacy sampled/case-control diagnostics for input-source comparison; primary deployment probability metrics remain in the calibrated full-grid tables.
