# Probability Period Overlays

Selected top test-set periods for each region from revision evaluation predictions.

Prediction source: `legacy`.

Feature source: `ERA5`.

Map surface: `dense-neural`.

Dense grid resolution: `0.1` degrees.

Prior correction: `enabled` (train prior `0.15`, deployment prior `0.001`).

| Region | Rank | Model | Period | Days | Selection | Fire locations | AP | Brier | Probability column |
|---|---:|---|---:|---:|---|---:|---:|---:|---|
| Global | 1 | Spatial climate TSN-MLP (global full) | 2022-04-18 to 2022-04-20 | 3 | average_precision | 120 | 0.951 | 0.103 | prob_raw |
| Eastern Siberia | 1 | Spatial climate TSN-MLP (global full) | 2022-03-10 to 2022-03-12 | 3 | average_precision | 9 | 1.000 | 0.042 | prob_raw |
| Far East | 1 | Spatial climate TSN-MLP (global full) | 2024-03-13 to 2024-03-15 | 3 | average_precision | 8 | 1.000 | 0.066 | prob_raw |
| Central Asia | 1 | Spatial climate TSN-MLP (global full) | 2023-01-11 to 2023-01-13 | 3 | average_precision | 8 | 1.000 | 0.050 | prob_raw |
| Europe | 1 | Spatial climate TSN-MLP (global full) | 2023-01-11 to 2023-01-13 | 3 | average_precision | 8 | 1.000 | 0.051 | prob_raw |
