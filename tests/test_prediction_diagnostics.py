from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.revision_evaluation.prediction_diagnostics import (
    PredictionDiagnosticConfig,
    _bilinear_interpolate_grid,
    _prediction_day_sample,
    run_prediction_diagnostics,
)


def test_prediction_diagnostics_writes_timeseries_and_error_maps(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    prediction_dir = results_dir / "shared_artifacts" / "predictions"
    prediction_dir.mkdir(parents=True)
    prediction_path = prediction_dir / "toy_model_test_predictions.parquet"
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2023-01-01", "2023-01-02"]
            ),
            "lat_rounded": [10.0, 10.0, 10.1, 10.1],
            "lon_rounded": [20.0, 20.1, 20.0, 20.1],
            "is_fire": [1, 0, 0, 1],
            "pred_proba": [0.8, 0.2, 0.4, 0.6],
            "eval_weight": [1.0, 1.0, 1.0, 1.0],
            "model_name": ["toy_model"] * 4,
        }
    ).to_parquet(prediction_path, index=False)

    regions_file = tmp_path / "regions.yaml"
    regions_file.write_text(
        "\n".join(
            [
                "regions:",
                "  - name: toy_region",
                "    lat_min: 9.9",
                "    lat_max: 10.2",
                "    lon_min: 19.9",
                "    lon_max: 20.2",
            ]
        ),
        encoding="utf-8",
    )

    manifest = run_prediction_diagnostics(
        PredictionDiagnosticConfig(
            results_dir=results_dir,
            regions_file=regions_file,
            source="legacy",
            model="toy_model",
            regions=["toy_region"],
            include_global=False,
            formats=["png"],
            country_shapes=tmp_path / "missing_shapes",
            grid_resolution=0.1,
        )
    )

    output_dir = Path(str(manifest["output_dir"]))
    timeseries = pd.read_csv(output_dir / "tables" / "timeseries_counts.csv")
    spatial_summary = pd.read_csv(output_dir / "tables" / "spatial_error_summary.csv")

    assert set(timeseries["period_start"]) == {"2022-01-01", "2023-01-01"}
    assert "2022-2023 mean" in set(spatial_summary["period"])
    assert {"2022", "2023"}.issubset(set(spatial_summary["period"].astype(str)))
    assert {
        "mean_predicted_risk",
        "mean_smoothed_observed_risk",
        "mean_risk_error",
        "mean_abs_risk_error",
    }.issubset(spatial_summary.columns)
    assert (output_dir / "plots" / "png" / "timeseries_toy_model_toy_region.png").exists()
    assert (output_dir / "plots" / "png" / "error_map_toy_model_toy_region_mean_test_years.png").exists()


def test_prediction_day_sample_samples_each_month_independently() -> None:
    rows = pd.DataFrame(
        {
            "datetime": pd.date_range("2022-01-01", "2022-02-28", freq="D"),
            "eval_weight": 1.0,
        }
    )

    sampled, metadata = _prediction_day_sample(rows, days_per_month=2)

    assert len(sampled) == 4
    assert metadata["sampled_days"] == 4
    assert [item["sampled_days"] for item in metadata["months"]] == [2, 2]
    by_month = sampled.groupby(sampled["datetime"].dt.month)["eval_weight"].first().to_dict()
    assert by_month == {1: 15.5, 2: 14.0}


def test_bilinear_interpolate_grid_to_requested_resolution() -> None:
    lon = np.array([20.0, 20.2])
    lat = np.array([10.0, 10.2])
    values = np.array([[0.0, 2.0], [2.0, 4.0]])

    interp_lon, interp_lat, interp_values, resolution = _bilinear_interpolate_grid(
        lon,
        lat,
        values,
        target_resolution=0.1,
    )

    assert resolution == 0.1
    assert np.allclose(interp_lon, [20.0, 20.1, 20.2])
    assert np.allclose(interp_lat, [10.0, 10.1, 10.2])
    assert np.isclose(interp_values[1, 1], 2.0)
