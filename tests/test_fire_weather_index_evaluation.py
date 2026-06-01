from pathlib import Path

import numpy as np
import pandas as pd

from src.revision_evaluation import fire_weather_index_evaluation as fwi


class TinyFWIConfig:
    deployment_grid_resolution = 0.1
    deployment_grid_universe = "land_or_burnable"
    random_error_trials = 1
    random_error_sample_size = 0


def test_fwi_inventory_available_dates_respects_requested_window():
    inventory = fwi.FWIInventory(
        pd.DataFrame(
            {
                "path": ["a.grib", "a.grib", "b.grib", "b.grib"],
                "file": ["a.grib", "a.grib", "b.grib", "b.grib"],
                "variable": ["fwinx", "fwinx", "fwinx", "fdsrte"],
                "date": pd.to_datetime(["2023-03-01", "2023-03-02", "2023-11-01", "2023-03-01"]),
                "year": [2023, 2023, 2023, 2023],
                "month": [3, 3, 11, 3],
                "grid_type": ["reduced_gg", "reduced_gg", "reduced_gg", "reduced_gg"],
            }
        )
    )

    dates = inventory.available_dates(["fwinx"], start_date="2023-03-02", end_date="2023-10-31")

    assert list(dates.strftime("%Y-%m-%d")) == ["2023-03-02"]
    summary = inventory.availability_summary(["fwinx"])
    assert summary.loc[0, "available_months"] == "2023-03,2023-11"
    assert inventory.variables_available_in_year(["fwinx", "fdsrte"], 2023) == ["fdsrte", "fwinx"]
    assert inventory.complete_case_dates(["fwinx", "fdsrte"]).strftime("%Y-%m-%d").tolist() == ["2023-03-01"]


def test_write_variable_metric_tables_for_ranking_scores(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2023-03-01",
                    "2023-03-01",
                    "2023-03-02",
                    "2023-03-02",
                    "2024-03-01",
                    "2024-03-01",
                ]
            ),
            "lat_rounded": [50.0, 50.1, 50.0, 50.1, 50.0, 50.1],
            "lon_rounded": [100.0, 100.1, 100.0, 100.1, 100.0, 100.1],
            "country": ["A"] * 6,
            "month": [3, 3, 3, 3, 3, 3],
            "year": [2023, 2023, 2023, 2023, 2024, 2024],
            "is_fire": [0, 1, 0, 1, 0, 1],
            "eval_weight": [1.0] * 6,
            "raw_score": [1.0, 9.0, 2.0, 8.0, 3.0, 7.0],
        }
    )

    metrics = fwi.write_variable_metric_tables(
        frame,
        variable="fwinx",
        output_dir=tmp_path,
        regions=[],
        config=TinyFWIConfig(),
    )

    assert metrics is not None
    global_row = metrics["global"]
    assert global_row["evaluation_type"] == fwi.EVALUATION_TYPE
    assert global_row["average_precision"] == 1.0
    assert global_row["raw_score_min"] == 1.0
    assert global_row["raw_score_max"] == 9.0
    assert set(metrics["by_year"]["period"]) == {"2023", "2024"}
    assert {"recall_at_top_q", "lift_at_q", "ap_lift"}.issubset(metrics["risk"].columns)
    assert "weighted_score_sum" in metrics["spatial"].columns
    assert np.isfinite(metrics["spatial"]["average_precision"].dropna()).any()


def test_fwi_logistic_regression_trains_on_2022_and_scores_other_years(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2022-03-01",
                    "2022-03-01",
                    "2022-03-02",
                    "2022-03-02",
                    "2023-03-01",
                    "2023-03-01",
                    "2024-03-01",
                    "2024-03-01",
                ]
            ),
            "lat_rounded": [50.0, 50.1, 50.0, 50.1, 50.0, 50.1, 50.0, 50.1],
            "lon_rounded": [100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1],
            "country": ["A"] * 8,
            "month": [3] * 8,
            "year": [2022, 2022, 2022, 2022, 2023, 2023, 2024, 2024],
            "is_fire": [0, 1, 0, 1, 0, 1, 0, 1],
            "eval_weight": [1.0] * 8,
            "fwinx": [1.0, 9.0, 2.0, 8.0, 1.5, 7.5, 2.5, 8.5],
            "fdsrte": [0.1, 0.9, 0.2, 0.8, 0.2, 0.7, 0.3, 0.9],
        }
    )

    metadata = fwi.write_fwi_logistic_regression_tables(
        frame,
        feature_columns=["fwinx", "fdsrte"],
        train_year=2022,
        output_dir=tmp_path,
        regions=[],
        config=TinyFWIConfig(),
        append_to_main=True,
    )

    assert metadata["train_year"] == 2022
    assert metadata["test_years"] == [2023, 2024]
    assert metadata["feature_columns"] == ["fwinx", "fdsrte"]
    comparison = pd.read_csv(tmp_path / "logistic_regression_model_comparison.csv")
    global_row = comparison[comparison["region"].eq("global")].iloc[0]
    assert global_row["model_name"] == "FWI Logistic Regression (train 2022)"
    assert global_row["available_days"] == 2
    assert global_row["average_precision"] > 0.5
    assert set(pd.read_csv(tmp_path / "logistic_regression_model_comparison_by_year.csv")["period"]) == {
        2023,
        2024,
    }
    assert (tmp_path / "logistic_regression_coefficients.csv").exists()
    assert "FWI Logistic Regression (train 2022)" in set(pd.read_csv(tmp_path / "model_comparison.csv")["model_name"])
