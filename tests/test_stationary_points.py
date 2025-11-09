import os
import sys
from pathlib import Path

import pandas as pd


# Ensure src package is importable when tests are run from repository root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.target_generation.stationary_points import (
    DEFAULT_MIN_MONTHS,
    collect_stationary_points,
    drop_stationary_points,
    save_stationary_points,
)


def create_modis_csv(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def test_collect_stationary_points_basic(tmp_path):
    modis_root = tmp_path / "modis"
    csv_path = modis_root / "2020" / "modis_2020_Test_Land.csv"

    df = pd.DataFrame(
        {
            "latitude": [10.0, 10.0, 10.0, 10.0, 20.0],
            "longitude": [30.0, 30.0, 30.0, 30.0, 40.0],
            "acq_date": [
                "2020-01-01",
                "2020-01-15",
                "2020-02-10",
                "2020-04-20",
                "2020-03-10",
            ],
        }
    )

    create_modis_csv(csv_path, df)

    result = collect_stationary_points(
        modis_dir=modis_root,
        min_months=3,
        countries=["Test_Land"],
        show_progress=False,
    )

    assert "Test_Land" in result
    stationary_df = result["Test_Land"]
    assert len(stationary_df) == 1

    row = stationary_df.iloc[0]
    assert row["latitude"] == 10.0
    assert row["longitude"] == 30.0
    assert row["unique_months"] == 3
    assert row["first_month"] == "2020-01"
    assert row["last_month"] == "2020-04"
    assert row["months"] == "2020-01;2020-02;2020-04"


def test_drop_stationary_points_filters_matches(tmp_path):
    data = pd.DataFrame(
        {
            "country": ["Test_Land", "Test_Land"],
            "latitude": [10.0, 50.0],
            "longitude": [30.0, 60.0],
            "acq_date": ["2020-01-01", "2020-01-02"],
            "brightness": [400, 410],
            "confidence": [95, 96],
        }
    )

    stationary_dir = tmp_path / "stationary"
    stationary_df = pd.DataFrame(
        {
            "country": ["Test_Land"],
            "latitude": [10.0],
            "longitude": [30.0],
            "unique_months": [DEFAULT_MIN_MONTHS],
            "first_month": ["2020-01"],
            "last_month": ["2020-04"],
            "months": ["2020-01;2020-02;2020-03;2020-04"],
        }
    )

    save_stationary_points({"Test_Land": stationary_df}, output_dir=stationary_dir)

    filtered, removed = drop_stationary_points(
        data,
        stationary_dir=stationary_dir,
        country_col="country",
        lat_col="latitude",
        lon_col="longitude",
    )

    assert removed == 1
    assert len(filtered) == 1
    assert filtered.iloc[0]["latitude"] == 50.0
