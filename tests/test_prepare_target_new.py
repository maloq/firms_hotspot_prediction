import numpy as np
import pandas as pd

from src.target_generation.prepare_target_new import (
    _add_recent_fire_history_features,
    _allocate_by_weights,
    _assign_soft_fire_labels,
    _desired_negative_count,
    _initial_positive_counts,
    _normalise_stratum_weights,
    _sample_stratified_negatives_from_cells,
)


def test_initial_positive_counts_keep_northwest_triple_weight():
    latitude = np.array([61.0, 56.0, 54.0])
    longitude = np.array([44.0, 50.0, 50.0])

    counts = _initial_positive_counts(latitude, longitude)

    np.testing.assert_array_equal(counts, np.array([3, 2, 1]))


def test_stratified_negative_count_targets_positive_fraction():
    assert _desired_negative_count(10, 0.10) == 90
    assert _desired_negative_count(3, 0.25) == 9

    weights = _normalise_stratum_weights(
        {
            "near_fire_hard": 0.3,
            "same_season": 0.2,
            "same_ecoregion": 0.2,
            "same_burnable_landcover": 0.2,
            "random_background": 0.1,
        }
    )
    allocation = _allocate_by_weights(10, weights)

    assert sum(allocation.values()) == 10
    assert allocation["near_fire_hard"] == 3
    assert allocation["random_background"] == 1


def test_near_fire_stratified_sampler_keeps_all_positives_out_of_negatives():
    positives = pd.DataFrame(
        {
            "lat_rounded": [10.0],
            "lon_rounded": [20.0],
            "acq_date": [pd.Timestamp("2020-06-15").date()],
            "month": [6],
            "year": [2020],
            "day": [15],
            "country": ["Testland"],
            "count": [1],
        }
    )
    cells = pd.DataFrame(
        {
            "lat_rounded": [10.0, 10.2, 9.8, 10.0, 10.0],
            "lon_rounded": [20.0, 20.0, 20.0, 20.2, 19.8],
            "country": ["Testland"] * 5,
            "ecoregion_name": ["Boreal"] * 5,
            "ecoregion_realm": ["Palearctic"] * 5,
            "burnable": [True] * 5,
            "landcover_key": ["forest"] * 5,
        }
    )
    settings = {
        "target_positive_fraction": 0.25,
        "stratum_weights": {
            "near_fire_hard": 1.0,
            "same_season": 0.0,
            "same_ecoregion": 0.0,
            "same_burnable_landcover": 0.0,
            "random_background": 0.0,
        },
        "exclude_positive_buffer_cells": 0,
        "exclude_positive_buffer_days": 0,
        "near_fire_min_cells": 2,
        "near_fire_max_cells": 2,
        "near_fire_day_window": 0,
        "max_attempt_multiplier": 200,
    }

    negatives = _sample_stratified_negatives_from_cells(
        positives=positives,
        cells=cells,
        date_start="2020-06-15",
        date_end="2020-06-15",
        settings=settings,
        seed=123,
    )

    assert len(negatives) == 3
    assert negatives["count"].eq(0).all()
    assert negatives["negative_stratum"].eq("near_fire_hard").all()
    assert not (
        negatives["lat_rounded"].eq(10.0)
        & negatives["lon_rounded"].eq(20.0)
        & pd.to_datetime(negatives["acq_date"]).eq(pd.Timestamp("2020-06-15"))
    ).any()


def test_soft_fire_labels_mark_near_fire_negatives_as_uncertain():
    target = pd.DataFrame(
        {
            "count": [1, 0, 0],
            "nearest_positive_distance_cells": [np.nan, 2.0, np.nan],
            "nearest_positive_delta_days": [np.nan, 1.0, np.nan],
        }
    )

    result = _assign_soft_fire_labels(
        target,
        {
            "enabled": True,
            "column": "soft_label",
            "max_negative_label": 0.5,
            "spatial_decay_cells": 2.0,
            "temporal_decay_days": 7.0,
            "max_distance_cells": 5,
            "max_delta_days": 7,
        },
    )

    assert result["soft_label"].iloc[0] == 1.0
    assert 0.0 < result["soft_label"].iloc[1] < 0.5
    assert result["soft_label"].iloc[2] == 0.0


def test_lagged_fire_history_excludes_last_month():
    positives = pd.DataFrame(
        {
            "lat_rounded": [10.0, 10.0, 10.0],
            "lon_rounded": [20.0, 20.0, 20.0],
            "acq_date": [
                pd.Timestamp("2020-06-10").date(),
                pd.Timestamp("2020-07-20").date(),
                pd.Timestamp("2020-08-19").date(),
            ],
            "country": ["Testland"] * 3,
            "count": [1, 1, 1],
        }
    )
    target = pd.DataFrame(
        {
            "lat_rounded": [10.0, 10.1],
            "lon_rounded": [20.0, 20.0],
            "acq_date": [
                pd.Timestamp("2020-08-29").date(),
                pd.Timestamp("2020-08-29").date(),
            ],
            "country": ["Testland"] * 2,
            "count": [0, 0],
        }
    )

    result = _add_recent_fire_history_features(
        target,
        positives=positives,
        settings={
            "enabled": True,
            "radii_cells": [0, 1],
            "min_lag_days": 30,
            "count_windows": [
                {"name": "last_month", "start_days": 30, "end_days": 60},
                {"name": "last_year", "start_days": 30, "end_days": 395},
            ],
            "include_days_since": False,
        },
    )

    assert result["past_fire_count_r0_last_month"].tolist() == [1, 0]
    assert result["past_fire_count_r0_last_year"].tolist() == [2, 0]
    assert result["past_fire_count_r1_last_month"].tolist() == [1, 1]
    assert result["past_fire_count_r1_last_year"].tolist() == [2, 2]
    assert "days_since_fire_r0" not in result.columns
