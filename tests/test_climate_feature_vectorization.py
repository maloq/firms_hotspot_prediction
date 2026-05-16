import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from src.feature_generation.prepare_climate_data import (
    _build_feature_matrix_vectorized,
    check_fragmented_dataset_bounds,
    check_dataset_bounds,
    extract_climate_timeseries,
    extract_climate_timeseries_fragmented,
)
from src.feature_generation.load_climate_data import ClimateFragment
from src.feature_generation.prepare_climate_features import (
    _extract_single_series,
    get_feature_configs_and_names,
)


def test_vectorized_climate_features_match_single_series_extractor():
    matrix = np.array(
        [
            np.arange(1, 11, dtype=np.float32),
            np.array([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10], dtype=np.float32),
            np.full(10, np.nan, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    params, names = get_feature_configs_and_names(
        sample_ts_array=matrix[0],
        variable_name="t2m",
        lags_global=[1, 3],
        windows_global=[3, 5],
        spans_global=[3],
        features_to_include={
            "t2m": ["rolling", "rolling_ext", "ewm", "diff", "pct_change", "trend"],
        },
        trend_window_global=[3, 5],
        max_length_global=10,
    )

    fast = _build_feature_matrix_vectorized(matrix, "t2m", params, names)
    assert fast is not None

    slow = np.array(
        [
            [np.float32(_extract_single_series(row, "t2m", **params)[name]) for name in names]
            for row in matrix
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(fast, slow, equal_nan=True)


def test_batched_climate_timeseries_extraction_preserves_rows_and_windows():
    times = pd.date_range("2020-01-01", periods=8, freq="D")
    lats = np.array([10.0, 11.0], dtype=np.float32)
    lons = np.array([20.0, 21.0], dtype=np.float32)
    values = np.empty((len(times), len(lats), len(lons)), dtype=np.float32)

    for t_idx in range(len(times)):
        for lat_idx in range(len(lats)):
            for lon_idx in range(len(lons)):
                values[t_idx, lat_idx, lon_idx] = t_idx * 100 + lat_idx * 10 + lon_idx

    ds = xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"), values),
        },
        coords={
            "valid_time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    target = pl.DataFrame(
        {
            "acq_date": [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-05")],
            "lat_rounded": [10.0, 10.0, 11.0],
            "lon_rounded": [20.0, 20.0, 21.0],
        }
    )

    result = extract_climate_timeseries(ds, "t2m", target, n_days=3, location_batch_size=1)

    expected = np.array(
        [
            [200.0, 300.0, 400.0],
            [300.0, 400.0, 500.0],
            [211.0, 311.0, 411.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result, expected)


def test_climate_block_cache_is_reused_across_calls(tmp_path):
    times = pd.date_range("2020-01-01", periods=5, freq="D")
    lats = np.array([10.0, 11.0], dtype=np.float32)
    lons = np.array([20.0, 21.0], dtype=np.float32)
    base_values = np.arange(len(times) * len(lats) * len(lons), dtype=np.float32).reshape(
        len(times), len(lats), len(lons)
    )

    ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), base_values)},
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    target = pl.DataFrame(
        {
            "acq_date": [pd.Timestamp("2020-01-04")],
            "lat_rounded": [10.0],
            "lon_rounded": [20.0],
        }
    )

    first = extract_climate_timeseries(
        ds,
        "t2m",
        target,
        n_days=2,
        block_cache_dir=str(tmp_path),
        block_cache_source_token="unit-test-source",
    )

    changed_ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), base_values + 10000)},
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    second = extract_climate_timeseries(
        changed_ds,
        "t2m",
        target,
        n_days=2,
        block_cache_dir=str(tmp_path),
        block_cache_source_token="unit-test-source",
    )

    np.testing.assert_allclose(second, first)
    assert list((tmp_path / "climate_block_cache" / "t2m").glob("*.npy"))


def test_dataset_bounds_accept_descending_latitude_coordinates():
    times = pd.date_range("2020-01-03", periods=2, freq="D")
    lats = np.array([12.0, 11.0, 10.0], dtype=np.float32)
    lons = np.array([20.0, 21.0, 22.0], dtype=np.float32)
    values = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), values)},
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    target = pl.DataFrame(
        {
            "acq_date": [pd.Timestamp("2020-01-04")],
            "lat_rounded": [11.0],
            "lon_rounded": [21.0],
        }
    )

    result = check_dataset_bounds(ds, target, n_days=2)

    assert result["sufficient"]
    assert result["details"]["latitude"]["sufficient"]


def test_dataset_bounds_reject_missing_future_time_coverage():
    times = pd.date_range("2020-01-01", periods=5, freq="D")
    lats = np.array([10.0, 11.0], dtype=np.float32)
    lons = np.array([20.0, 21.0], dtype=np.float32)
    values = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), values)},
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    target = pl.DataFrame(
        {
            "acq_date": [pd.Timestamp("2020-01-08")],
            "lat_rounded": [10.0],
            "lon_rounded": [20.0],
        }
    )

    result = check_dataset_bounds(ds, target, n_days=2)

    assert not result["sufficient"]
    assert not result["details"]["time"]["sufficient"]


def _write_fragment_dataset(path, lon_values, base_value):
    times = pd.date_range("2020-01-01", periods=6, freq="D")
    lats = np.array([10.0, 11.0], dtype=np.float32)
    lons = np.asarray(lon_values, dtype=np.float32)
    values = np.empty((len(times), len(lats), len(lons)), dtype=np.float32)
    for t_idx in range(len(times)):
        for lat_idx in range(len(lats)):
            for lon_idx in range(len(lons)):
                values[t_idx, lat_idx, lon_idx] = base_value + t_idx * 100 + lat_idx * 10 + lon_idx

    ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), values)},
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path)


def _test_fragment(path, lon_min, lon_max):
    return ClimateFragment(
        variable="t2m",
        files=(str(path),),
        time_start=np.datetime64("2020-01-01", "ns"),
        time_end=np.datetime64("2020-01-06", "ns"),
        lat_min=10.0,
        lat_max=11.0,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_step=1.0,
        lon_step=1.0,
        lat_size=2,
        lon_size=2,
        dtype="float32",
    )


def test_fragmented_extraction_routes_rows_and_leaves_gaps_nan(tmp_path):
    west_path = tmp_path / "west.nc"
    east_path = tmp_path / "east.nc"
    _write_fragment_dataset(west_path, [20.0, 21.0], 0.0)
    _write_fragment_dataset(east_path, [40.0, 41.0], 1000.0)
    fragments = [_test_fragment(west_path, 20.0, 21.0), _test_fragment(east_path, 40.0, 41.0)]

    target = pl.DataFrame(
        {
            "acq_date": [
                pd.Timestamp("2020-01-04"),
                pd.Timestamp("2020-01-05"),
                pd.Timestamp("2020-01-04"),
            ],
            "lat_rounded": [10.0, 11.0, 10.0],
            "lon_rounded": [20.0, 41.0, 30.0],
        }
    )

    coverage = check_fragmented_dataset_bounds(fragments, target, n_days=2)
    assert not coverage["sufficient"]
    assert coverage["details"]["covered_rows"] == 2
    assert coverage["details"]["missing_rows"] == 1

    result = extract_climate_timeseries_fragmented(
        fragments,
        "t2m",
        target,
        n_days=2,
        assignments=coverage["assignments"],
    )

    expected = np.array(
        [
            [200.0, 300.0],
            [1311.0, 1411.0],
            [np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result, expected, equal_nan=True)
