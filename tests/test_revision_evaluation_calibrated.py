from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from src.revision_evaluation import full_grid_evaluation
from src.revision_evaluation.calibration import apply_calibrator, fit_calibrator
from src.revision_evaluation.deployment_grid import (
    DeploymentGridChunk,
    REQUIRED_DEPLOYMENT_COLUMNS,
    _cross_cells_dates,
    _filter_frame_to_bounds,
    _iter_cell_date_row_blocks,
    _positive_flat_indices,
    _sample_cross_cells_dates,
    coordinate_bounds_for_grid,
    iter_deployment_grid_chunks,
)
from src.revision_evaluation.probability_metrics import (
    expected_observed_count_ratio,
    make_reliability_bins,
    max_weighted_f1,
    weighted_brier_score,
    weighted_log_loss,
)
from src.revision_evaluation.tabular import (
    catboost_categorical_features,
    model_feature_columns,
    validate_no_leakage_features,
)
from src.target_generation.prepare_target_new import normalize_confidence_threshold_for_series


@dataclass
class TinyConfig:
    run_full_grid_evaluation: bool = True
    calibration_method: str = "platt_global"
    deployment_grid_universe: str = "test_grid"
    calibration_start_date: str = "2021-01-01"
    calibration_end_date: str = "2021-01-02"
    test_start_date: str = "2022-01-01"
    test_end_date: str = "2022-01-02"
    n_reliability_bins: int = 4
    reliability_binning: str = "equal_count"
    save_full_grid_predictions: bool = True
    save_calibrated_predictions: bool = True
    cache_full_grid_features: bool = True
    weighted_grid_sample_include_all_positives: bool = True
    deployment_grid_coordinate_bounds: list[float] | None = None
    deployment_grid_clip_to_feature_bounds: bool = True


def test_deployment_grid_chunk_has_required_columns_and_positive_weights():
    cells = pd.DataFrame(
        {
            "lat_rounded": [10.0, 10.1],
            "lon_rounded": [20.0, 20.1],
            "country": ["Test_Land", "Test_Land"],
        }
    )
    labels = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2022-01-01")],
            "lat_rounded": [10.0],
            "lon_rounded": [20.0],
            "country": ["Test_Land"],
            "is_fire": [1],
        }
    )
    rows = _cross_cells_dates(cells, pd.date_range("2022-01-01", "2022-01-02"), labels, "Test_Land")

    assert set(REQUIRED_DEPLOYMENT_COLUMNS).issubset(rows.columns)
    assert "acq_date" in rows.columns
    assert pd.to_datetime(rows["acq_date"]).equals(pd.to_datetime(rows["datetime"]))
    assert np.isfinite(rows["eval_weight"]).all()
    assert (rows["eval_weight"] > 0).all()
    assert rows["is_fire"].sum() == 1


def test_deployment_grid_label_join_uses_resolution_keys_not_exact_floats():
    cells = pd.DataFrame(
        {
            "lat_rounded": np.array([43.20000076293945, 43.29999923706055], dtype=np.float32),
            "lon_rounded": np.array([132.5, 132.60000610351562], dtype=np.float32),
            "country": ["Test_Land", "Test_Land"],
        }
    )
    labels = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2022-01-01")],
            "lat_rounded": [43.2],
            "lon_rounded": [132.5],
            "country": ["Test_Land"],
            "is_fire": [1],
        }
    )
    dates = pd.date_range("2022-01-01", "2022-01-02")

    rows = _cross_cells_dates(cells, dates, labels, "Test_Land", resolution=0.1)
    positive_flat = _positive_flat_indices(cells, dates, labels, "Test_Land", resolution=0.1)

    assert rows["is_fire"].sum() == 1
    assert positive_flat.tolist() == [0]


def test_deployment_grid_builder_does_not_call_random_negative_sampler():
    source = inspect.getsource(iter_deployment_grid_chunks)
    assert "add_negative_samples" not in source
    assert "prepare_target_data" not in source


def test_deployment_grid_clips_to_feature_coordinate_bounds():
    bounds = coordinate_bounds_for_grid(TinyConfig(), {"coordinate_bounds": [35, 6, 75, 179]}, {})
    assert bounds == (35.0, 6.0, 75.0, 179.0)

    cells = pd.DataFrame(
        {
            "lat_rounded": [34.9, 35.0, 50.0, 75.0, 75.1],
            "lon_rounded": [10.0, 5.9, 60.0, 179.0, 60.0],
            "country": ["Test_Land"] * 5,
        }
    )
    clipped = _filter_frame_to_bounds(cells, bounds, resolution=0.1)

    assert clipped[["lat_rounded", "lon_rounded"]].to_numpy().tolist() == [[50.0, 60.0], [75.0, 179.0]]


def test_deployment_grid_row_blocks_respect_max_rows_before_feature_generation():
    cells = pd.DataFrame(
        {
            "lat_rounded": np.arange(5, dtype=float),
            "lon_rounded": np.arange(5, dtype=float),
            "country": ["Test_Land"] * 5,
        }
    )
    blocks = list(
        _iter_cell_date_row_blocks(
            cells,
            pd.date_range("2022-01-01", periods=3),
            pd.DataFrame(),
            "Test_Land",
            max_rows=6,
        )
    )

    assert len(blocks) == 3
    assert all(len(rows) <= 6 for _, rows in blocks)
    assert sum(len(rows) for _, rows in blocks) == 15


def test_weighted_deployment_sample_uses_known_positive_weights():
    cells = pd.DataFrame(
        {
            "lat_rounded": np.arange(5, dtype=float),
            "lon_rounded": np.arange(5, dtype=float),
            "country": ["Test_Land"] * 5,
        }
    )
    config = TinyConfig()
    config.weighted_grid_sample_fraction = 0.4
    rows = _sample_cross_cells_dates(
        cells,
        pd.date_range("2022-01-01", periods=3),
        pd.DataFrame(),
        "Test_Land",
        config=config,
        split_name="test",
        period="2022-01",
    )

    assert len(rows) == 6
    assert np.isclose(rows["eval_weight"].iloc[0], 15 / 6)
    assert rows["eval_weight"].nunique() == 1
    assert set(REQUIRED_DEPLOYMENT_COLUMNS).issubset(rows.columns)


def test_weighted_deployment_sample_includes_all_positive_grid_days():
    cells = pd.DataFrame(
        {
            "lat_rounded": np.arange(5, dtype=float),
            "lon_rounded": np.arange(5, dtype=float),
            "country": ["Test_Land"] * 5,
        }
    )
    labels = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-03")],
            "lat_rounded": [0.0, 4.0],
            "lon_rounded": [0.0, 4.0],
            "country": ["Test_Land", "Test_Land"],
            "is_fire": [1, 1],
        }
    )
    config = TinyConfig()
    config.weighted_grid_sample_fraction = 0.2
    rows = _sample_cross_cells_dates(
        cells,
        pd.date_range("2022-01-01", periods=3),
        labels,
        "Test_Land",
        config=config,
        split_name="test",
        period="2022-01",
    )

    positives = rows.loc[rows["is_fire"].eq(1), ["datetime", "lat_rounded", "lon_rounded", "eval_weight"]]
    assert len(positives) == 2
    assert positives["eval_weight"].eq(1.0).all()
    assert np.isclose(rows["eval_weight"].sum(), 15.0)


def test_feature_cache_rejects_missing_positive_rows():
    chunk = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-01")],
            "lat_rounded": [10.0, 10.1],
            "lon_rounded": [20.0, 20.1],
            "country": ["Test_Land", "Test_Land"],
            "is_fire": [1, 0],
            "eval_weight": [1.0, 10.0],
        }
    )
    stale_cache = chunk.loc[chunk["is_fire"].eq(0)].copy()

    assert not full_grid_evaluation._feature_cache_matches_chunk(stale_cache, chunk)


def test_feature_cache_allows_landsea_dropped_positive_subset():
    chunk = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2022-01-01")] * 4,
            "lat_rounded": [10.0, 10.1, 10.2, 10.3],
            "lon_rounded": [20.0, 20.1, 20.2, 20.3],
            "country": ["Test_Land"] * 4,
            "is_fire": [1, 1, 1, 0],
            "eval_weight": [1.0, 1.0, 1.0, 10.0],
        }
    )
    cache_after_mask = chunk.iloc[[0, 1, 3]].copy()

    assert full_grid_evaluation._feature_cache_matches_chunk(cache_after_mask, chunk)


def test_catboost_categorical_features_infers_unconfigured_string_columns():
    frame = pd.DataFrame(
        {
            "country": ["Kazakhstan", "Russian_Federation"],
            "ecoregion_name": ["Steppe", "Taiga"],
            "slt": [1.0, 2.0],
            "t2m_mean_30d": [275.0, 280.0],
        }
    )

    cats = catboost_categorical_features(
        frame,
        ["country", "ecoregion_name", "slt", "t2m_mean_30d"],
        {
            "cat_features": ["ecoregion_name", "slt"],
            "numerical_cat_features": ["slt"],
        },
    )

    assert cats == ["ecoregion_name", "slt", "country"]


def test_feature_cache_matches_float32_coordinate_cache_keys():
    chunk = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2022-01-01")],
            "lat_rounded": [41.4],
            "lon_rounded": [47.7],
            "country": ["Test_Land"],
            "is_fire": [0],
            "eval_weight": [200.0],
        }
    )
    cache = chunk.copy()
    cache["lat_rounded"] = cache["lat_rounded"].astype("float32")
    cache["lon_rounded"] = cache["lon_rounded"].astype("float32")
    cache["eval_weight"] = np.float32(199.999695)

    assert full_grid_evaluation._feature_cache_matches_chunk(cache, chunk, resolution=0.1)


def test_feature_cache_rejects_old_float_join_cache_with_too_few_positives():
    chunk = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2022-01-01")] * 6,
            "lat_rounded": np.arange(6, dtype=float),
            "lon_rounded": np.arange(6, dtype=float),
            "country": ["Test_Land"] * 6,
            "is_fire": [1, 1, 1, 1, 0, 0],
            "eval_weight": [1.0, 1.0, 1.0, 1.0, 10.0, 10.0],
        }
    )
    stale_cache = chunk.iloc[[0, 4, 5]].copy()

    assert not full_grid_evaluation._feature_cache_matches_chunk(stale_cache, chunk)


def test_forbidden_leakage_columns_are_removed_and_checked():
    df = pd.DataFrame(
        columns=[
            "datetime",
            "lat_rounded",
            "lon_rounded",
            "count",
            "brightness",
            "confidence",
            "feature_x",
            "ecoregion_name",
            "historical_fire_count",
        ]
    )
    features = model_feature_columns(
        df,
        ignored_features=["datetime"],
        use_lat_lon_features=False,
        use_ecoregion_features=False,
        use_historical_fire_features=False,
    )
    assert features == ["feature_x"]
    validate_no_leakage_features(features)


def test_probability_metrics_weighted_count_ratio():
    y = np.array([1, 0, 1])
    p = np.array([0.8, 0.1, 0.2])
    w = np.array([2.0, 1.0, 1.0])

    assert weighted_brier_score(y, p, w) > 0
    assert weighted_log_loss(y, p, w) > 0
    f1 = max_weighted_f1(y, p, w)
    assert np.isclose(f1["max_f1"], 1.0)
    assert np.isclose(f1["recall_at_max_f1"], 1.0)
    counts = expected_observed_count_ratio(y, p, w)
    assert counts["observed_fire_positive_grid_cells"] == 3.0
    assert np.isclose(counts["expected_fire_positive_grid_cells"], 1.9)


def test_reliability_equal_count_bins_are_populated():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.70, 0.80, 0.90, 0.95])
    w = np.ones_like(p)

    bins = make_reliability_bins(y, p, w, n_bins=4, strategy="equal_count")

    assert bins["bin"].tolist() == [0, 1, 2, 3]
    assert bins["n_unweighted"].tolist() == [2, 2, 2, 2]
    assert np.isclose(bins["n_weighted"].sum(), 8.0)


def test_calibrator_fits_and_outputs_probabilities():
    frame = pd.DataFrame(
        {
            "raw_score": [-3, -2, -1, 1, 2, 3],
            "prob_raw": [0.05, 0.1, 0.2, 0.7, 0.85, 0.95],
            "is_fire": [0, 0, 0, 1, 1, 1],
            "eval_weight": [1, 1, 1, 1, 1, 1],
            "month": [1, 1, 2, 2, 3, 3],
            "country": ["A", "A", "A", "A", "A", "A"],
        }
    )
    calibrator = fit_calibrator(frame, "platt_month")
    prob = apply_calibrator(calibrator, frame)
    assert np.all((prob >= 0) & (prob <= 1))


def test_prior_offset_calibrator_matches_calibration_count():
    frame = pd.DataFrame(
        {
            "raw_score": [-6, -5, -4, -3, -2, -1],
            "prob_raw": [0.002, 0.006, 0.018, 0.047, 0.119, 0.269],
            "is_fire": [0, 0, 0, 0, 0, 1],
            "eval_weight": [10, 10, 10, 10, 10, 1],
            "month": [1, 1, 1, 1, 1, 1],
            "country": ["A"] * 6,
        }
    )
    calibrator = fit_calibrator(frame, "prior_offset_month")
    prob = apply_calibrator(calibrator, frame)

    assert np.all((prob >= 0) & (prob <= 1))
    assert np.isclose(np.sum(prob * frame["eval_weight"]), np.sum(frame["is_fire"] * frame["eval_weight"]))


def test_full_grid_calibrated_smoke(monkeypatch, tmp_path: Path):
    def fake_chunks(*, split_name, **kwargs):
        if split_name == "calibration":
            rows = pd.DataFrame(
                {
                    "datetime": pd.date_range("2021-01-01", periods=6),
                    "lat_rounded": np.arange(6, dtype=float),
                    "lon_rounded": np.arange(6, dtype=float),
                    "country": ["A"] * 6,
                    "month": [1] * 6,
                    "year": [2021] * 6,
                    "is_fire": [0, 0, 0, 1, 1, 1],
                    "eval_weight": [1.0] * 6,
                    "x": [-3, -2, -1, 1, 2, 3],
                }
            )
        else:
            rows = pd.DataFrame(
                {
                    "datetime": pd.date_range("2022-01-01", periods=6),
                    "lat_rounded": np.arange(6, dtype=float),
                    "lon_rounded": np.arange(6, dtype=float),
                    "country": ["A"] * 6,
                    "month": [1] * 6,
                    "year": [2022] * 6,
                    "is_fire": [0, 0, 1, 0, 1, 1],
                    "eval_weight": [1.0] * 6,
                    "x": [-3, -2, -1, 1, 2, 3],
                }
            )
        yield DeploymentGridChunk(split_name=split_name, country="A", period="tiny", rows=rows)

    monkeypatch.setattr(full_grid_evaluation, "iter_deployment_grid_chunks", fake_chunks)

    def predict_raw(X):
        raw = X["x"].to_numpy(dtype=float)
        return {
            "raw_score": raw,
            "prob_raw": 1.0 / (1.0 + np.exp(-raw)),
            "raw_score_source": "catboost_raw_formula_val",
        }

    stale_failure = tmp_path / "primary_full_grid_calibrated" / "failures" / "Tiny_CatBoost.json"
    stale_failure.parent.mkdir(parents=True)
    stale_failure.write_text("{}")

    metrics = full_grid_evaluation.evaluate_model_full_grid_calibrated(
        model_name="Tiny CatBoost",
        model_type="CatBoost",
        feature_columns=["x"],
        config=TinyConfig(),
        output_dir=tmp_path,
        predict_raw_fn=predict_raw,
        feature_config={},
        target_config={},
        feature_set="tiny",
    )

    assert metrics["evaluation_type"] == "primary_full_grid_calibrated"
    prediction_path = tmp_path / "primary_full_grid_calibrated" / "predictions" / "Tiny_CatBoost_test_predictions.parquet"
    pred = pd.read_parquet(prediction_path)
    assert "raw_score" in pred.columns
    assert pred["raw_score_source"].eq("catboost_raw_formula_val").all()
    assert pred["prob_calibrated"].between(0, 1).all()
    assert not (tmp_path / "main_model_comparison.csv").exists()
    primary_dir = tmp_path / "primary_full_grid_calibrated"
    for name in [
        "prevalence_audit.csv",
        "risk_concentration.csv",
        "count_correction.csv",
        "spatial_scale_evaluation.csv",
    ]:
        assert (primary_dir / name).exists()
    correction = pd.read_csv(primary_dir / "count_correction.csv")
    assert {"raw_expected_observed_count_ratio", "calibrated_expected_observed_count_ratio"}.issubset(correction.columns)
    risk = pd.read_csv(primary_dir / "risk_concentration.csv")
    assert {"recall_at_top_q", "lift_at_q", "ap_lift"}.issubset(risk.columns)
    assert (tmp_path / "primary_full_grid_calibrated" / "feature_cache" / "calibration_A_tiny.parquet").exists()
    assert not stale_failure.exists()


def test_confidence_threshold_scale_normalization():
    threshold, scale = normalize_confidence_threshold_for_series(pd.Series([10, 50, 95]), 0.85)
    assert scale == "percent_0_100"
    assert threshold == 85.0
