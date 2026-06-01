from __future__ import annotations

import numpy as np
import pandas as pd

from src.revision_evaluation.neural_full_grid import (
    dense_neural_feature_columns,
    make_dense_neural_raw_predict_fn,
    neural_prediction_path,
)


class DummyPredictor:
    dynamic_columns = ["t2m_lag_7", "d2m_lag_7"]
    dynamic_mode = "summary"
    static_columns = ["slt", "z"]
    categorical_columns = ["ecoregion_name", "slt"]

    def __init__(self) -> None:
        self.batch_lengths: list[int] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.batch_lengths.append(len(frame))
        return np.linspace(0.1, 0.9, len(frame), dtype=np.float32)


def test_dense_neural_feature_columns_keep_coordinates_and_dedupe() -> None:
    assert dense_neural_feature_columns(DummyPredictor()) == [
        "datetime",
        "lat_rounded",
        "lon_rounded",
        "t2m_lag_7",
        "d2m_lag_7",
        "slt",
        "z",
        "ecoregion_name",
    ]


def test_dense_neural_predict_fn_chunks_rows() -> None:
    predictor = DummyPredictor()
    frame = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=5)})

    result = make_dense_neural_raw_predict_fn(predictor, rows_per_prediction_batch=2)(frame)

    assert predictor.batch_lengths == [2, 2, 1]
    assert result["raw_score_source"] == "dense_neural_predictor_logit"
    assert result["prob_raw"].shape == (5,)
    assert result["raw_score"].shape == (5,)


def test_dense_neural_feature_columns_skip_dynamic_columns_for_daily_spatial() -> None:
    predictor = DummyPredictor()
    predictor.dynamic_mode = "daily_spatial"

    assert dense_neural_feature_columns(predictor) == [
        "datetime",
        "lat_rounded",
        "lon_rounded",
        "slt",
        "z",
        "ecoregion_name",
    ]


def test_neural_prediction_path_for_spatial_no_tp() -> None:
    assert str(neural_prediction_path("spatial_tsn_no_tp")) == (
        "outputs/nn_global_full_spatial_tsn_no_tp/legacy_sampled_predictions/test_predictions.parquet"
    )


def test_neural_prediction_path_for_fullgrid_optimized_mlp() -> None:
    assert str(neural_prediction_path("minimal_mlp_fullgrid_opt")) == (
        "outputs/nn_global_full_minimal_mlp_fullgrid_opt/legacy_sampled_predictions/test_predictions.parquet"
    )


def test_neural_prediction_path_for_fullgrid_rank_optimized_mlp() -> None:
    assert str(neural_prediction_path("minimal_mlp_fullgrid_rank_opt")) == (
        "outputs/nn_global_full_minimal_mlp_fullgrid_rank_opt/legacy_sampled_predictions/test_predictions.parquet"
    )
