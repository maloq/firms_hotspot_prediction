from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.revision_evaluation.probability_overlays import (
    DATE_COL,
    LAT_COL,
    LON_COL,
    PROB_COL,
    TARGET_COL,
    WEIGHT_COL,
    Region,
    clear_prediction_frame_cache,
    period_metrics_for_model,
    read_prediction_columns,
)


def test_read_prediction_columns_reuses_cached_frame(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            DATE_COL: pd.date_range("2024-01-01", periods=3),
            LAT_COL: [10.0, 10.1, 10.2],
            LON_COL: [20.0, 20.1, 20.2],
            "is_fire": [0, 1, 0],
            "pred_proba": [0.1, 0.8, 0.2],
        }
    ).to_parquet(path)
    clear_prediction_frame_cache()

    calls: list[object] = []
    original_read_parquet = pd.read_parquet

    def wrapped_read_parquet(*args, **kwargs):
        calls.append(kwargs.get("columns"))
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", wrapped_read_parquet)

    first = read_prediction_columns(path, "auto")
    first["local_mutation"] = "not cached"
    second = read_prediction_columns(path, "auto")

    assert len(calls) == 1
    assert "local_mutation" not in second.columns
    assert first is not second
    assert second[PROB_COL].tolist() == [0.1, 0.8, 0.2]


def test_period_metrics_for_model_scores_sorted_windows(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-01"]
            ),
            LAT_COL: [10.0, 10.0, 10.1, 10.2],
            LON_COL: [20.0, 20.0, 20.1, 20.2],
            TARGET_COL: [1, 1, 0, 0],
            PROB_COL: [0.8, 0.9, 0.2, 0.1],
            WEIGHT_COL: [1.0, 1.0, 2.0, 1.0],
            "model_name": ["demo"] * 4,
            "model_type": ["demo"] * 4,
        }
    )

    metrics = period_metrics_for_model(
        frame,
        prediction_path=tmp_path / "demo.parquet",
        regions=[Region(name="global", display_name="Global")],
        window_days=2,
        require_full_periods=True,
    ).sort_values("period_start")

    assert metrics["period_start"].tolist() == ["2024-01-01", "2024-01-02"]
    assert metrics["support"].tolist() == [3, 2]
    assert metrics["positive_rows"].tolist() == [1, 1]
    np.testing.assert_allclose(metrics["weighted_support"], [4.0, 3.0])
    np.testing.assert_allclose(metrics["expected_fire_positive_grid_cells"], [1.4, 1.2])
    np.testing.assert_allclose(metrics["observed_fire_positive_grid_cells"], [1.0, 1.0])
