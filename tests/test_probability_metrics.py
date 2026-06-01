from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.revision_evaluation.full_grid_evaluation import write_full_grid_model_selection
from src.revision_evaluation.probability_metrics import reliability_summary


def test_reliability_summary_reports_weighted_ece_and_rmse() -> None:
    metrics = reliability_summary(
        [0, 0, 1, 1],
        [0.1, 0.3, 0.7, 0.9],
        sample_weight=[1.0, 1.0, 2.0, 2.0],
        n_bins=2,
        strategy="equal_count",
    )

    assert metrics["reliability_ece"] == pytest.approx(0.2)
    assert metrics["reliability_mce"] == pytest.approx(0.2)
    assert metrics["reliability_rmse"] == pytest.approx(0.2)


def test_full_grid_model_selection_can_target_reliability(tmp_path) -> None:
    primary = tmp_path / "primary_full_grid_calibrated"
    primary.mkdir()
    pd.DataFrame(
        [
            {
                "model_name": "A",
                "model_type": "CatBoost",
                "feature_set": "full",
                "region": "global",
                "split": "test",
                "average_precision": 0.2,
                "reliability_ece": 0.08,
            },
            {
                "model_name": "B",
                "model_type": "Neural",
                "feature_set": "full",
                "region": "global",
                "split": "test",
                "average_precision": 0.1,
                "reliability_ece": 0.03,
            },
        ]
    ).to_csv(primary / "model_comparison.csv", index=False)

    ranking = write_full_grid_model_selection(
        tmp_path,
        SimpleNamespace(full_grid_selection_metric="reliability_ece"),
    )

    assert ranking.iloc[0]["model_name"] == "B"
    assert ranking.iloc[0]["selection_direction"] == "min"
    assert (primary / "best_model.json").exists()
