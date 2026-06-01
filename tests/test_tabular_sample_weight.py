from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.revision_evaluation.tabular import catboost_training_sample_weight


def test_catboost_training_sample_weight_is_capped_powered_and_normalized() -> None:
    frame = pd.DataFrame(
        {
            "count": [1, 0, 0, 0],
            "sample_weight": [1.0, 4.0, 16.0, 10_000.0],
        }
    )
    weights, info = catboost_training_sample_weight(
        frame,
        np.array([True, True, True, True]),
        column="sample_weight",
        power=0.5,
        cap_quantile=0.75,
    )

    assert info["used_sample_weight"] is True
    assert info["sample_weight_column"] == "sample_weight"
    assert weights is not None
    assert float(weights.mean()) == pytest.approx(1.0)
    assert weights[-1] == weights.max()
    assert weights[-1] < 4.0
