import numpy as np
import pandas as pd

from train_catboost import (
    expand_soft_labels_for_binary_training,
    probability_calibration_frame,
)


def test_soft_label_expansion_duplicates_uncertain_negatives_with_weights():
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = pd.Series([1, 0, 0])
    soft = pd.Series([1.0, 0.25, 0.0])

    X_fit, y_fit, weights, info = expand_soft_labels_for_binary_training(X, y, soft)

    assert len(X_fit) == 4
    assert y_fit.tolist() == [1, 0, 0, 1]
    np.testing.assert_allclose(weights, np.array([1.0, 0.75, 1.0, 0.25]))
    assert info["soft_negative_rows"] == 1


def test_probability_calibration_frame_excludes_configured_years():
    X = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2021-06-01", "2022-06-01", "2022-07-01"]),
            "month": [6, 6, 7],
        }
    )
    y = pd.Series([1, 0, 1])
    prob = np.array([0.8, 0.2, 0.7])

    frame, info = probability_calibration_frame(
        X,
        y,
        prob,
        date_column="datetime",
        exclude_years=[2021],
    )

    assert info["used_rows"] == 2
    assert info["excluded_rows"] == 1
    assert frame["is_fire"].tolist() == [0, 1]
    assert frame["month"].tolist() == [6, 7]
