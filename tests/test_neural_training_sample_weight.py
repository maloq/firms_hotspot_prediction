from __future__ import annotations

import numpy as np
import pytest

from src.neural_net.train_nn import neural_training_sample_weight


def test_neural_training_sample_weight_is_capped_powered_and_normalized() -> None:
    y = np.array([1, 0, 0, 0], dtype=np.int8)
    class_weight_sample = np.array([4.0, 1.0, 1.0, 1.0], dtype=np.float32)
    prepared = np.array([1.0, 4.0, 16.0, 10_000.0], dtype=np.float32)

    weights, info = neural_training_sample_weight(
        y_train=y,
        class_weight_sample=class_weight_sample,
        prepared_sample_weight=prepared,
        config={
            "enabled": True,
            "power": 0.5,
            "cap_quantile": 0.75,
            "normalize": True,
            "multiply_class_weights": True,
            "normalize_after_multiply": True,
        },
    )

    assert info["used_sample_weight"] is True
    assert info["source"] == "sample_weight"
    assert float(weights.mean()) == pytest.approx(1.0)
    assert weights[-1] == weights.max()
    assert weights[-1] < 4.0
    assert info["positive_final_weight_mean"] > weights[1]


def test_neural_training_sample_weight_returns_class_weights_when_disabled() -> None:
    base = np.array([2.0, 1.0], dtype=np.float32)

    weights, info = neural_training_sample_weight(
        y_train=np.array([1, 0], dtype=np.int8),
        class_weight_sample=base,
        prepared_sample_weight=None,
        config={"enabled": False},
    )

    assert info["used_sample_weight"] is False
    np.testing.assert_allclose(weights, base)
