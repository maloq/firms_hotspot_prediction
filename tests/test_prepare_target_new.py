import numpy as np

from src.target_generation.prepare_target_new import _initial_positive_counts


def test_initial_positive_counts_keep_northwest_triple_weight():
    latitude = np.array([61.0, 56.0, 54.0])
    longitude = np.array([44.0, 50.0, 50.0])

    counts = _initial_positive_counts(latitude, longitude)

    np.testing.assert_array_equal(counts, np.array([3, 2, 1]))
