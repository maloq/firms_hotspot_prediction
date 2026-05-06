"""Night-light feature sampling wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_generation.prepare_road_features import get_road_features_for_coords


def get_night_light_features_for_coords(
    coords: np.ndarray,
    feature_map_path: str = "data/land_features/night_lights_features_1km",
) -> pd.DataFrame:
    """Sample precomputed night-light feature maps for WGS84 coordinates."""
    return get_road_features_for_coords(coords=coords, npz_path=feature_map_path)
