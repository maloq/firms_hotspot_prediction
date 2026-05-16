from __future__ import annotations

import numpy as np
import xarray as xr

from src.feature_generation.load_climate_data import _padded_coord_range


def test_padded_coord_range_keeps_nearest_grid_cells_at_chunk_edges():
    coord = xr.DataArray(np.arange(35.0, 76.0, 1.0), dims=["latitude"])

    lower, upper = _padded_coord_range((41.2, 50.7), coord)

    assert lower < 40.7
    assert upper > 51.2
