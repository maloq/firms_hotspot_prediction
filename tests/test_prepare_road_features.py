import json

import numpy as np
from pyproj import CRS

from src.feature_generation.prepare_road_features import get_road_features_for_coords


def test_get_road_features_from_directory_store(tmp_path):
    feature_dir = tmp_path / "road_features"
    feature_dir.mkdir()

    np.save(feature_dir / "distance_to_road_meters.float32.npy", np.array(
        [[0.0, 1000.0], [2000.0, 3000.0]],
        dtype=np.float32,
    ))
    np.save(feature_dir / "road_density_gaussian_5km.float32.npy", np.array(
        [[0.1, 0.2], [0.3, 0.4]],
        dtype=np.float32,
    ))
    manifest = {
        "format": "road_feature_map_v1",
        "shape": [2, 2],
        "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 2.0],
        "crs_wkt": CRS.from_epsg(4326).to_wkt(),
        "resolution_m": 1000.0,
        "features": {
            "distance_to_road_meters": {
                "path": "distance_to_road_meters.float32.npy",
                "dtype": "float32",
            },
            "road_density_gaussian_5km": {
                "path": "road_density_gaussian_5km.float32.npy",
                "dtype": "float32",
            },
        },
    }
    (feature_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    coords = np.array([[1.5, 0.5, 5.0], [0.5, 1.5, 5.0]], dtype=np.float64)
    result = get_road_features_for_coords(coords, str(feature_dir))

    assert np.allclose(result["distance_to_road_meters"].to_numpy()[:2], [0.0, 3000.0])
    assert np.allclose(result["road_density_gaussian_5km"].to_numpy()[:2], [0.1, 0.4])
    assert np.isnan(result["distance_to_road_meters"].iloc[2])


def test_get_road_features_from_legacy_npz(tmp_path):
    npz_path = tmp_path / "roads_features.npz"
    np.savez(
        npz_path,
        distance_meters=np.array([[0.0, 1000.0], [2000.0, 3000.0]], dtype=np.float32),
        density_5=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        transform=np.array([1.0, 0.0, 0.0, 0.0, -1.0, 2.0]),
        crs_wkt=CRS.from_epsg(4326).to_wkt(),
    )

    coords = np.array([[1.5, 0.5], [0.5, 1.5]], dtype=np.float64)
    result = get_road_features_for_coords(coords, str(npz_path))

    assert np.isclose(result["distance_to_road_meters"].iloc[0], 0.0)
    assert np.isclose(result["density_5"].iloc[0], 0.1)
