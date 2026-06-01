from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from src.feature_generation import prepare_night_light_features as night_lights
from src.feature_generation.prepare_night_light_features import (
    _available_annual_sources,
    _apply_cf_cvg_filter,
    _nearest_available_years,
    _sample_black_marble_features,
    get_night_light_features_for_coords,
    get_recent_night_light_radiance_for_coords,
)


def test_nearest_available_years_prefers_closest_and_earlier_ties():
    chosen = _nearest_available_years(
        [1999, 2000, 2001, 2002, 2024, 2025, np.nan],
        [2000, 2002, 2024],
    )

    assert chosen.tolist() == [2000, 2000, 2000, 2002, 2024, 2024, 2024]


def test_available_annual_sources_discovers_zenodo_viirs_files(tmp_path: Path):
    expected = tmp_path / (
        "nightlights.average_viirs.v21_m_500m_s_20230101_"
        "20231231_go_epsg4326_v20250904.tif"
    )
    expected.touch()
    (tmp_path / "not_a_viirs_file.tif").touch()

    sources = _available_annual_sources(tmp_path)

    assert sources == {2023: expected}


def test_available_annual_sources_discovers_eog_vnl_v22_files(tmp_path: Path):
    median = tmp_path / "VNL_npp_2024_global_vcmslcfg_v2_c202502261200.median_masked.dat.tif"
    cf_cvg = tmp_path / "VNL_npp_2024_global_vcmslcfg_v2_c202502261200.cf_cvg.dat.tif"
    median.touch()
    cf_cvg.touch()

    assert _available_annual_sources(tmp_path, "*median_masked*.tif") == {2024: median}
    assert _available_annual_sources(tmp_path, "*cf_cvg*.tif") == {2024: cf_cvg}


def test_recent_radiance_uses_cache_without_opening_tiff(tmp_path: Path):
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    (
        source_dir
        / "nightlights.average_viirs.v21_m_500m_s_20240101_20241231_go_epsg4326_v20250904.tif"
    ).touch()
    cache_path = tmp_path / "recent_cache.parquet"
    pd.DataFrame(
        {
            "lat_key": [10_000_000],
            "lon_key": [20_000_000],
            "source_year": [2024],
            "lat": [10.0],
            "lon": [20.0],
            "night_light_radiance_recent": [42.0],
        }
    ).to_parquet(cache_path, index=False)

    values = get_recent_night_light_radiance_for_coords(
        coords=np.array([[10.0], [20.0]]),
        years=[2025],
        annual_source_dir=source_dir,
        cache_path=cache_path,
    )

    assert values["night_light_radiance_recent"].tolist() == [42.0]


def test_cf_cvg_filter_only_zeroes_low_coverage_northern_pixels():
    features = pd.DataFrame(
        {
            "night_light_radiance_recent": [10.0, 20.0, 30.0],
            "night_light_cf_cvg_recent": [1.0, 1.0, 10.0],
        }
    )

    _apply_cf_cvg_filter(
        features,
        coords=np.array([[57.0, 60.0, 61.0], [30.0, 30.0, 30.0]]),
        radiance_feature_name="night_light_radiance_recent",
        coverage_feature_name="night_light_cf_cvg_recent",
        filtered_feature_name="night_light_radiance_recent_cf_filtered",
        min_cf_cvg=5.0,
        north_lat_min=58.0,
    )

    assert features["night_light_radiance_recent_cf_filtered"].tolist() == [10.0, 0.0, 30.0]


def test_legacy_viirs_feature_map_is_prefixed(monkeypatch):
    def fake_raster_features(coords, npz_path):
        if npz_path == "black_marble":
            return pd.DataFrame({"night_light_radiance_2024": [10.0]})
        return pd.DataFrame(
            {
                "night_light_radiance_2024": [5.0],
                "night_light_presence_1km": [1],
            }
        )

    monkeypatch.setattr(night_lights, "get_road_features_for_coords", fake_raster_features)

    features = get_night_light_features_for_coords(
        coords=np.array([[10.0], [20.0]]),
        feature_map_path="black_marble",
        legacy_viirs_feature_map_path="legacy_viirs",
    )

    assert features["night_light_radiance_2024"].tolist() == [10.0]
    assert features["viirs_night_light_radiance_2024"].tolist() == [5.0]
    assert features["viirs_night_light_presence_1km"].tolist() == [1]


def test_black_marble_sampling_applies_scale_and_quality_filter(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(night_lights, "BLACK_MARBLE_TILE_SIZE", 10)
    source_dir = tmp_path / "black_marble"
    tile_path = source_dir / "2024" / "VNP46A4.A2024001.h18v05.002.2025010100000.h5"
    tile_path.parent.mkdir(parents=True)

    radiance = np.zeros((10, 10), dtype=np.uint16)
    quality = np.full((10, 10), 255, dtype=np.uint8)
    observations = np.zeros((10, 10), dtype=np.uint16)
    radiance[4, 6] = 123
    radiance[4, 7] = 200
    quality[4, 6] = 0
    quality[4, 7] = 1
    observations[4, 6] = 7
    observations[4, 7] = 9

    with h5py.File(tile_path, "w") as handle:
        group = handle.create_group("HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields")
        radiance_ds = group.create_dataset("NearNadir_Composite_Snow_Free", data=radiance)
        radiance_ds.attrs["scale_factor"] = 0.1
        radiance_ds.attrs["_FillValue"] = np.uint16(65535)
        quality_ds = group.create_dataset("NearNadir_Composite_Snow_Free_Quality", data=quality)
        quality_ds.attrs["_FillValue"] = np.uint8(255)
        observations_ds = group.create_dataset("NearNadir_Composite_Snow_Free_Num", data=observations)
        observations_ds.attrs["_FillValue"] = np.uint16(65535)

    features = _sample_black_marble_features(
        coords=np.array([[35.5, 35.5], [6.5, 7.5]]),
        years=[2024, 2024],
        source_dir=source_dir,
        radiance_sds="NearNadir_Composite_Snow_Free",
        quality_sds="NearNadir_Composite_Snow_Free_Quality",
        observations_sds="NearNadir_Composite_Snow_Free_Num",
        radiance_feature_name="radiance",
        quality_feature_name="quality",
        observations_feature_name="observations",
        filtered_feature_name="filtered",
        quality_keep_values=(0,),
        cache_path=None,
    )

    assert features["radiance"].tolist() == pytest.approx([12.3, 20.0])
    assert features["quality"].tolist() == [0.0, 1.0]
    assert features["observations"].tolist() == [7.0, 9.0]
    assert features["filtered"].tolist() == pytest.approx([12.3, 0.0])


def test_black_marble_fallback_does_not_mark_quality_as_good(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(night_lights, "BLACK_MARBLE_TILE_SIZE", 10)
    source_dir = tmp_path / "black_marble"
    source_dir.mkdir()
    cache_path = tmp_path / "black_marble_cache.parquet"
    pd.DataFrame(
        {
            "lat_key": [35_500_000],
            "lon_key": [6_500_000],
            "source_year": [2024],
            "lat": [35.5],
            "lon": [6.5],
            "radiance": [1.0],
            "quality": [0.0],
            "observations": [1.0],
        }
    ).to_parquet(cache_path, index=False)

    features = _sample_black_marble_features(
        coords=np.array([[35.5], [7.5]]),
        years=[2024],
        source_dir=source_dir,
        radiance_sds="NearNadir_Composite_Snow_Free",
        quality_sds="NearNadir_Composite_Snow_Free_Quality",
        observations_sds="NearNadir_Composite_Snow_Free_Num",
        radiance_feature_name="radiance",
        quality_feature_name="quality",
        observations_feature_name="observations",
        filtered_feature_name="filtered",
        quality_keep_values=(0,),
        cache_path=cache_path,
        missing_tile_strategy="feature_map",
        fallback_radiance_values=np.array([12.0], dtype=np.float32),
    )

    assert features["radiance"].tolist() == [12.0]
    assert features["quality"].tolist() == [255.0]
    assert features["observations"].tolist() == [0.0]
    assert features["filtered"].tolist() == [0.0]
