import pandas as pd

from src.feature_generation.make_features import merge_features


def test_merge_features_uses_configurable_landsea_threshold():
    anchors = pd.DataFrame(
        {
            "count": [0, 0],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        }
    )
    land = pd.DataFrame({"landseamask": [10.0, 40.0]})

    result = merge_features(
        df_with_climate_features_and_target_anchors=anchors,
        all_climate_feature_names=[],
        elevation_features=None,
        elevation_feature_names=[],
        road_features=None,
        road_feature_names=[],
        night_light_features=None,
        night_light_feature_names=[],
        fire_index_features=None,
        fire_index_feature_names=[],
        land_df=land,
        land_feature_names=["landseamask"],
        ecoregion_features=None,
        ecoregion_feature_names=[],
        landsea_distance_features=None,
        landsea_distance_feature_names=[],
        anchor_cols=["count", "datetime"],
        drop_by_sea_mask=True,
        landsea_mask_threshold=30,
    )

    assert result["landseamask"].tolist() == [10.0]
