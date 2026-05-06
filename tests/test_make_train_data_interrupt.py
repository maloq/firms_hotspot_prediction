import pytest
import pandas as pd

import make_train_data


def test_sequential_run_stops_on_keyboard_interrupt(monkeypatch):
    calls = []
    worker_args = [
        ({}, "First", "out", "target_cache", "climate_cache", False, True, False),
        ({}, "Second", "out", "target_cache", "climate_cache", False, True, False),
    ]

    def raise_interrupt(*args):
        calls.append(args[1])
        raise KeyboardInterrupt

    monkeypatch.setattr(make_train_data, "_make_country_features", raise_interrupt)

    with pytest.raises(SystemExit) as exc_info:
        make_train_data._run_sequential(worker_args)

    assert exc_info.value.code == 130
    assert calls == ["First"]


def test_single_pass_splits_combined_features_by_country(monkeypatch, tmp_path):
    def fake_load_target(base_config, country, target_cache_dir, test_mode, use_cache):
        return pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2020-01-01")],
                "lat_rounded": [10.0],
                "lon_rounded": [20.0],
                "month": [1],
                "day": [1],
                "year": [2020],
                "count": [1],
                "_source_country": [country],
            }
        )

    def fake_make_features_from_target_df(config, combined_target, **kwargs):
        return combined_target.assign(feature_value=range(len(combined_target)))

    monkeypatch.setattr(make_train_data, "_load_country_target", fake_load_target)
    monkeypatch.setattr(make_train_data, "make_features_from_target_df", fake_make_features_from_target_df)

    make_train_data._run_single_pass(
        base_config={},
        countries=["A", "B"],
        output_dir=str(tmp_path),
        target_cache_dir=str(tmp_path / "targets"),
        climate_cache_dir=str(tmp_path / "climate"),
        test_mode=False,
        use_cache=True,
        force=True,
    )

    a_df = pd.read_parquet(tmp_path / "train_test_features_30d_A.parquet")
    b_df = pd.read_parquet(tmp_path / "train_test_features_30d_B.parquet")

    assert "_source_country" not in a_df.columns
    assert "_source_country" not in b_df.columns
    assert len(a_df) == 1
    assert len(b_df) == 1
