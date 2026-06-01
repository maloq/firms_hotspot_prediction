from __future__ import annotations

import pandas as pd

from src.feature_generation.calendar_features import (
    CALENDAR_CONTEXT_FEATURE_COLUMNS,
    add_calendar_context_features,
)


def test_add_calendar_context_features_weekend_and_yesterday_holiday() -> None:
    frame = pd.DataFrame(
        {
            "datetime": ["2024-05-10", "2024-05-11", "2024-01-02"],
            "country": ["Russian_Federation", "Russian_Federation", "Russian_Federation"],
        }
    )

    out = add_calendar_context_features(frame)

    assert CALENDAR_CONTEXT_FEATURE_COLUMNS == [
        col for col in out.columns if col.startswith("is_weekend_") or col.startswith("is_holiday_")
    ]
    assert out.loc[0, "is_weekend_today"] == 0
    assert out.loc[0, "is_holiday_yesterday"] == 1
    assert out.loc[0, "is_holiday_today_or_yesterday"] == 1
    assert out.loc[1, "is_weekend_today"] == 1
    assert out.loc[1, "is_weekend_today_or_yesterday"] == 1
    assert out.loc[2, "is_holiday_today"] == 1
    assert out.loc[2, "is_holiday_yesterday"] == 1


def test_add_calendar_context_features_without_country_uses_common_fixed_holidays() -> None:
    frame = pd.DataFrame({"datetime": ["2024-12-26", "2024-03-08"]})

    out = add_calendar_context_features(frame)

    assert out.loc[0, "is_holiday_yesterday"] == 1
    assert out.loc[0, "is_holiday_today_or_yesterday"] == 1
    assert out.loc[1, "is_holiday_today"] == 0
