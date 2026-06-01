from __future__ import annotations

import pandas as pd


CALENDAR_CONTEXT_FEATURE_COLUMNS = [
    "is_weekend_today",
    "is_weekend_yesterday",
    "is_weekend_today_or_yesterday",
    "is_holiday_today",
    "is_holiday_yesterday",
    "is_holiday_today_or_yesterday",
]


_COMMON_FIXED_HOLIDAYS = {
    (1, 1),
    (5, 1),
    (12, 25),
}

_RUSSIA_FIXED_HOLIDAYS = {
    *{(1, day) for day in range(1, 9)},
    (2, 23),
    (3, 8),
    (5, 1),
    (5, 9),
    (6, 12),
    (11, 4),
}

_FORMER_SOVIET_FIXED_HOLIDAYS = {
    (1, 1),
    (1, 7),
    (3, 8),
    (5, 1),
    (5, 9),
}


def _country_key(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _fixed_holiday_pairs(country: object) -> set[tuple[int, int]]:
    key = _country_key(country)
    if key in {"russian_federation", "russia", "ru", "rus"}:
        return _RUSSIA_FIXED_HOLIDAYS
    if key in {
        "belarus",
        "kazakhstan",
        "ukraine",
        "armenia",
        "azerbaijan",
        "georgia",
        "kyrgyzstan",
        "moldova",
    }:
        return _FORMER_SOVIET_FIXED_HOLIDAYS
    return _COMMON_FIXED_HOLIDAYS


def _is_fixed_holiday(dates: pd.Series, countries: pd.Series | None) -> pd.Series:
    month_day = list(zip(dates.dt.month.astype("Int64"), dates.dt.day.astype("Int64")))
    if countries is None:
        holidays = _COMMON_FIXED_HOLIDAYS
        return pd.Series([pair in holidays for pair in month_day], index=dates.index, dtype=bool)

    out = []
    for pair, country in zip(month_day, countries):
        out.append(pair in _fixed_holiday_pairs(country))
    return pd.Series(out, index=dates.index, dtype=bool)


def add_calendar_context_features(
    frame: pd.DataFrame,
    *,
    date_col: str = "datetime",
    country_col: str = "country",
) -> pd.DataFrame:
    """Add weekend and fixed-holiday context for the event date and previous day."""

    if date_col not in frame.columns:
        return frame

    out = frame.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    yesterday = dates - pd.Timedelta(days=1)
    countries = out[country_col] if country_col in out.columns else None

    weekend_today = dates.dt.weekday >= 5
    weekend_yesterday = yesterday.dt.weekday >= 5
    holiday_today = _is_fixed_holiday(dates, countries)
    holiday_yesterday = _is_fixed_holiday(yesterday, countries)

    out["is_weekend_today"] = weekend_today.fillna(False).astype("int8")
    out["is_weekend_yesterday"] = weekend_yesterday.fillna(False).astype("int8")
    out["is_weekend_today_or_yesterday"] = (
        weekend_today.fillna(False) | weekend_yesterday.fillna(False)
    ).astype("int8")
    out["is_holiday_today"] = holiday_today.fillna(False).astype("int8")
    out["is_holiday_yesterday"] = holiday_yesterday.fillna(False).astype("int8")
    out["is_holiday_today_or_yesterday"] = (
        holiday_today.fillna(False) | holiday_yesterday.fillna(False)
    ).astype("int8")
    return out
