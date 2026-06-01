"""Select and plot high-skill non-winter fire periods.

This stage complements the probability overlay maps with compact temporal
strips: forecast fire-positive grid-cell counts, observed MODIS-derived active
fire labels, and an observed fire-positive area proxy for the same selected
period. If prediction files carry a forecast lead-time column, the top panel is
drawn by lead day; otherwise it shows the available model prediction row.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

from .probability_overlays import (  # noqa: E402
    DATE_COL,
    DEFAULT_PROB_COL,
    LAT_COL,
    LON_COL,
    METRIC_DIRECTIONS,
    PROB_COL,
    TARGET_COL,
    WEIGHT_COL,
    Region,
    compact_overlay_table,
    find_prediction_files_for_model,
    load_regions,
    load_world_boundaries,
    load_yaml,
    metric_formatter,
    period_metrics_for_model,
    probability_colormap,
    plot_selected_period,
    read_prediction_columns,
    safe_slug,
    select_best_periods,
    write_wide_jsonl,
)


LEAD_COLUMN_CANDIDATES = [
    "lead_time_days",
    "lead_days",
    "forecast_lead_days",
    "prediction_lead_days",
    "horizon_days",
    "forecast_day",
    "lead_time",
    "leadtime",
    "leadtime_hours",
    "step",
]


@dataclass(frozen=True)
class FirePeriodTimelineConfig:
    results_dir: Path
    regions_file: Path = Path("configs/regions_example.yaml")
    feature_config: Path = Path("configs/features_config_30d.yaml")
    target_config: Path = Path("configs/target_config.yaml")
    source: str = "legacy"
    model: str = "best_neural"
    prob_col: str = DEFAULT_PROB_COL
    selection_metric: str = "average_precision"
    min_wildfires: int = 7
    spatial_tolerance_degrees: float = 0.0
    window_days: int = 28
    top_periods: int = 1
    allow_overlapping_periods: bool = False
    regions: list[str] | None = None
    include_global: bool = True
    allow_partial_periods: bool = False
    excluded_months: list[int] = field(default_factory=lambda: [12, 1, 2])
    prefer_centered_activity: bool = True
    center_peak_min_fraction: float = 0.25
    center_peak_max_fraction: float = 0.75
    max_start_activity_fraction: float = 0.15
    min_middle_activity_fraction: float = 0.50
    output_dir: Path | None = None
    source_label: str | None = None
    formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 320
    lead_column: str | None = None
    max_lead_days: int = 10
    burned_area_label: str = "Burnt Area"
    count_colormap: str = "fire_risk"
    count_norm_gamma: float = 0.42
    count_vmax_percentile: float = 95.0
    generate_overlay_maps: bool = True
    overlay_surface_source: str = "dense-neural"
    overlay_window_days: int | None = 3
    overlay_center_on_observed_peak: bool = True
    overlay_map_summary: str = "sum"
    overlay_dense_model_path: Path | None = None
    overlay_dense_neural_model_path: Path | None = None
    overlay_dense_neural_training_features: Path = Path(
        "data/saved_features_boost/train_test_features_30d_all_extended_north_14cb_laggedfire.parquet"
    )
    overlay_dense_neural_batch_size: int = 8192
    overlay_dense_neural_device: str = "auto"
    overlay_overwrite_dense: bool = False
    overlay_grid_resolution: float | None = None
    overlay_interpolation_factor: int = 5
    overlay_prior_correction: bool = True
    overlay_train_prior: float = 0.15
    overlay_deploy_prior: float = 0.001
    overlay_colormap: str = "YlOrRd"
    overlay_color_floor: float | None = None
    overlay_color_vmax: float | None = None
    overlay_verbose_feature_generation: bool = False
    overlay_country_shapes: Path = Path("data/countries")


def default_output_dir(results_dir: Path) -> Path:
    return results_dir / "shared_artifacts" / "fire_period_timelines"


def season_mask(metrics: pd.DataFrame, excluded_months: Sequence[int]) -> pd.Series:
    excluded = {int(month) for month in excluded_months}
    if not excluded:
        return pd.Series(True, index=metrics.index)
    starts = pd.to_datetime(metrics["period_start"], errors="coerce")
    ends = pd.to_datetime(metrics["period_end"], errors="coerce")
    keep: list[bool] = []
    for start, end in zip(starts, ends):
        if pd.isna(start) or pd.isna(end) or end < start:
            keep.append(False)
            continue
        months = set(pd.date_range(start.normalize(), end.normalize(), freq="D").month)
        keep.append(months.isdisjoint(excluded))
    return pd.Series(keep, index=metrics.index)


def resolve_lead_column(path: Path, requested: str | None) -> str | None:
    schema_names = set(pq.read_schema(path).names)
    if requested:
        if requested not in schema_names:
            raise ValueError(f"{path} is missing requested lead-time column {requested!r}.")
        return requested
    for candidate in LEAD_COLUMN_CANDIDATES:
        if candidate in schema_names:
            return candidate
    return None


def read_prediction_columns_with_lead(path: Path, prob_col: str, lead_column: str | None) -> tuple[pd.DataFrame, str | None]:
    frame = read_prediction_columns(path, prob_col)
    resolved = resolve_lead_column(path, lead_column)
    if resolved is None:
        return frame, None
    lead_values = pd.read_parquet(path, columns=[resolved])[resolved]
    frame = frame.copy()
    frame[resolved] = lead_values.reindex(frame.index).to_numpy()
    return frame, resolved


def normalised_lead_days(values: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    lower = column.lower()
    finite = numeric[np.isfinite(numeric)]
    if "hour" in lower or (not finite.empty and finite.max() > 60 and np.allclose((finite % 24).fillna(0), 0)):
        numeric = numeric / 24.0
    return numeric.round().astype("Int64")


def selected_period_frame(frame: pd.DataFrame, selection: pd.Series, region: Region) -> pd.DataFrame:
    period_start = pd.Timestamp(selection["period_start"]).normalize()
    period_end = pd.Timestamp(selection["period_end"]).normalize()
    mask = (frame[DATE_COL] >= period_start) & (frame[DATE_COL] <= period_end) & region.mask(frame)
    return frame.loc[mask].copy()


def temporal_activity_summary(daily_counts: np.ndarray) -> dict[str, float]:
    values = np.asarray(daily_counts, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    window_days = int(len(values))
    total = float(values.sum())
    if window_days == 0 or total <= 0:
        return {
            "observed_peak_day_index": math.nan,
            "observed_center_of_mass_day": math.nan,
            "first_quarter_fire_fraction": math.nan,
            "middle_half_fire_fraction": math.nan,
            "centered_activity_score": math.nan,
        }

    indices = np.arange(window_days, dtype=float)
    center = max((window_days - 1) / 2.0, 1.0)
    peak_idx = int(np.argmax(values))
    center_of_mass = float(np.sum(values * indices) / total)
    first_days = max(1, int(math.ceil(window_days * 0.25)))
    middle_start = max(0, int(math.floor(window_days * 0.25)))
    middle_end = min(window_days, int(math.ceil(window_days * 0.75)))
    first_fraction = float(values[:first_days].sum() / total)
    middle_fraction = float(values[middle_start:middle_end].sum() / total)
    peak_score = max(0.0, 1.0 - abs(float(peak_idx) - center) / center)
    mass_score = max(0.0, 1.0 - abs(center_of_mass - center) / center)
    centered_activity_score = (
        0.35 * peak_score
        + 0.30 * mass_score
        + 0.20 * middle_fraction
        + 0.15 * max(0.0, 1.0 - first_fraction)
    )
    return {
        "observed_peak_day_index": float(peak_idx),
        "observed_center_of_mass_day": center_of_mass,
        "first_quarter_fire_fraction": first_fraction,
        "middle_half_fire_fraction": middle_fraction,
        "centered_activity_score": centered_activity_score,
    }


def add_temporal_activity_metrics(
    metrics: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    regions: Sequence[Region],
) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    out = metrics.copy()
    for column in [
        "observed_peak_day_index",
        "observed_center_of_mass_day",
        "first_quarter_fire_fraction",
        "middle_half_fire_fraction",
        "centered_activity_score",
    ]:
        out[column] = math.nan

    for region in regions:
        metric_mask = out["region"].eq(region.name)
        if not metric_mask.any():
            continue
        region_frame = frame.loc[region.mask(frame)]
        if region_frame.empty:
            continue
        daily = (
            region_frame.assign(observed=region_frame[TARGET_COL] * region_frame[WEIGHT_COL])
            .groupby(DATE_COL, observed=True)["observed"]
            .sum()
        )
        for idx, row in out.loc[metric_mask].iterrows():
            dates = pd.date_range(
                pd.Timestamp(row["period_start"]).normalize(),
                pd.Timestamp(row["period_end"]).normalize(),
                freq="D",
            )
            summary = temporal_activity_summary(daily.reindex(dates, fill_value=0.0).to_numpy(dtype=float))
            for column, value in summary.items():
                out.at[idx, column] = value

    return out


def select_centered_fire_periods(
    metrics: pd.DataFrame,
    *,
    metric: str,
    min_wildfires: int,
    top_periods: int,
    allow_overlapping_periods: bool,
    prefer_centered_activity: bool,
    center_peak_min_fraction: float,
    center_peak_max_fraction: float,
    max_start_activity_fraction: float,
    min_middle_activity_fraction: float,
) -> pd.DataFrame:
    if not prefer_centered_activity:
        return select_best_periods(
            metrics,
            metric=metric,
            min_wildfires=min_wildfires,
            top_periods=top_periods,
            allow_overlapping_periods=allow_overlapping_periods,
        )
    if metric not in METRIC_DIRECTIONS:
        raise ValueError(f"Unsupported selection metric: {metric}")
    if metrics.empty:
        raise ValueError("No fire-period timeline metrics were computed.")
    if top_periods < 1:
        raise ValueError(f"top_periods must be >= 1; received {top_periods}.")

    selections: list[pd.Series] = []
    for region, group in metrics.groupby("region", observed=True, sort=False):
        direction = METRIC_DIRECTIONS[metric]
        candidates = group[group["observed_positive_locations"] >= int(min_wildfires)].copy()
        if candidates.empty:
            raise ValueError(
                f"No fire-period timeline period for region {region!r} has at least "
                f"{int(min_wildfires)} observed positive locations."
            )
        metric_values = pd.to_numeric(candidates[metric], errors="coerce")
        candidates = candidates[np.isfinite(metric_values)].copy()
        if candidates.empty:
            raise ValueError(
                f"No finite {metric} fire-period timeline period for region {region!r} "
                f"after applying min_wildfires={int(min_wildfires)}."
            )

        window_days = int(candidates["window_days"].iloc[0])
        last_day = max(window_days - 1, 1)
        peak_min = int(math.floor(last_day * float(center_peak_min_fraction)))
        peak_max = int(math.ceil(last_day * float(center_peak_max_fraction)))
        peak_day = pd.to_numeric(candidates["observed_peak_day_index"], errors="coerce")
        start_fraction = pd.to_numeric(candidates["first_quarter_fire_fraction"], errors="coerce")
        middle_fraction = pd.to_numeric(candidates["middle_half_fire_fraction"], errors="coerce")
        shape_mask = (
            peak_day.between(peak_min, peak_max, inclusive="both")
            & (start_fraction <= float(max_start_activity_fraction))
            & (middle_fraction >= float(min_middle_activity_fraction))
        )
        fallback_reason = ""
        if shape_mask.any():
            candidates = candidates.loc[shape_mask].copy()
        else:
            fallback_reason = (
                "No period satisfied centered-activity thresholds; ranked by centered-activity score."
            )
            candidates = candidates[np.isfinite(pd.to_numeric(candidates["centered_activity_score"], errors="coerce"))]
            if candidates.empty:
                return select_best_periods(
                    group,
                    metric=metric,
                    min_wildfires=min_wildfires,
                    top_periods=top_periods,
                    allow_overlapping_periods=allow_overlapping_periods,
                )

        ascending = direction == "min"
        if fallback_reason:
            sort_cols = ["centered_activity_score", metric, "count_abs_error", "weighted_brier_score", "period_start"]
            sort_ascending = [False, ascending, True, True, True]
        else:
            sort_cols = [metric, "centered_activity_score", "count_abs_error", "weighted_brier_score", "period_start"]
            sort_ascending = [ascending, False, True, True, True]
        ranked = candidates.sort_values(sort_cols, ascending=sort_ascending, na_position="last").copy()

        picked: list[pd.Series] = []
        picked_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for _, candidate in ranked.iterrows():
            period_start = pd.Timestamp(candidate["period_start"])
            period_end = pd.Timestamp(candidate["period_end"])
            overlaps = any(period_start <= end and period_end >= start for start, end in picked_intervals)
            if allow_overlapping_periods or not overlaps:
                picked.append(candidate.copy())
                picked_intervals.append((period_start, period_end))
            if len(picked) >= top_periods:
                break

        if len(picked) < top_periods:
            raise ValueError(
                f"Only {len(picked)} centered fire-period timeline period(s) were available for "
                f"region {region!r}; requested {top_periods}."
            )

        for rank, selected in enumerate(picked, start=1):
            selected["selection_metric"] = metric
            selected["selection_direction"] = direction
            selected["selection_fallback_reason"] = fallback_reason
            selected["period_rank"] = rank
            selected["center_peak_day_min"] = peak_min
            selected["center_peak_day_max"] = peak_max
            selected["max_start_activity_fraction"] = float(max_start_activity_fraction)
            selected["min_middle_activity_fraction"] = float(min_middle_activity_fraction)
            selections.append(selected)

    return pd.DataFrame(selections).reset_index(drop=True)


def date_index(selection: pd.Series) -> pd.DatetimeIndex:
    return pd.date_range(
        pd.Timestamp(selection["period_start"]).normalize(),
        pd.Timestamp(selection["period_end"]).normalize(),
        freq="D",
    )


def daily_prediction_matrix(
    period: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    lead_column: str | None,
    max_lead_days: int,
    fallback_label: str,
) -> tuple[np.ndarray, list[str], str | None]:
    if lead_column and lead_column in period.columns:
        work = period.copy()
        work["__lead_day"] = normalised_lead_days(work[lead_column], lead_column)
        work = work.dropna(subset=["__lead_day"])
        work["__lead_day"] = work["__lead_day"].astype(int)
        work = work[(work["__lead_day"] >= 1) & (work["__lead_day"] <= int(max_lead_days))]
        if not work.empty:
            daily = (
                work.assign(expected=work[PROB_COL] * work[WEIGHT_COL])
                .groupby(["__lead_day", DATE_COL], observed=True)["expected"]
                .sum()
                .reset_index()
            )
            leads = sorted(int(value) for value in daily["__lead_day"].dropna().unique())
            leads = list(reversed(leads))
            matrix = np.zeros((len(leads), len(dates)), dtype=float)
            date_pos = {date: idx for idx, date in enumerate(dates)}
            lead_pos = {lead: idx for idx, lead in enumerate(leads)}
            for _, row in daily.iterrows():
                date = pd.Timestamp(row[DATE_COL]).normalize()
                lead = int(row["__lead_day"])
                if date in date_pos and lead in lead_pos:
                    matrix[lead_pos[lead], date_pos[date]] = float(row["expected"])
            return matrix, [f"Day {lead}" for lead in leads], lead_column

    daily = (
        period.assign(expected=period[PROB_COL] * period[WEIGHT_COL])
        .groupby(DATE_COL, observed=True)["expected"]
        .sum()
        .reindex(dates, fill_value=0.0)
    )
    return daily.to_numpy(dtype=float)[None, :], [fallback_label], None


def grid_cell_area_kha(latitudes: np.ndarray, resolution: float) -> np.ndarray:
    radius_km = 6371.0088
    half = math.radians(float(resolution) / 2.0)
    dlon = math.radians(float(resolution))
    lat_rad = np.radians(np.asarray(latitudes, dtype=float))
    area_km2 = radius_km * radius_km * dlon * (np.sin(lat_rad + half) - np.sin(lat_rad - half))
    return np.maximum(area_km2, 0.0) / 10.0


def daily_observed_counts(period: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    return (
        period.assign(observed=period[TARGET_COL] * period[WEIGHT_COL])
        .groupby(DATE_COL, observed=True)["observed"]
        .sum()
        .reindex(dates, fill_value=0.0)
    )


def daily_fire_positive_area(period: pd.DataFrame, dates: pd.DatetimeIndex, resolution: float) -> pd.Series:
    positives = period.loc[period[TARGET_COL] > 0, [DATE_COL, LAT_COL, LON_COL]].drop_duplicates()
    if positives.empty:
        return pd.Series(0.0, index=dates)
    positives = positives.copy()
    positives["area_kha"] = grid_cell_area_kha(positives[LAT_COL].to_numpy(dtype=float), resolution)
    return positives.groupby(DATE_COL, observed=True)["area_kha"].sum().reindex(dates, fill_value=0.0)


def date_edges(dates: pd.DatetimeIndex) -> np.ndarray:
    centers = mdates.date2num(dates.to_pydatetime())
    if len(centers) == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5])
    step = np.median(np.diff(centers))
    return np.concatenate([[centers[0] - step / 2.0], centers + step / 2.0])


def compact_count_label(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value > 0:
        return f"{value:.1f}"
    return "0"


def period_slug(selection: pd.Series) -> str:
    return (
        f"{safe_slug(selection['model_name'])}_{safe_slug(selection['region'])}_"
        f"rank{int(selection.get('period_rank', 1)):02d}_"
        f"{pd.Timestamp(selection['period_start']):%Y%m%d}_"
        f"{int(selection.get('window_days', 0))}d"
    )


def overlay_selection_for_timeline(selection: pd.Series, *, window_days: int | None, center_on_peak: bool) -> pd.Series:
    if window_days is None:
        return selection.copy()
    overlay_days = int(window_days)
    timeline_days = int(selection.get("window_days", overlay_days))
    if overlay_days < 1:
        raise ValueError("fire_period_timeline_overlay_window_days must be >= 1 when set")
    if overlay_days >= timeline_days:
        return selection.copy()

    timeline_start = pd.Timestamp(selection["period_start"]).normalize()
    timeline_end = pd.Timestamp(selection["period_end"]).normalize()
    if center_on_peak and pd.notna(selection.get("observed_peak_day_index", math.nan)):
        center_offset = int(round(float(selection["observed_peak_day_index"])))
    else:
        center_offset = timeline_days // 2
    start_offset = center_offset - overlay_days // 2
    start_offset = max(0, min(start_offset, timeline_days - overlay_days))
    overlay_start = timeline_start + pd.Timedelta(days=start_offset)
    overlay_end = overlay_start + pd.Timedelta(days=overlay_days - 1)
    overlay_end = min(overlay_end, timeline_end)

    out = selection.copy()
    out["timeline_period_start"] = selection["period_start"]
    out["timeline_period_end"] = selection["period_end"]
    out["timeline_window_days"] = timeline_days
    out["period_start"] = overlay_start.date().isoformat()
    out["period_end"] = overlay_end.date().isoformat()
    out["window_days"] = int((overlay_end - overlay_start).days + 1)
    return out


def plot_fire_period_timeline(
    frame: pd.DataFrame,
    *,
    selection: pd.Series,
    region: Region,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
    grid_resolution: float,
    source_label: str | None,
    lead_column: str | None,
    max_lead_days: int,
    burned_area_label: str,
    count_colormap: str,
    count_norm_gamma: float,
    count_vmax_percentile: float,
) -> list[Path]:
    period = selected_period_frame(frame, selection, region)
    if period.empty:
        return []

    dates = date_index(selection)
    model_label = str(selection.get("model_name", "Forecast"))
    fallback_label = str(source_label or "Forecast")
    matrix, row_labels, used_lead_column = daily_prediction_matrix(
        period,
        dates,
        lead_column=lead_column,
        max_lead_days=max_lead_days,
        fallback_label=fallback_label,
    )
    observed = daily_observed_counts(period, dates)
    area = daily_fire_positive_area(period, dates, grid_resolution)

    peak_count = float(max(np.nanmax(matrix) if matrix.size else 0.0, observed.max()))
    count_values = np.concatenate([np.ravel(matrix), observed.to_numpy(dtype=float)])
    positive_counts = count_values[np.isfinite(count_values) & (count_values > 0)]
    if len(positive_counts):
        percentile = min(100.0, max(50.0, float(count_vmax_percentile)))
        display_vmax = float(np.nanpercentile(positive_counts, percentile))
    else:
        display_vmax = 1.0
    vmax = max(1.0, min(max(peak_count, 1.0), display_vmax))
    gamma = float(count_norm_gamma)
    if not math.isfinite(gamma) or gamma <= 0:
        gamma = 1.0
    peak_area = float(area.max()) if not area.empty else 0.0

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "figure.titlesize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    row_count = max(len(row_labels), 1)
    fig_height = max(4.6, 2.65 + 0.32 * row_count)
    fig = plt.figure(figsize=(11.8, fig_height))
    grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[max(1.45, 0.26 * row_count + 0.8), 0.48, 0.62],
        hspace=0.36,
    )
    ax_forecast = fig.add_subplot(grid[0])
    ax_observed = fig.add_subplot(grid[1], sharex=ax_forecast)
    ax_area = fig.add_subplot(grid[2], sharex=ax_forecast)

    x_edges = date_edges(dates)
    y_edges = np.arange(row_count + 1)
    cmap = probability_colormap(count_colormap)
    norm = mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax, clip=False)
    mesh = ax_forecast.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    ax_forecast.set_yticks(np.arange(row_count) + 0.5)
    ax_forecast.set_yticklabels(row_labels)
    ax_forecast.invert_yaxis()
    start = pd.Timestamp(selection["period_start"])
    end = pd.Timestamp(selection["period_end"])
    title = f"Forecast   {start:%-d %b %Y} to {end:%-d %b %Y}"
    ax_forecast.set_title(title, loc="left", fontweight="bold", pad=7)
    ax_forecast.text(
        0.995,
        1.11,
        "Fire count",
        transform=ax_forecast.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
    )
    count_extend = "max" if peak_count > vmax * 1.001 else "neither"
    cbar = fig.colorbar(
        mesh,
        ax=ax_forecast,
        orientation="horizontal",
        fraction=0.08,
        pad=0.05,
        extend=count_extend,
    )
    cbar.set_ticks([0.0, vmax])
    vmax_label = compact_count_label(vmax)
    if count_extend == "max":
        vmax_label = f">={vmax_label}"
    cbar.set_ticklabels(["0", vmax_label])
    cbar.ax.tick_params(length=0, pad=1)
    cbar.outline.set_linewidth(0.6)

    ax_observed.pcolormesh(
        x_edges,
        np.asarray([0, 1]),
        observed.to_numpy(dtype=float)[None, :],
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    ax_observed.set_yticks([0.5])
    ax_observed.set_yticklabels(["Active\nFires"])
    ax_observed.set_title("Observed (MODIS)", loc="left", fontweight="bold", pad=3)
    ax_observed.text(
        0.995,
        1.28,
        f"Peak Fire Count: {compact_count_label(observed.max())}",
        transform=ax_observed.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
    )

    x_centers = mdates.date2num(dates.to_pydatetime())
    sizes = np.zeros(len(area), dtype=float)
    if peak_area > 0:
        sizes = 14.0 + 420.0 * np.sqrt(area.to_numpy(dtype=float) / peak_area)
    positive_area = area.to_numpy(dtype=float) > 0
    if np.any(positive_area):
        ax_area.scatter(
            x_centers[positive_area],
            np.zeros(int(positive_area.sum())),
            s=sizes[positive_area],
            color="#064b5a",
            edgecolors="#032f38",
            linewidths=0.4,
            alpha=0.95,
            rasterized=True,
        )
    ax_area.set_yticks([0])
    ax_area.set_yticklabels([burned_area_label.replace(" ", "\n")])
    ax_area.text(
        0.995,
        1.18,
        f"Peak {burned_area_label}: {peak_area:.0f} Kha",
        transform=ax_area.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
    )
    ax_area.set_ylim(-0.65, 0.65)

    for ax in [ax_forecast, ax_observed, ax_area]:
        ax.grid(axis="x", color="#d8d8d8", linewidth=0.45)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.7)

    ax_area.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, math.ceil(len(dates) / 6))))
    ax_area.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_area.set_xlim(x_edges[0], x_edges[-1])
    for label in ax_area.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")
    plt.setp(ax_forecast.get_xticklabels(), visible=False)
    plt.setp(ax_observed.get_xticklabels(), visible=False)

    note_parts = [region.display_name, model_label]
    if source_label:
        note_parts.insert(0, source_label)
    if used_lead_column is None:
        note_parts.append("no explicit lead-time column in prediction file")
    fig.text(0.01, 0.01, " | ".join(note_parts), fontsize=8, color="#555555", ha="left", va="bottom")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.12)

    for subdir in ["png", "pdf"]:
        (output_dir / "plots" / subdir).mkdir(parents=True, exist_ok=True)
    prefix = f"{safe_slug(source_label)}_" if source_label else ""
    stem = f"{prefix}fire_period_timeline_{period_slug(selection)}"
    written: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt == "png":
            out = output_dir / "plots" / "png" / f"{stem}.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
        elif fmt == "pdf":
            out = output_dir / "plots" / "pdf" / f"{stem}.pdf"
            fig.savefig(out, bbox_inches="tight")
        else:
            out = output_dir / f"{stem}.{fmt}"
            fig.savefig(out, dpi=dpi if fmt in {"jpg", "jpeg", "tif", "tiff"} else None, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return written


def cleanup_old_plot_files(output_dir: Path) -> None:
    for subdir in [output_dir / "plots" / "png", output_dir / "plots" / "pdf"]:
        if not subdir.exists():
            continue
        for pattern in ["*fire_period_timeline_*", "*probability_overlay_*"]:
            for path in subdir.glob(pattern):
                if path.is_file():
                    path.unlink()


def write_markdown_summary(
    output_dir: Path,
    selected: pd.DataFrame,
    *,
    source: str,
    source_label: str | None,
    excluded_months: Sequence[int],
    generate_overlay_maps: bool,
) -> None:
    excluded = ", ".join(str(month) for month in excluded_months) or "none"
    lines = [
        "# Fire Period Timelines",
        "",
        "Selected non-winter test-set periods and plotted forecast/observed daily fire-positive counts.",
        "",
        "When enabled, the timeline selector keeps the best metric period whose observed activity is centered within the 28-day window.",
        "",
        f"Prediction source: `{source}`.",
        "",
        f"Feature source: `{source_label or 'default'}`.",
        "",
        f"Excluded months: `{excluded}`.",
        "",
        f"Matching overlay maps: `{'enabled' if generate_overlay_maps else 'disabled'}`.",
        "",
        "| Region | Rank | Model | Period | Days | Fire locations | AP | Brier |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in selected.iterrows():
        period = f"{row['period_start']} to {row['period_end']}"
        lines.append(
            "| {region} | {rank} | {model} | {period} | {days} | {locations} | {ap} | {brier} |".format(
                region=row["region_display"],
                rank=int(row.get("period_rank", 1)),
                model=row["model_name"],
                period=period,
                days=int(row.get("window_days", 0)),
                locations=int(row.get("observed_positive_locations", 0)),
                ap=metric_formatter(row.get("average_precision")),
                brier=metric_formatter(row.get("weighted_brier_score")),
            )
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fire_period_timelines(config: FirePeriodTimelineConfig) -> dict[str, object]:
    output_dir = config.output_dir or default_output_dir(config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    if config.window_days < 1:
        raise ValueError("fire_period_timeline_window_days must be >= 1")
    if config.top_periods < 1:
        raise ValueError("fire_period_timeline_top_periods must be >= 1")

    if config.overlay_grid_resolution is None:
        target_config = load_yaml(config.target_config)
        grid_resolution = float(target_config.get("spatial_coarseness", 0.1))
    else:
        grid_resolution = float(config.overlay_grid_resolution)

    only = set(config.regions) if config.regions else None
    regions = load_regions(config.regions_file, include_global=config.include_global, only=only)
    prediction_files = find_prediction_files_for_model(config.results_dir, config.model, config.source)

    metric_frames: list[pd.DataFrame] = []
    lead_columns: dict[str, str | None] = {}
    for path in prediction_files:
        print(f"Scoring fire-period timeline candidates from {path}")
        frame, lead_col = read_prediction_columns_with_lead(path, config.prob_col, config.lead_column)
        lead_columns[str(path)] = lead_col
        metrics = period_metrics_for_model(
            frame,
            prediction_path=path,
            regions=regions,
            window_days=int(config.window_days),
            require_full_periods=not config.allow_partial_periods,
            spatial_tolerance_degrees=config.spatial_tolerance_degrees,
        )
        metrics = add_temporal_activity_metrics(metrics, frame, regions=regions)
        metric_frames.append(metrics)
        del frame

    period_metrics = pd.concat(metric_frames, ignore_index=True)
    period_metrics["season_filter"] = np.where(
        season_mask(period_metrics, config.excluded_months),
        "kept_nonwinter",
        "excluded_winter",
    )
    selected_pool = period_metrics[period_metrics["season_filter"].eq("kept_nonwinter")].copy()
    if selected_pool.empty:
        raise ValueError(
            "No fire-period timeline candidates remained after excluding months "
            f"{list(config.excluded_months)}."
        )

    wide_dir = output_dir / "artifacts" / "wide_tables"
    write_wide_jsonl(wide_dir / "period_probability_metrics.jsonl.gz", period_metrics)
    metrics_path = output_dir / "tables" / "period_probability_metrics.csv"
    compact_overlay_table(
        selected_pool.sort_values(config.selection_metric, ascending=False).head(200),
        [
            "region_display",
            "model_name",
            "period_start",
            "period_end",
            "window_days",
            "observed_positive_locations",
            "average_precision",
            "weighted_brier_score",
            "observed_peak_day_index",
            "first_quarter_fire_fraction",
            "middle_half_fire_fraction",
            "centered_activity_score",
        ],
    ).to_csv(metrics_path, index=False)

    selected = select_centered_fire_periods(
        selected_pool,
        metric=config.selection_metric,
        min_wildfires=config.min_wildfires,
        top_periods=config.top_periods,
        allow_overlapping_periods=config.allow_overlapping_periods,
        prefer_centered_activity=config.prefer_centered_activity,
        center_peak_min_fraction=config.center_peak_min_fraction,
        center_peak_max_fraction=config.center_peak_max_fraction,
        max_start_activity_fraction=config.max_start_activity_fraction,
        min_middle_activity_fraction=config.min_middle_activity_fraction,
    )
    selected["excluded_months"] = ",".join(str(month) for month in config.excluded_months)
    write_wide_jsonl(wide_dir / "selected_fire_periods.jsonl.gz", selected)
    selected_path = output_dir / "tables" / "selected_fire_periods.csv"
    compact_overlay_table(
        selected,
        [
            "region_display",
            "model_name",
            "period_rank",
            "period_start",
            "period_end",
            "window_days",
            "observed_positive_locations",
            "average_precision",
            "observed_peak_day_index",
            "first_quarter_fire_fraction",
            "middle_half_fire_fraction",
            "centered_activity_score",
        ],
    ).to_csv(selected_path, index=False)
    write_markdown_summary(
        output_dir,
        selected,
        source=config.source,
        source_label=config.source_label,
        excluded_months=config.excluded_months,
        generate_overlay_maps=config.generate_overlay_maps,
    )

    cleanup_old_plot_files(output_dir)
    world = load_world_boundaries(config.overlay_country_shapes)
    region_by_name = {region.name: region for region in regions}
    formats = [fmt.strip() for fmt in config.formats if fmt.strip()]
    written: list[Path] = []
    overlay_written: list[Path] = []
    for prediction_path, group in selected.groupby("prediction_path", sort=False):
        path = Path(str(prediction_path))
        frame, lead_col = read_prediction_columns_with_lead(path, config.prob_col, config.lead_column)
        for _, selection in group.iterrows():
            region = region_by_name[str(selection["region"])]
            written.extend(
                plot_fire_period_timeline(
                    frame,
                    selection=selection,
                    region=region,
                    output_dir=output_dir,
                    formats=formats,
                    dpi=config.dpi,
                    grid_resolution=grid_resolution,
                    source_label=config.source_label,
                    lead_column=lead_col,
                    max_lead_days=config.max_lead_days,
                    burned_area_label=config.burned_area_label,
                    count_colormap=config.count_colormap,
                    count_norm_gamma=config.count_norm_gamma,
                    count_vmax_percentile=config.count_vmax_percentile,
                )
            )
            if config.generate_overlay_maps:
                overlay_selection = overlay_selection_for_timeline(
                    selection,
                    window_days=config.overlay_window_days,
                    center_on_peak=config.overlay_center_on_observed_peak,
                )
                overlay_written.extend(
                    plot_selected_period(
                        frame,
                        selection=overlay_selection,
                        region=region,
                        output_dir=output_dir,
                        results_dir=config.results_dir,
                        formats=formats,
                        dpi=config.dpi,
                        map_summary=config.overlay_map_summary,
                        grid_resolution=grid_resolution,
                        interpolation_factor=config.overlay_interpolation_factor,
                        surface_source=config.overlay_surface_source,
                        dense_model_path=config.overlay_dense_model_path
                        or (config.results_dir / "shared_artifacts" / "models" / "catboost_full.cbm"),
                        dense_neural_model_path=config.overlay_dense_neural_model_path,
                        dense_neural_training_features=config.overlay_dense_neural_training_features,
                        dense_neural_batch_size=config.overlay_dense_neural_batch_size,
                        dense_neural_device=config.overlay_dense_neural_device,
                        feature_config_path=config.feature_config,
                        overwrite_dense=config.overlay_overwrite_dense,
                        prior_correction=config.overlay_prior_correction,
                        train_prior=config.overlay_train_prior,
                        deploy_prior=config.overlay_deploy_prior,
                        colormap=config.overlay_colormap,
                        color_floor=config.overlay_color_floor,
                        color_vmax=config.overlay_color_vmax,
                        source_label=config.source_label,
                        verbose_feature_generation=config.overlay_verbose_feature_generation,
                        world=world,
                    )
                )
        del frame

    manifest = {
        "output_dir": str(output_dir),
        "period_metrics": str(metrics_path),
        "selected_periods": str(selected_path),
        "timeline_plots": [str(path) for path in written],
        "overlay_plots": [str(path) for path in overlay_written],
        "plot_count": len(written) + len(overlay_written),
        "model": config.model,
        "source": config.source,
        "source_label": config.source_label,
        "selection_metric": config.selection_metric,
        "window_days": int(config.window_days),
        "excluded_months": list(config.excluded_months),
        "prefer_centered_activity": bool(config.prefer_centered_activity),
        "center_peak_min_fraction": float(config.center_peak_min_fraction),
        "center_peak_max_fraction": float(config.center_peak_max_fraction),
        "max_start_activity_fraction": float(config.max_start_activity_fraction),
        "min_middle_activity_fraction": float(config.min_middle_activity_fraction),
        "count_colormap": config.count_colormap,
        "count_norm_gamma": float(config.count_norm_gamma),
        "count_vmax_percentile": float(config.count_vmax_percentile),
        "overlay_window_days": config.overlay_window_days,
        "overlay_center_on_observed_peak": bool(config.overlay_center_on_observed_peak),
        "lead_columns": lead_columns,
        "regions": [region.name for region in regions],
        "burned_area_note": "Burnt-area panel is an observed fire-positive grid-cell area proxy, not satellite burned-area retrieval.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote fire-period timelines: {len(written)} timeline plot(s), {len(overlay_written)} overlay map(s)")
    print(f"Wrote selected periods: {selected_path}")
    return manifest
