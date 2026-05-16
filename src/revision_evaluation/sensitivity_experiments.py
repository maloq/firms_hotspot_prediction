from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import EvaluationConfig
from .artifacts import ensure_dir, prune_empty_dirs, read_yaml, write_json, write_text
from .tabular import (
    CATBOOST_IMPORT_ERROR,
    DATE_COLUMN,
    DEFAULT_FEATURES_PATH,
    DEFAULT_IGNORED_FEATURES,
    LAT_COLUMN,
    LON_COLUMN,
    SEED,
    TARGET_COLUMN,
    CatBoostClassifier,
    Pool,
    Region,
    build_feature_sets,
    compute_metric_errors,
    feature_group,
    load_regions,
    markdown_table,
    model_feature_columns,
    normalize_cat_columns,
    positive_labels,
    validate_no_leakage_features,
    write_markdown_table,
)


TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_SEVEN_DAY_FEATURES_PATH = Path("data/saved_features/train_test_features_7d_all.parquet")


@dataclass(frozen=True)
class SensitivityConfig:
    output_dir: Path
    features_path: Path = DEFAULT_FEATURES_PATH
    seven_day_features_path: Path = DEFAULT_SEVEN_DAY_FEATURES_PATH
    feature_config: Path = Path("configs/features_config_30d.yaml")
    catboost_config: Path = Path("configs/catboost_train_config.yaml")
    regions_file: Path = Path("configs/regions_example.yaml")
    target_config: Path = Path("configs/target_config.yaml")
    catboost_iterations: int = 260
    catboost_task_type: str = "GPU"
    random_error_trials: int = 5
    random_error_sample_size: int = 50_000
    use_lat_lon_features: bool = True
    run_label_sensitivity: bool = True
    run_lead_time_sensitivity: bool = True
    seed: int = SEED


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(frame[DATE_COLUMN]):
        frame = frame.copy()
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    return frame


def split_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    years = pd.to_datetime(frame[DATE_COLUMN]).dt.year
    return {
        "train": years.between(2001, 2018).to_numpy(),
        "validation": years.between(2019, 2020).to_numpy(),
        "test": years.between(2021, 2025).to_numpy(),
    }


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float | None]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5, None
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5, None
    f1 = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    if not np.isfinite(f1).any():
        return 0.5, None
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def safe_metric(func: Any, *args: Any) -> float | None:
    try:
        value = float(func(*args))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def metric_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return {
            "support": 0,
            "positives": 0,
            "negatives": 0,
            "positive_rate": np.nan,
            "precision": None,
            "recall": None,
            "f1": None,
            "average_precision": None,
            "roc_auc": None,
            "brier_score": None,
            "threshold": threshold,
        }
    y_pred = (y_prob >= threshold).astype(int)
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "support": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int(len(y_true) - y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "predicted_positives": int(y_pred.sum()),
        "precision": safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y_true, y_pred),
        "recall": safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_true, y_pred),
        "f1": safe_metric(lambda a, b: f1_score(a, b, zero_division=0), y_true, y_pred),
        "average_precision": safe_metric(average_precision_score, y_true, y_prob) if has_both_classes else None,
        "roc_auc": safe_metric(roc_auc_score, y_true, y_prob) if has_both_classes else None,
        "brier_score": safe_metric(brier_score_loss, y_true, y_prob) if has_both_classes else None,
        "threshold": float(threshold),
    }


def evaluate_periods(
    *,
    experiment: str,
    model: str,
    feature_set: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    regions: list[Region],
    extra: dict[str, Any] | None = None,
    error_trials: int,
    error_sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extra = extra or {}
    rows: list[dict[str, Any]] = []
    by_period_rows: list[dict[str, Any]] = []
    years = pd.to_datetime(frame[DATE_COLUMN]).dt.year.to_numpy()

    def add(target: list[dict[str, Any]], region_name: str, display: str, period: str, mask: np.ndarray) -> None:
        mask_y = y_true[mask]
        mask_prob = y_prob[mask]
        metrics = metric_dict(mask_y, mask_prob, threshold)
        metrics.update(
            compute_metric_errors(
                mask_y,
                mask_prob,
                threshold,
                trials=error_trials,
                sample_size=error_sample_size,
                seed=seed + len(rows) + len(by_period_rows),
            )
        )
        target.append(
            {
                "experiment": experiment,
                "model": model,
                "feature_set": feature_set,
                "region": region_name,
                "region_display": display,
                "period": period,
                **metrics,
                **extra,
            }
        )

    all_mask = np.ones(len(frame), dtype=bool)
    add(rows, "global", "Global", "2021-2025", all_mask)
    for region in regions:
        add(rows, region.name, region.display_name, "2021-2025", region.mask(frame))

    period_specs = [(str(year), years == year) for year in TEST_YEARS]
    period_specs.extend(
        [
            ("2021-2023", (years >= 2021) & (years <= 2023)),
            ("2021-2025", (years >= 2021) & (years <= 2025)),
        ]
    )
    for period, period_mask in period_specs:
        if not period_mask.any():
            continue
        add(by_period_rows, "global", "Global", period, period_mask)
        for region in regions:
            add(by_period_rows, region.name, region.display_name, period, period_mask & region.mask(frame))
    return pd.DataFrame(rows), pd.DataFrame(by_period_rows)


def catboost_pool(frame: pd.DataFrame, y: np.ndarray | None, cat_features: list[str]) -> Pool:
    if Pool is None:
        raise RuntimeError(f"catboost import failed: {CATBOOST_IMPORT_ERROR}")
    cats = [col for col in cat_features if col in frame.columns]
    if y is None:
        return Pool(frame, cat_features=cats) if cats else Pool(frame)
    return Pool(frame, label=y, cat_features=cats) if cats else Pool(frame, label=y)


def fit_catboost(
    *,
    experiment_id: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    y_all: np.ndarray,
    masks: dict[str, np.ndarray],
    feature_config: dict[str, Any],
    output_dir: Path,
    settings: SensitivityConfig,
    train_row_positions: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    if CatBoostClassifier is None:
        raise RuntimeError(f"catboost import failed: {CATBOOST_IMPORT_ERROR}")
    validate_no_leakage_features(feature_columns)
    cat_features = [col for col in feature_config.get("cat_features", []) if col in feature_columns]
    numerical_cat = [col for col in feature_config.get("numerical_cat_features", []) if col in feature_columns]
    train_positions = np.flatnonzero(masks["train"]) if train_row_positions is None else train_row_positions
    val_positions = np.flatnonzero(masks["validation"])
    test_positions = np.flatnonzero(masks["test"])

    X_train = normalize_cat_columns(frame.iloc[train_positions][feature_columns], cat_features, numerical_cat)
    X_val = normalize_cat_columns(frame.iloc[val_positions][feature_columns], cat_features, numerical_cat)
    y_train = y_all[train_positions].astype(np.int8)
    y_val = y_all[val_positions].astype(np.int8)
    params = {
        "iterations": settings.catboost_iterations,
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 0.35,
        "min_data_in_leaf": 80,
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "class_weights": [1.0, 4.0],
        "random_seed": settings.seed,
        "random_strength": 1.0,
        "verbose": False,
        "allow_writing_files": False,
    }
    if settings.catboost_task_type:
        params["task_type"] = settings.catboost_task_type

    model = CatBoostClassifier(**params)
    try:
        model.fit(
            catboost_pool(X_train, y_train, cat_features),
            eval_set=catboost_pool(X_val, y_val, cat_features),
            use_best_model=True,
            early_stopping_rounds=100,
        )
    except Exception:
        if params.get("task_type") == "GPU":
            params.pop("task_type", None)
            model = CatBoostClassifier(**params)
            model.fit(
                catboost_pool(X_train, y_train, cat_features),
                eval_set=catboost_pool(X_val, y_val, cat_features),
                use_best_model=True,
                early_stopping_rounds=100,
            )
        else:
            raise

    val_prob = np.asarray(model.predict_proba(catboost_pool(X_val, None, cat_features)))[:, 1]
    threshold, val_f1 = choose_threshold(y_val, val_prob)
    test_frame = frame.iloc[test_positions]
    X_test = normalize_cat_columns(test_frame[feature_columns], cat_features, numerical_cat)
    test_prob = np.asarray(model.predict_proba(catboost_pool(X_test, None, cat_features)))[:, 1]

    model_dir = ensure_dir(output_dir / "models")
    model_path = model_dir / f"{experiment_id}.cbm"
    model.save_model(model_path)
    write_json(
        model_dir / f"{experiment_id}_features.json",
        {
            "features": feature_columns,
            "categorical_features": cat_features,
            "use_lat_lon_features": settings.use_lat_lon_features,
        },
    )
    diagnostics = {
        "model_path": str(model_path),
        "feature_count": len(feature_columns),
        "cat_features": cat_features,
        "best_iteration": model.get_best_iteration(),
        "validation_f1_at_threshold": val_f1,
        "threshold": threshold,
        "train_rows": int(len(train_positions)),
    }
    return diagnostics, test_prob, test_positions, threshold


def country_aliases(country: str) -> list[str]:
    mapping = {
        "Serbia": "Republic_of_Serbia",
        "Dem_Rep_Korea": "North_Korea",
        "Macedonia_Former_Yugoslav_Republic_of": "North_Macedonia",
    }
    aliases = [country, mapping.get(country, country)]
    aliases.append(country.replace("Macedonia_Former_Yugoslav_Republic_of", "North_Macedonia"))
    aliases.append(country.replace("Dem_Rep_Korea", "North_Korea"))
    aliases.append(country.replace("Serbia", "Republic_of_Serbia"))
    return list(dict.fromkeys(aliases))


def read_modis_for_countries(modis_dir: Path, countries: list[str], start_year: int, end_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    usecols = ["latitude", "longitude", "brightness", "confidence", "acq_date"]
    for year in range(start_year, end_year + 1):
        year_dir = modis_dir / str(year)
        if not year_dir.exists():
            continue
        for country in countries:
            for alias in country_aliases(country):
                path = year_dir / f"modis_{year}_{alias}.csv"
                if not path.exists():
                    continue
                part = pd.read_csv(path, usecols=lambda column: column in usecols)
                if {"latitude", "longitude", "brightness", "confidence", "acq_date"}.issubset(part.columns):
                    part["country"] = country
                    frames.append(part)
                break
    if not frames:
        return pd.DataFrame(columns=usecols + ["country"])
    out = pd.concat(frames, ignore_index=True)
    out["acq_date"] = pd.to_datetime(out["acq_date"], errors="coerce").dt.date
    return out.dropna(subset=["acq_date", "latitude", "longitude"])


def apply_target_thresholds(frame: pd.DataFrame, *, strict: bool, target_config: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("No raw MODIS rows were available for label reconstruction.")
    if strict:
        low_brightness, low_conf = 400.0, 80.0
        high_brightness, high_conf = 380.0, 80.0
    else:
        low_brightness = float(target_config.get("brightness_threshold", 380))
        low_conf = float(target_config.get("confidence_threshold", 0.85))
        high_brightness = float(target_config.get("brightness_threshold_high_lat", 360))
        high_conf = float(target_config.get("confidence_threshold_high_lat", 0.70))
    high_lat = frame["latitude"].abs() > 58
    mask = ((~high_lat) & (frame["brightness"] > low_brightness) & (frame["confidence"] > low_conf)) | (
        high_lat & (frame["brightness"] > high_brightness) & (frame["confidence"] > high_conf)
    )
    return frame.loc[mask].copy()


def target_positive_keys(
    raw_modis: pd.DataFrame,
    *,
    strict: bool,
    output_dir: Path,
    target_config: dict[str, Any],
    name: str,
) -> pd.DataFrame:
    frame = apply_target_thresholds(raw_modis, strict=strict, target_config=target_config)
    try:
        from src.target_generation.stationary_points import drop_stationary_points

        frame, _ = drop_stationary_points(
            frame,
            stationary_dir=Path(target_config.get("stationary_points_dir", "data/modis_stationary_points")),
            country_col="country",
            lat_col="latitude",
            lon_col="longitude",
        )
    except Exception:
        pass
    frame["datetime"] = pd.to_datetime(frame["acq_date"])
    frame["lat_key"] = np.rint(frame["latitude"].astype(float) * 10).astype(np.int32)
    frame["lon_key"] = np.rint(frame["longitude"].astype(float) * 10).astype(np.int32)
    frame["date_key"] = frame["datetime"].values.astype("datetime64[D]").astype("int64")
    keys = frame.groupby(["date_key", "lat_key", "lon_key"], observed=True).size().reset_index(name="raw_detection_count")
    cache_dir = ensure_dir(output_dir / "target_caches")
    keys.to_parquet(cache_dir / f"{name}_positive_keys.parquet", index=False)
    return keys


def relabel_from_keys(frame: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    base = pd.DataFrame(
        {
            "row_id": np.arange(len(frame), dtype=np.int64),
            "date_key": pd.to_datetime(frame[DATE_COLUMN]).values.astype("datetime64[D]").astype("int64"),
            "lat_key": np.rint(pd.to_numeric(frame[LAT_COLUMN], errors="coerce").to_numpy() * 10).astype(np.int32),
            "lon_key": np.rint(pd.to_numeric(frame[LON_COLUMN], errors="coerce").to_numpy() * 10).astype(np.int32),
        }
    )
    matched = base.merge(
        keys[["date_key", "lat_key", "lon_key"]].drop_duplicates(),
        on=["date_key", "lat_key", "lon_key"],
        how="left",
        indicator=True,
    )
    return (matched["_merge"].to_numpy() == "both").astype(np.int8)


def sample_negative_ratio(y: np.ndarray, masks: dict[str, np.ndarray], ratio: float, seed: int) -> np.ndarray:
    train_pos = np.flatnonzero(masks["train"] & (y == 1))
    train_neg = np.flatnonzero(masks["train"] & (y == 0))
    if len(train_pos) == 0 or len(train_neg) == 0:
        raise ValueError("Cannot build negative-ratio sample without both positive and negative training rows.")
    rng = np.random.default_rng(seed)
    n_neg = min(len(train_neg), int(len(train_pos) * ratio))
    sampled_neg = rng.choice(train_neg, size=n_neg, replace=False)
    return np.sort(np.concatenate([train_pos, sampled_neg]))


def add_failure(failures: list[dict[str, Any]], experiment: str, exc: Exception, next_action: str) -> None:
    failures.append(
        {
            "experiment": experiment,
            "reason": str(exc),
            "attempted_fixes": "Sensitivity stage continued with remaining experiments.",
            "affects_paper_claims": "Limited to this sensitivity table.",
            "next_action": next_action,
        }
    )


def write_tables(output_dir: Path, stem: str, title: str, frame: pd.DataFrame, by_year: pd.DataFrame, caption: str) -> None:
    frame.to_csv(output_dir / f"{stem}.csv", index=False)
    by_year.to_csv(output_dir / f"{stem}_by_year.csv", index=False)
    write_markdown_table(output_dir / f"{stem}.md", title, frame)


def run_label_sensitivity(
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    regions: list[Region],
    feature_config: dict[str, Any],
    target_config: dict[str, Any],
    feature_sets: dict[str, dict[str, Any]],
    settings: SensitivityConfig,
    failures: list[dict[str, Any]],
) -> None:
    log_lines = ["# Label Sensitivity Build Log", ""]
    modis_dir = Path(feature_config.get("modis_data_path", "/home/ids/vmorozov/data/modis"))
    countries = list(feature_config.get("prediction_countries") or feature_config.get("modis_countries") or [])
    raw_modis = read_modis_for_countries(modis_dir, countries, 2001, 2025)
    log_lines.append(f"- Raw MODIS rows loaded: {len(raw_modis):,}")
    log_lines.append(f"- Countries requested: {len(countries)}")

    y_main = positive_labels(frame[TARGET_COLUMN])
    variants: list[dict[str, Any]] = [
        {
            "experiment_id": "main_current_labels",
            "label": "Main/current labels",
            "y": y_main,
            "features": feature_sets["full"]["columns"],
            "notes": "Existing feature-matrix labels.",
        },
        {
            "experiment_id": "main_no_historical_fire_features",
            "label": "Main labels, no historical-fire features",
            "y": y_main,
            "features": feature_sets["no_history"]["columns"],
            "notes": "Historical-fire context removed when such columns exist.",
        },
    ]
    try:
        variants.append(
            {
                "experiment_id": "alternative_negative_ratio",
                "label": "Alternative negative ratio",
                "y": y_main,
                "features": feature_sets["full"]["columns"],
                "train_positions": sample_negative_ratio(y_main, masks, 3.0, settings.seed),
                "notes": "Training negatives downsampled to 3:1 negatives:positives; validation/test unchanged.",
            }
        )
    except Exception as exc:
        add_failure(failures, "Alternative negative ratio", exc, "Inspect training split class balance.")
    for experiment_id, label, strict in [
        ("no_morphological_expansion", "No morphological expansion/dilation", False),
        ("stricter_modis_thresholds", "Stricter MODIS thresholds", True),
    ]:
        try:
            keys = target_positive_keys(
                raw_modis,
                strict=strict,
                output_dir=settings.output_dir,
                target_config=target_config,
                name=experiment_id,
            )
            y_variant = relabel_from_keys(frame, keys)
            variants.append(
                {
                    "experiment_id": experiment_id,
                    "label": label,
                    "y": y_variant,
                    "features": feature_sets["full"]["columns"],
                    "notes": "Raw MODIS positives matched to saved grid/date rows.",
                }
            )
            log_lines.append(f"- {label}: positive keys={len(keys):,}; aligned positives={int(y_variant.sum()):,}")
        except Exception as exc:
            add_failure(failures, label, exc, "Inspect raw MODIS schema, thresholds, and country aliases.")

    rows_all: list[pd.DataFrame] = []
    rows_year: list[pd.DataFrame] = []
    for spec in variants:
        try:
            diagnostics, prob, test_positions, threshold = fit_catboost(
                experiment_id=f"label_{spec['experiment_id']}",
                frame=frame,
                feature_columns=spec["features"],
                y_all=spec["y"],
                masks=masks,
                feature_config=feature_config,
                output_dir=settings.output_dir,
                settings=settings,
                train_row_positions=spec.get("train_positions"),
            )
            test_frame = frame.iloc[test_positions]
            rows, by_year = evaluate_periods(
                experiment=spec["label"],
                model="CatBoost",
                feature_set="full features" if spec["features"] == feature_sets["full"]["columns"] else "no historical-fire/proximity features",
                frame=test_frame,
                y_true=spec["y"][test_positions],
                y_prob=prob,
                threshold=threshold,
                regions=regions,
                extra={"notes": spec["notes"], "validation_threshold": threshold, "train_rows": diagnostics["train_rows"]},
                error_trials=settings.random_error_trials,
                error_sample_size=settings.random_error_sample_size,
                seed=settings.seed,
            )
            rows_all.append(rows)
            rows_year.append(by_year)
        except Exception as exc:
            add_failure(failures, spec["label"], exc, "Rerun this label variant with CPU CatBoost or inspect target labels.")

    label_df = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame()
    label_year = pd.concat(rows_year, ignore_index=True) if rows_year else pd.DataFrame()
    write_tables(settings.output_dir, "label_sensitivity", "Label-Construction Sensitivity", label_df, label_year, "Label-construction sensitivity.")
    write_text(settings.output_dir / "target_caches" / "target_variant_build_log.md", "\n".join(log_lines))


def lead_feature_subset(columns: list[str], lead_days: int) -> list[str]:
    if lead_days == 30:
        return columns
    keep: list[str] = []
    for col in columns:
        if feature_group(col) != "weather_history":
            keep.append(col)
            continue
        name = col.lower()
        if any(f"_{window}" in name for window in [7, 14] if window <= lead_days):
            keep.append(col)
    return keep


def run_lead_time_sensitivity(
    regions: list[Region],
    feature_config: dict[str, Any],
    catboost_config: dict[str, Any],
    settings: SensitivityConfig,
    failures: list[dict[str, Any]],
) -> None:
    rows_all: list[pd.DataFrame] = []
    rows_year: list[pd.DataFrame] = []
    matrix_log = ["# Lead-Time Feature Matrix Log", ""]
    datasets = [
        (7, settings.seven_day_features_path),
        (14, settings.features_path),
        (30, settings.features_path),
    ]
    ignored = (
        catboost_config.get("catboost_train", {})
        .get("features", {})
        .get("ignored", DEFAULT_IGNORED_FEATURES)
        if isinstance(catboost_config.get("catboost_train"), dict)
        else DEFAULT_IGNORED_FEATURES
    )
    for lead_days, path in datasets:
        if not path.exists():
            add_failure(failures, f"{lead_days}-day lead-time sensitivity", FileNotFoundError(path), "Regenerate the lead-specific feature matrix.")
            continue
        try:
            frame = ensure_datetime(pd.read_parquet(path))
            masks = split_masks(frame)
            all_features = model_feature_columns(
                frame,
                ignored,
                use_lat_lon_features=settings.use_lat_lon_features,
            )
            features = lead_feature_subset(all_features, lead_days)
            matrix_log.append(f"- {lead_days}-day matrix: `{path}`, rows={len(frame):,}, features_used={len(features):,}.")
            diagnostics, prob, test_positions, threshold = fit_catboost(
                experiment_id=f"lead_time_{lead_days}d",
                frame=frame,
                feature_columns=features,
                y_all=positive_labels(frame[TARGET_COLUMN]),
                masks=masks,
                feature_config=feature_config,
                output_dir=settings.output_dir,
                settings=settings,
            )
            test_frame = frame.iloc[test_positions]
            rows, by_year = evaluate_periods(
                experiment=f"{lead_days}-day horizon",
                model="CatBoost",
                feature_set=f"{lead_days}-day lead metadata/features",
                frame=test_frame,
                y_true=positive_labels(test_frame[TARGET_COLUMN]),
                y_prob=prob,
                threshold=threshold,
                regions=regions,
                extra={
                    "lead_time_days": lead_days,
                    "validation_threshold": threshold,
                    "matrix_path": str(path),
                    "train_rows": diagnostics["train_rows"],
                },
                error_trials=settings.random_error_trials,
                error_sample_size=settings.random_error_sample_size,
                seed=settings.seed,
            )
            rows_all.append(rows)
            rows_year.append(by_year)
        except Exception as exc:
            add_failure(failures, f"{lead_days}-day lead-time sensitivity", exc, "Inspect feature availability and CatBoost logs.")

    lead_df = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame()
    lead_year = pd.concat(rows_year, ignore_index=True) if rows_year else pd.DataFrame()
    write_tables(settings.output_dir, "lead_time_sensitivity", "Lead-Time Sensitivity", lead_df, lead_year, "Lead-time sensitivity.")
    write_text(settings.output_dir / "lead_time_feature_build_log.md", "\n".join(matrix_log))
    plot_metric_lines(lead_year, settings.output_dir / "plots/lead_time_pr_auc", "lead_time_days", "average_precision", "Lead-Time PR-AUC")
    plot_metric_lines(lead_year, settings.output_dir / "plots/lead_time_f1", "lead_time_days", "f1", "Lead-Time F1")


def plot_metric_lines(frame: pd.DataFrame, base: Path, x_col: str, metric_col: str, title: str) -> None:
    if frame.empty or metric_col not in frame.columns or x_col not in frame.columns:
        return
    plot_df = frame[(frame["region"].eq("global")) & (frame["period"].isin([str(year) for year in TEST_YEARS]))].copy()
    plot_df = plot_df.dropna(subset=[metric_col]).sort_values([x_col, "period"])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, group in plot_df.groupby(x_col):
        ax.plot(group["period"].astype(str), group[metric_col].astype(float), marker="o", label=str(key))
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(metric_col)
    ax.legend(title=x_col)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ensure_dir(base.parent)
    fig.savefig(base.with_suffix(".png"), dpi=220)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def write_sensitivity_failures(output_dir: Path, failures: list[dict[str, Any]]) -> None:
    lines = ["# Sensitivity Failures And Limitations", ""]
    if not failures:
        lines.append("No failed label or lead-time sensitivity experiments remain.")
    for item in failures:
        lines.extend(
            [
                f"## {item.get('experiment')}",
                "",
                f"- Reason: {item.get('reason')}",
                f"- Attempted fixes: {item.get('attempted_fixes')}",
                f"- Affects paper claims: {item.get('affects_paper_claims')}",
                f"- Next action: {item.get('next_action')}",
                "",
            ]
        )
    write_text(output_dir / "sensitivity_failures.md", "\n".join(lines))


def run_sensitivity_experiments(settings: SensitivityConfig) -> dict[str, Any]:
    started = time.time()
    set_seeds(settings.seed)
    ensure_dir(settings.output_dir)
    feature_config = read_yaml(settings.feature_config)
    target_config = read_yaml(settings.target_config)
    catboost_config = read_yaml(settings.catboost_config)
    regions = load_regions(settings.regions_file)
    failures: list[dict[str, Any]] = []

    frame = ensure_datetime(pd.read_parquet(settings.features_path))
    masks = split_masks(frame)
    ignored = (
        catboost_config.get("catboost_train", {})
        .get("features", {})
        .get("ignored", DEFAULT_IGNORED_FEATURES)
        if isinstance(catboost_config.get("catboost_train"), dict)
        else DEFAULT_IGNORED_FEATURES
    )
    all_features = model_feature_columns(
        frame,
        ignored,
        use_lat_lon_features=settings.use_lat_lon_features,
    )
    feature_sets = build_feature_sets(all_features)

    if settings.run_label_sensitivity:
        run_label_sensitivity(frame, masks, regions, feature_config, target_config, feature_sets, settings, failures)
    if settings.run_lead_time_sensitivity:
        run_lead_time_sensitivity(regions, feature_config, catboost_config, settings, failures)

    write_sensitivity_failures(settings.output_dir, failures)
    manifest = {
        "completed_at": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(time.time() - started, 2),
        "features_path": settings.features_path,
        "use_lat_lon_features": settings.use_lat_lon_features,
        "experiments": {
            "label_sensitivity": settings.run_label_sensitivity,
            "lead_time_sensitivity": settings.run_lead_time_sensitivity,
        },
        "failures": failures,
    }
    write_json(settings.output_dir / "sensitivity_manifest.json", manifest)
    prune_empty_dirs(settings.output_dir)
    return manifest


def table_preview(frame: pd.DataFrame, max_rows: int = 8) -> str:
    return markdown_table(frame.head(max_rows)) if not frame.empty else "_No rows._"


def config_from_evaluation(config: EvaluationConfig) -> SensitivityConfig:
    return SensitivityConfig(
        output_dir=config.output_dir,
        features_path=config.features_path,
        seven_day_features_path=config.seven_day_features_path,
        feature_config=config.feature_config,
        target_config=config.target_config,
        catboost_config=config.catboost_config,
        regions_file=config.regions_file,
        catboost_iterations=config.sensitivity_catboost_iterations,
        catboost_task_type=config.catboost_task_type,
        random_error_trials=config.random_error_trials,
        random_error_sample_size=config.random_error_sample_size,
        use_lat_lon_features=config.use_lat_lon_features,
        run_label_sensitivity=config.run_label_sensitivity,
        run_lead_time_sensitivity=config.run_lead_time_sensitivity,
        seed=config.seed if config.seed is not None else SEED,
    )


def run_from_evaluation_config(config: EvaluationConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "commands_used.txt").open("a", encoding="utf-8") as fh:
        fh.write("config-driven revision_evaluation.sensitivity_experiments\n")
    return run_sensitivity_experiments(config_from_evaluation(config))
