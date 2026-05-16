from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pandas as pd

from .config import EvaluationConfig
from .full_grid_evaluation import evaluate_model_full_grid_calibrated, safe_slug
from .tabular import CatBoostClassifier, load_regions, load_yaml, make_catboost_raw_predict_fn


SOURCE_MODELS = {
    "seas5_to_seas5": {
        "label": "SEAS5/ECMWF -> SEAS5/ECMWF",
        "train_source": "SEAS5/ECMWF",
        "evaluation_source": "SEAS5/ECMWF deployment grid",
        "stem": "source_seas5_to_seas5_available",
    },
    "era5_to_seas5": {
        "label": "ERA5 -> SEAS5/ECMWF",
        "train_source": "ERA5",
        "evaluation_source": "SEAS5/ECMWF deployment grid",
        "stem": "source_era5_to_seas5_available",
    },
    "mixed_to_seas5": {
        "label": "ERA5 + SEAS5/ECMWF -> SEAS5/ECMWF",
        "train_source": "ERA5 + SEAS5/ECMWF",
        "evaluation_source": "SEAS5/ECMWF deployment grid",
        "stem": "source_era5_plus_seas5_to_seas5_available",
    },
    "era5_to_seas5_no_tp": {
        "label": "ERA5 -> SEAS5/ECMWF (no tp)",
        "train_source": "ERA5 (no tp features)",
        "evaluation_source": "SEAS5/ECMWF deployment grid",
        "stem": "source_era5_to_seas5_available_no_tp",
    },
}


PRIMARY_TABLE_MAP = {
    "model_comparison": "primary_full_grid_calibrated_model_comparison.jsonl.gz",
    "model_comparison_by_year": "primary_full_grid_calibrated_model_comparison_by_year.jsonl.gz",
    "probability_metrics": "primary_full_grid_calibrated_probability_metrics.jsonl.gz",
    "monthly_count_calibration": "primary_full_grid_calibrated_monthly_count_calibration.jsonl.gz",
    "country_count_calibration": "primary_full_grid_calibrated_country_count_calibration.jsonl.gz",
    "region_count_calibration": "primary_full_grid_calibrated_region_count_calibration.jsonl.gz",
    "reliability_bins": "primary_full_grid_calibrated_reliability_bins.jsonl.gz",
    "prevalence_audit": "primary_full_grid_calibrated_prevalence_audit.jsonl.gz",
    "risk_concentration": "primary_full_grid_calibrated_risk_concentration.jsonl.gz",
    "count_correction": "primary_full_grid_calibrated_count_correction.jsonl.gz",
    "spatial_scale_evaluation": "primary_full_grid_calibrated_spatial_scale_evaluation.jsonl.gz",
    "experiment_registry": "primary_full_grid_calibrated_experiment_registry.jsonl.gz",
}


@dataclass(frozen=True)
class SourceModelSpec:
    key: str
    label: str
    train_source: str
    evaluation_source: str
    model_path: Path
    feature_path: Path


def default_args(**overrides: Any) -> SimpleNamespace:
    data = {
        "config": Path("configs/revision_evaluation_all_models_with_nns_no_latlon.yaml"),
        "source_artifact_dir": Path(
            "results/revision_experiments_complete_with_nns_no_latlon/shared_artifacts/era5_source_comparison"
        ),
        "only_source": None,
        "skip_organizer": False,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "shared_artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "source_full_grid_evaluation.log", encoding="utf-8"),
        ],
    )


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, orient="records", lines=True, compression="gzip")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def hydrate_primary_tables(output_dir: Path) -> None:
    """Restore primary CSVs from organized raw JSONL so evaluator upserts safely."""

    raw_dir = output_dir / "shared_artifacts" / "raw_tables_jsonl"
    primary = output_dir / "primary_full_grid_calibrated"
    primary.mkdir(parents=True, exist_ok=True)
    for table_name, raw_name in PRIMARY_TABLE_MAP.items():
        raw = raw_dir / raw_name
        dest = primary / f"{table_name}.csv"
        if dest.exists() or not raw.exists():
            continue
        df = read_jsonl(raw)
        df.to_csv(dest, index=False)
        logging.info("Hydrated %s from %s rows=%d", dest, raw, len(df))


def load_source_specs(source_artifact_dir: Path, selected: Sequence[str] | None) -> list[SourceModelSpec]:
    keys = list(selected or SOURCE_MODELS)
    specs: list[SourceModelSpec] = []
    model_dir = source_artifact_dir / "models"
    for key in keys:
        meta = SOURCE_MODELS[key]
        stem = meta["stem"]
        model_path = model_dir / f"{stem}.cbm"
        feature_path = model_dir / f"{stem}_features.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing source model: {model_path}")
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing source feature metadata: {feature_path}")
        specs.append(
            SourceModelSpec(
                key=key,
                label=meta["label"],
                train_source=meta["train_source"],
                evaluation_source=meta["evaluation_source"],
                model_path=model_path,
                feature_path=feature_path,
            )
        )
    return specs


def load_catboost_model(path: Path) -> Any:
    if CatBoostClassifier is None:
        raise RuntimeError("catboost is not available in this environment.")
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def source_rows_from_primary(output_dir: Path, specs: Sequence[SourceModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = output_dir / "primary_full_grid_calibrated"
    comparison = pd.read_csv(primary / "model_comparison.csv")
    by_year = pd.read_csv(primary / "model_comparison_by_year.csv")
    labels = {spec.label for spec in specs}
    source_meta = {
        spec.label: {
            "train_source": spec.train_source,
            "evaluation_source": spec.evaluation_source,
            "source_model_key": spec.key,
        }
        for spec in specs
    }

    overall = comparison[
        comparison["model_name"].isin(labels) & comparison["region"].astype(str).eq("global")
    ].copy()
    yearly = by_year[
        by_year["model_name"].isin(labels) & by_year["region"].astype(str).eq("global")
    ].copy()
    for frame in [overall, yearly]:
        frame["train_source"] = frame["model_name"].map(lambda value: source_meta[value]["train_source"])
        frame["evaluation_source"] = frame["model_name"].map(lambda value: source_meta[value]["evaluation_source"])
        frame["source_model_key"] = frame["model_name"].map(lambda value: source_meta[value]["source_model_key"])
        frame["status"] = "completed_primary_full_grid_calibrated"
    return overall, yearly


def write_source_tables(output_dir: Path, overall: pd.DataFrame, yearly: pd.DataFrame) -> None:
    cols = [
        "source_model_key",
        "model_name",
        "train_source",
        "evaluation_source",
        "status",
        "test_period",
        "support",
        "weighted_support",
        "observed_prevalence",
        "mean_calibrated_predicted_probability",
        "average_precision",
        "roc_auc",
        "max_f1",
        "precision_at_max_f1",
        "recall_at_max_f1",
        "threshold_at_max_f1",
        "predicted_positive_grid_cells_at_max_f1",
        "weighted_brier_score",
        "weighted_logloss",
        "calibration_intercept",
        "calibration_slope",
        "expected_observed_count_ratio",
        "expected_fire_positive_grid_cells",
        "observed_fire_positive_grid_cells",
        "daily_expected_observed_count_mae",
    ]
    for col in cols:
        if col not in overall.columns:
            overall[col] = pd.NA
    overall[cols].to_csv(output_dir / "input_source_full_grid_calibrated.csv", index=False)

    year_cols = [
        "source_model_key",
        "model_name",
        "train_source",
        "evaluation_source",
        "status",
        "period",
        "support",
        "weighted_support",
        "observed_prevalence",
        "mean_calibrated_predicted_probability",
        "average_precision",
        "roc_auc",
        "max_f1",
        "precision_at_max_f1",
        "recall_at_max_f1",
        "threshold_at_max_f1",
        "weighted_brier_score",
        "weighted_logloss",
        "expected_observed_count_ratio",
    ]
    for col in year_cols:
        if col not in yearly.columns:
            yearly[col] = pd.NA
    yearly[year_cols].to_csv(output_dir / "input_source_full_grid_calibrated_by_year.csv", index=False)


def run_organizer(config: EvaluationConfig) -> None:
    from .experiment_library import organize_results

    organize_results(config.output_dir)


def run(args: Any) -> None:
    config = EvaluationConfig.from_yaml(args.config)
    config.overwrite_output_dir = False
    config.run_full_grid_evaluation = True
    setup_logging(config.output_dir)
    hydrate_primary_tables(config.output_dir)

    feature_config = load_yaml(config.feature_config)
    target_config = load_yaml(config.target_config)
    regions = load_regions(config.regions_file)
    specs = load_source_specs(args.source_artifact_dir, args.only_source)

    for spec in specs:
        meta = json.loads(spec.feature_path.read_text(encoding="utf-8"))
        feature_columns = list(meta["features"])
        categorical = list(meta.get("categorical_features", []))
        model = load_catboost_model(spec.model_path)
        logging.info("Running calibrated full-grid evaluation for %s", spec.label)
        evaluate_model_full_grid_calibrated(
            model_name=spec.label,
            model_type="CatBoost source comparison",
            feature_columns=feature_columns,
            categorical_columns=categorical,
            config=config,
            output_dir=config.output_dir,
            predict_raw_fn=make_catboost_raw_predict_fn(model, categorical, feature_config),
            feature_config=config.feature_config,
            target_config=target_config,
            regions=regions,
            model_path=spec.model_path,
            feature_set=f"train_source={spec.train_source}; eval_source={spec.evaluation_source}",
        )

    overall, yearly = source_rows_from_primary(config.output_dir, specs)
    write_source_tables(config.output_dir, overall, yearly)
    write_json(
        config.output_dir / "shared_artifacts" / "era5_source_comparison" / "full_grid_calibrated_manifest.json",
        {
            "models": [spec.label for spec in specs],
            "evaluation_type": "primary_full_grid_calibrated",
            "calibration_method": config.calibration_method,
            "calibration_period": f"{config.calibration_start_date} to {config.calibration_end_date}",
            "test_period": f"{config.test_start_date} to {config.test_end_date}",
            "deployment_grid_countries": config.deployment_grid_countries,
            "full_grid_mode": config.full_grid_mode,
            "weighted_grid_sample_fraction": config.weighted_grid_sample_fraction,
        },
    )
    if not args.skip_organizer:
        run_organizer(config)


def main() -> int:
    run(default_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
