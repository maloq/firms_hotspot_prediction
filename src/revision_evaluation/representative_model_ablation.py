from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor, SGDClassifier
from sklearn.model_selection import train_test_split

from .era5_source_comparison import (
    ECMWF_FEATURE_ROOT,
    ERA5_FEATURE_ROOT,
    best_neural_experiment_id,
    drop_climate_variable_features,
    era5_inventory,
    filter_full_era5_climate_coverage,
    load_available_seas5_frame,
    load_or_build_era5_climate_features,
    neural_prediction_path,
    replace_weather_features,
    split_masks_by_year,
    train_source_catboost,
    write_raw_table,
)
from .tabular import (
    DEFAULT_FEATURES_PATH,
    DEFAULT_IGNORED_FEATURES,
    DEFAULT_RESULTS_DIR,
    DATE_COLUMN,
    LAT_COLUMN,
    LON_COLUMN,
    SEED,
    TARGET_COLUMN,
    OrdinalTabularEncoder,
    build_feature_sets,
    catboost_categorical_features,
    choose_threshold_by_f1,
    evaluate_predictions,
    load_regions,
    load_yaml,
    model_feature_columns,
    normalize_cat_columns,
    positive_labels,
    predict_catboost,
    predict_linear_batches,
    predict_poisson_point_process_batches,
    save_predictions,
    stratified_sample_positions,
    validate_no_leakage_features,
    write_json,
)


EXPERIMENT_STEM = "representative_model_ablation_era5_to_ecmwf_no_tp"
EXPERIMENT_TYPE = "representative_model_ablation"


MODEL_SPECS = [
    {
        "model_key": "catboost_full",
        "model": "CatBoost",
        "model_group": "Gradient boosted trees",
        "notes": "Best tabular model family; trained on ERA5 features and tested on SEAS5/ECMWF.",
    },
    {
        "model_key": "random_forest",
        "model": "Random Forest",
        "model_group": "Bagged tree ensemble",
        "notes": "Tree ensemble baseline with different bias/variance behavior from CatBoost.",
    },
    {
        "model_key": "logistic_regression",
        "model": "Logistic Regression",
        "model_group": "Linear discriminative",
        "notes": "Linear SGD baseline with train-only ordinal categorical encoding.",
    },
    {
        "model_key": "poisson_point_process",
        "model": "Poisson Point-Process GLM",
        "model_group": "Statistical intensity model",
        "notes": "Counts-as-intensity baseline converted to event probability with 1-exp(-lambda).",
    },
    {
        "model_key": "best_neural",
        "model": "Best neural spatial TSN",
        "model_group": "Deep daily-spatial sequence",
        "notes": "Best neural checkpoint retrained on ERA5 dynamic climate inputs with tp removed.",
    },
]


def default_args(**overrides: Any) -> SimpleNamespace:
    data = {
        "features_path": DEFAULT_FEATURES_PATH,
        "feature_config": Path("configs/features_config_30d.yaml"),
        "catboost_config": Path("configs/catboost_train_config.yaml"),
        "regions_file": Path("configs/regions_example.yaml"),
        "output_dir": DEFAULT_RESULTS_DIR,
        "era5_feature_root": ERA5_FEATURE_ROOT,
        "ecmwf_feature_root": ECMWF_FEATURE_ROOT,
        "cache_dir": Path("data/saved_features/revision_evaluation/era5_source_comparison"),
        "train_start_year": 2001,
        "validation_start_year": 2019,
        "test_start_year": 2021,
        "test_end_year": None,
        "catboost_iterations": 260,
        "catboost_depth": 5,
        "catboost_learning_rate": 0.03,
        "catboost_task_type": "GPU",
        "catboost_verbose": 100,
        "rf_max_train_rows": 300_000,
        "linear_epochs": 4,
        "point_process_max_train_rows": 500_000,
        "point_process_alpha": 1e-4,
        "point_process_max_iter": 200,
        "prediction_batch_size": 100_000,
        "random_error_trials": 5,
        "random_error_sample_size": 50_000,
        "seed": SEED,
        "use_lat_lon_features": False,
        "force_rebuild_era5_features": False,
        "drop_climate_variables": ["tp"],
        "neural_model": "nn_global_full_spatial_tsn_no_tp",
    }
    data.update(overrides)
    data["drop_climate_variables"] = normalize_string_list(data.get("drop_climate_variables"))
    return SimpleNamespace(**data)


def normalize_string_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    return [str(item).strip() for item in value if str(item).strip()]


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "shared_artifacts" / "logs" / "representative_model_ablation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def _predict_tabular_batches(model: Any, encoder: OrdinalTabularEncoder, frame: pd.DataFrame, batch_size: int) -> np.ndarray:
    probs = np.empty(len(frame), dtype=np.float32)
    for slc, X_batch in encoder.transform_batches(frame, batch_size):
        probs[slc] = model.predict_proba(X_batch)[:, 1].astype(np.float32)
    return probs


def _evaluate_predictions(
    *,
    experiment_id: str,
    model: str,
    model_group: str,
    feature_set: str,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    validation_prob: np.ndarray,
    test_prob: np.ndarray,
    regions: list[Any],
    args: Any,
    artifact_root: Path,
    notes: str,
    feature_count: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_val = positive_labels(validation_frame[TARGET_COLUMN]) if TARGET_COLUMN in validation_frame.columns else validation_frame["target_binary"].to_numpy(dtype=np.int8)
    y_test = positive_labels(test_frame[TARGET_COLUMN]) if TARGET_COLUMN in test_frame.columns else test_frame["target_binary"].to_numpy(dtype=np.int8)
    threshold_info = choose_threshold_by_f1(y_val, validation_prob)
    threshold = float(threshold_info["threshold"])

    save_predictions(artifact_root, experiment_id, "validation", validation_frame, y_val, validation_prob, threshold)
    save_predictions(artifact_root, experiment_id, "test", test_frame, y_test, test_prob, threshold)

    rows: list[dict[str, Any]] = []
    for split_name, frame, y_true, prob in [
        ("validation", validation_frame, y_val, validation_prob),
        ("test", test_frame, y_test, test_prob),
    ]:
        rows.extend(
            evaluate_predictions(
                experiment_id,
                EXPERIMENT_TYPE,
                model,
                feature_set,
                split_name,
                frame,
                y_true,
                prob,
                threshold,
                regions,
                error_trials=args.random_error_trials,
                error_sample_size=args.random_error_sample_size,
                seed=args.seed,
            )
        )

    for row in rows:
        row["model_group"] = model_group
        row["status"] = "completed_available_domain"
        row["training_source"] = "ERA5"
        row["validation_source"] = "ERA5"
        row["test_source"] = "SEAS5/ECMWF"
        row["notes"] = notes
        row["validation_f1_at_threshold"] = threshold_info.get("validation_f1")

    registry_row = {
        "experiment_id": experiment_id,
        "model": model,
        "model_group": model_group,
        "status": "completed",
        "threshold": threshold,
        "validation_f1": threshold_info.get("validation_f1"),
        "feature_count": feature_count,
        "notes": notes,
    }
    return registry_row, rows


def train_logistic(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    feature_config: dict[str, Any],
    args: Any,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, Path | None]:
    categorical = catboost_categorical_features(train_frame, feature_columns, feature_config)
    encoder = OrdinalTabularEncoder(feature_columns, categorical, standardize=True).fit(train_frame[feature_columns])
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        learning_rate="optimal",
        class_weight={0: 1.0, 1: 4.0},
        random_state=args.seed,
        average=True,
    )
    y_train = positive_labels(train_frame[TARGET_COLUMN])
    classes = np.array([0, 1], dtype=np.int8)
    logging.info("Training representative logistic regression for %d epochs", args.linear_epochs)
    for epoch in range(int(args.linear_epochs)):
        order = np.arange(len(train_frame))
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(order)
        for start in range(0, len(order), int(args.prediction_batch_size)):
            positions = order[start : start + int(args.prediction_batch_size)]
            X_batch = encoder.transform(train_frame.iloc[positions][feature_columns])
            y_batch = y_train[positions]
            sample_weight = np.where(y_batch == 1, 4.0, 1.0)
            model.partial_fit(X_batch, y_batch, classes=classes, sample_weight=sample_weight)
        logging.info("Completed logistic epoch %d/%d", epoch + 1, args.linear_epochs)

    model_path: Path | None = None
    try:
        import joblib

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "representative_logistic_regression.joblib"
        joblib.dump({"model": model, "encoder": encoder}, model_path)
    except Exception:
        logging.exception("Could not save logistic regression model artifact.")

    return (
        predict_linear_batches(model, encoder, validation_frame[feature_columns], int(args.prediction_batch_size)),
        predict_linear_batches(model, encoder, test_frame[feature_columns], int(args.prediction_batch_size)),
        model_path,
    )


def train_poisson(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    feature_config: dict[str, Any],
    args: Any,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, Path | None]:
    categorical = catboost_categorical_features(train_frame, feature_columns, feature_config)
    encoder = OrdinalTabularEncoder(feature_columns, categorical, standardize=True).fit(train_frame[feature_columns])
    y_train_binary = positive_labels(train_frame[TARGET_COLUMN])
    y_train_count = pd.to_numeric(train_frame[TARGET_COLUMN], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    fit_positions = stratified_sample_positions(y_train_binary, int(args.point_process_max_train_rows), int(args.seed))
    X_fit = encoder.transform(train_frame.iloc[fit_positions][feature_columns])
    y_fit = y_train_count[fit_positions]
    sample_weight = np.where(y_fit > 0, 4.0, 1.0)
    model = PoissonRegressor(
        alpha=float(args.point_process_alpha),
        max_iter=int(args.point_process_max_iter),
        fit_intercept=True,
    )
    logging.info("Training representative Poisson GLM on %d rows", len(fit_positions))
    model.fit(X_fit, y_fit, sample_weight=sample_weight)

    model_path: Path | None = None
    try:
        import joblib

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "representative_poisson_point_process.joblib"
        joblib.dump({"model": model, "encoder": encoder}, model_path)
    except Exception:
        logging.exception("Could not save Poisson GLM model artifact.")

    return (
        predict_poisson_point_process_batches(model, encoder, validation_frame[feature_columns], int(args.prediction_batch_size)),
        predict_poisson_point_process_batches(model, encoder, test_frame[feature_columns], int(args.prediction_batch_size)),
        model_path,
    )


def train_random_forest(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    feature_config: dict[str, Any],
    args: Any,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, Path | None]:
    categorical = catboost_categorical_features(train_frame, feature_columns, feature_config)
    encoder = OrdinalTabularEncoder(feature_columns, categorical, standardize=False).fit(train_frame[feature_columns])
    y_train = positive_labels(train_frame[TARGET_COLUMN])
    if int(args.rf_max_train_rows) > 0 and len(train_frame) > int(args.rf_max_train_rows):
        idx = np.arange(len(train_frame))
        sampled_idx, _ = train_test_split(
            idx,
            train_size=int(args.rf_max_train_rows),
            stratify=y_train if len(np.unique(y_train)) > 1 else None,
            random_state=int(args.seed),
            shuffle=True,
        )
        sampled_idx = np.sort(sampled_idx)
    else:
        sampled_idx = np.arange(len(train_frame))
    X_fit = encoder.transform(train_frame.iloc[sampled_idx][feature_columns])
    y_fit = y_train[sampled_idx]
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=20,
        class_weight={0: 1.0, 1: 4.0},
        n_jobs=-1,
        random_state=int(args.seed),
        bootstrap=True,
    )
    logging.info("Training representative Random Forest on %d rows", len(sampled_idx))
    model.fit(X_fit, y_fit)

    model_path: Path | None = None
    try:
        import joblib

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "representative_random_forest.joblib"
        joblib.dump({"model": model, "encoder": encoder}, model_path)
    except Exception:
        logging.exception("Could not save Random Forest model artifact.")

    return (
        _predict_tabular_batches(model, encoder, validation_frame[feature_columns], int(args.prediction_batch_size)),
        _predict_tabular_batches(model, encoder, test_frame[feature_columns], int(args.prediction_batch_size)),
        model_path,
    )


def load_neural_source_predictions(
    output_dir: Path,
    requested_model: str | None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, str]:
    neural_id = best_neural_experiment_id(output_dir, requested_model or "nn_global_full_spatial_tsn_no_tp")
    neural_prediction_path(output_dir, neural_id)
    source_pred_dir = output_dir / "shared_artifacts" / "era5_source_comparison" / "predictions"
    validation_path = source_pred_dir / "source_neural_era5_to_era5_available_no_tp_validation_predictions.parquet"
    test_path = source_pred_dir / "source_neural_era5_to_seas5_available_no_tp_test_predictions.parquet"
    if not validation_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "Best-neural ERA5-validation and ECMWF-test source predictions are required. "
            f"Missing: {[str(path) for path in [validation_path, test_path] if not path.is_file()]}"
        )
    val = pd.read_parquet(validation_path)
    test = pd.read_parquet(test_path)
    val[DATE_COLUMN] = pd.to_datetime(val[DATE_COLUMN], errors="coerce")
    test[DATE_COLUMN] = pd.to_datetime(test[DATE_COLUMN], errors="coerce")
    val[TARGET_COLUMN] = val["target_binary"].astype(np.int8)
    test[TARGET_COLUMN] = test["target_binary"].astype(np.int8)
    return (
        val,
        val["pred_proba"].to_numpy(dtype=np.float32),
        test,
        test["pred_proba"].to_numpy(dtype=np.float32),
        neural_id,
    )


def tables_from_metrics(metric_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    common = [
        "model_group",
        "model",
        "status",
        "training_source",
        "validation_source",
        "test_source",
        "region",
        "region_display",
        "precision",
        "recall",
        "f1",
        "f1_error",
        "average_precision",
        "average_precision_error",
        "roc_auc",
        "brier_score",
        "threshold",
        "validation_f1_at_threshold",
        "notes",
    ]
    overall = metrics[metrics["split"].eq("test")].loc[:, common].copy()
    yearly = metrics[metrics["split"].astype(str).str.fullmatch(r"test_\d{4}")].copy()
    if not yearly.empty:
        yearly["year"] = yearly["split"].str.replace("test_", "", regex=False).astype(int)
        yearly = yearly.loc[:, common[:8] + ["year"] + common[8:]].copy()
    return overall, yearly


def plot_model_bars(overall: pd.DataFrame, output_dir: Path) -> None:
    global_rows = overall[overall["region"].astype(str).eq("global")].copy()
    if global_rows.empty:
        return
    plot_dir = output_dir / "shared_artifacts" / "source_plots_mixed"
    plot_dir.mkdir(parents=True, exist_ok=True)
    global_rows = global_rows.sort_values("average_precision", ascending=True)
    labels = global_rows["model"].astype(str).tolist()
    for metric, stem, title, color in [
        ("average_precision", f"{EXPERIMENT_STEM}_pr_auc", "Representative ERA5 -> ECMWF no-tp model PR-AUC", "#2f6f9f"),
        ("f1", f"{EXPERIMENT_STEM}_f1", "Representative ERA5 -> ECMWF no-tp model F1", "#7a9a3a"),
    ]:
        values = pd.to_numeric(global_rows[metric], errors="coerce").to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8.0, 4.3))
        ax.barh(labels, values, color=color)
        ax.set_xlabel(metric.replace("_", " ").upper() if metric == "f1" else "PR-AUC")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(0, max(0.75, float(np.nanmax(values)) * 1.12 if len(values) else 0.75))
        for idx, value in enumerate(values):
            if np.isfinite(value):
                ax.text(value + 0.01, idx, f"{value:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        for suffix in [".png", ".pdf"]:
            fig.savefig(plot_dir / f"{stem}{suffix}", dpi=250, bbox_inches="tight")
        plt.close(fig)


def write_compact_experiment(output_dir: Path, overall: pd.DataFrame, yearly: pd.DataFrame) -> None:
    exp_dir = output_dir / "experiments" / "31_representative_model_ablation_era5_to_ecmwf_no_tp"
    table_dir = exp_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    global_rows = overall[overall["region"].astype(str).eq("global")].copy()
    global_rows = global_rows.sort_values("average_precision", ascending=False)
    global_rows.loc[:, ["f1", "f1_error", "average_precision", "average_precision_error", "roc_auc"]] = global_rows[
        ["f1", "f1_error", "average_precision", "average_precision_error", "roc_auc"]
    ].round(4)
    global_rows.rename(
        columns={
            "model_group": "Model Group",
            "model": "Model",
            "f1": "F1",
            "f1_error": "F1 Error",
            "average_precision": "PR-AUC",
            "average_precision_error": "PR-AUC Error",
            "roc_auc": "ROC-AUC",
            "notes": "Notes",
        }
    ).loc[:, ["Model Group", "Model", "F1", "F1 Error", "PR-AUC", "PR-AUC Error", "ROC-AUC"]].to_csv(
        table_dir / "representative_model_metrics.csv",
        index=False,
    )
    protocol = global_rows.rename(
        columns={
            "model_group": "Model Group",
            "model": "Model",
            "training_source": "Train",
            "validation_source": "Validation",
            "test_source": "Test",
            "notes": "Notes",
        }
    ).loc[:, ["Model Group", "Model", "Train", "Validation", "Test", "Notes"]]
    protocol.to_csv(table_dir / "representative_model_protocol.csv", index=False)
    if not yearly.empty:
        year_rows = yearly[yearly["region"].astype(str).eq("global")].copy()
        year_rows.loc[:, ["f1", "average_precision", "roc_auc"]] = year_rows[["f1", "average_precision", "roc_auc"]].round(4)
        year_rows.rename(
            columns={
                "model": "Model",
                "year": "Year",
                "f1": "F1",
                "average_precision": "PR-AUC",
                "roc_auc": "ROC-AUC",
            }
        ).loc[:, ["Model", "Year", "F1", "PR-AUC", "ROC-AUC"]].to_csv(
            table_dir / "representative_model_by_year.csv",
            index=False,
        )

    (exp_dir / "description.md").write_text(
        "\n".join(
            [
                "# Representative Model Ablation ERA5 To ECMWF No Tp",
                "",
                "## Purpose",
                "Compact model-family comparison on the ERA5-trained, SEAS5/ECMWF-tested no-tp source-shift setting.",
                "",
                "## Source Tables",
                "- `representative_model_ablation_era5_to_ecmwf_no_tp.csv`",
                "- `representative_model_ablation_era5_to_ecmwf_no_tp_by_year.csv`",
                "",
                "## Notes",
                "- The model set is intentionally small: one representative from each broad family plus the best neural model.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    best = global_rows.iloc[0] if not global_rows.empty else None
    analysis = [
        "# Analysis",
        "",
        (
            f"- Best global PR-AUC in this compact source-shift comparison: "
            f"{best['model']} ({best['average_precision']:.4f})."
            if best is not None
            else "- No completed global rows were available."
        ),
        "- All tabular models use ERA5-derived train/validation climate features with `tp` removed, then score SEAS5/ECMWF test rows.",
        "- The neural row uses the ERA5 spatial TSN checkpoint retrained with `tp` removed from the dynamic climate inputs.",
    ]
    (exp_dir / "analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")

    plot_src = output_dir / "shared_artifacts" / "source_plots_mixed"
    for stem in [f"{EXPERIMENT_STEM}_pr_auc", f"{EXPERIMENT_STEM}_f1"]:
        for suffix, subdir in [(".png", "png"), (".pdf", "pdf")]:
            src = plot_src / f"{stem}{suffix}"
            if src.exists():
                dst = exp_dir / "plots" / subdir / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)


def run(args: Any) -> dict[str, Any]:
    setup_logging(args.output_dir)
    np.random.seed(args.seed)
    feature_config = load_yaml(args.feature_config)
    catboost_config = load_yaml(args.catboost_config)
    regions = load_regions(args.regions_file)
    variables = [str(variable) for variable in feature_config["climate_data_params"]["climate_variables"]]
    comparison_variables = [variable for variable in variables if variable not in set(args.drop_climate_variables)]
    if not comparison_variables:
        raise ValueError("Representative model ablation cannot drop every climate variable.")

    inventory = era5_inventory(args.era5_feature_root, comparison_variables)
    seas5, max_year = load_available_seas5_frame(
        args.features_path,
        inventory,
        train_start_year=args.train_start_year,
        test_end_year=args.test_end_year,
    )
    rows_before = len(seas5)
    seas5, coverage_rows = filter_full_era5_climate_coverage(
        seas5,
        era5_root=args.era5_feature_root,
        variables=comparison_variables,
        n_days=int(feature_config["climate_data_params"]["n_days"]),
    )
    masks = split_masks_by_year(
        seas5,
        train_start_year=args.train_start_year,
        validation_start_year=args.validation_start_year,
        test_start_year=args.test_start_year,
        test_end_year=max_year,
    )

    ignored = (
        catboost_config.get("catboost_train", {})
        .get("features", {})
        .get("ignored", DEFAULT_IGNORED_FEATURES)
        if isinstance(catboost_config.get("catboost_train"), dict)
        else DEFAULT_IGNORED_FEATURES
    )
    feature_columns = model_feature_columns(
        seas5,
        ignored,
        use_lat_lon_features=args.use_lat_lon_features,
    )
    validate_no_leakage_features(feature_columns)
    feature_columns = build_feature_sets(feature_columns)["full"]["columns"]
    feature_columns = drop_climate_variable_features(feature_columns, args.drop_climate_variables)

    era5_climate, era5_cols = load_or_build_era5_climate_features(
        seas5,
        feature_config,
        inventory,
        args.cache_dir,
        era5_root=args.era5_feature_root,
        force_rebuild=args.force_rebuild_era5_features,
        test_end_year=max_year,
    )
    missing = sorted(set([col for col in feature_columns if col in era5_cols or col.startswith(tuple(f"{v}_" for v in comparison_variables))]) - set(era5_cols) - set(seas5.columns))
    if missing:
        raise ValueError(f"Unexpected missing generated ERA5 source columns: {missing[:5]}")
    era5 = replace_weather_features(seas5, era5_climate, feature_columns)

    era5_train = era5.loc[masks["train"]].reset_index(drop=True)
    era5_val = era5.loc[masks["validation"]].reset_index(drop=True)
    seas5_test = seas5.loc[masks["test"]].reset_index(drop=True)

    artifact_root = args.output_dir / "shared_artifacts" / "representative_model_ablation"
    artifact_root.mkdir(parents=True, exist_ok=True)
    registry_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    notes_base = (
        "Representative model-family ablation on ERA5 -> SEAS5/ECMWF source shift with tp removed. "
        f"Rows use ERA5-covered sampled case-control domain, years {args.test_start_year}-{max_year}; "
        f"coverage filter {rows_before} -> {len(seas5)} rows."
    )
    model_dir = artifact_root / "models"

    for spec in MODEL_SPECS:
        key = spec["model_key"]
        model = spec["model"]
        model_group = spec["model_group"]
        notes = f"{notes_base} {spec['notes']}"
        experiment_id = f"{EXPERIMENT_STEM}_{key}"
        logging.info("Running representative model %s", model)
        if key == "catboost_full":
            cat_model, cat_features = train_source_catboost(
                experiment_id,
                era5_train,
                era5_val,
                feature_columns,
                feature_config,
                args,
                model_dir,
            )
            numerical_cat = feature_config.get("numerical_cat_features", [])
            val_prob = predict_catboost(
                cat_model,
                normalize_cat_columns(era5_val[feature_columns], cat_features, numerical_cat),
                cat_features,
            )
            test_prob = predict_catboost(
                cat_model,
                normalize_cat_columns(seas5_test[feature_columns], cat_features, numerical_cat),
                cat_features,
            )
        elif key == "logistic_regression":
            val_prob, test_prob, _ = train_logistic(era5_train, era5_val, seas5_test, feature_columns, feature_config, args, model_dir)
        elif key == "poisson_point_process":
            val_prob, test_prob, _ = train_poisson(era5_train, era5_val, seas5_test, feature_columns, feature_config, args, model_dir)
        elif key == "random_forest":
            val_prob, test_prob, _ = train_random_forest(era5_train, era5_val, seas5_test, feature_columns, feature_config, args, model_dir)
        elif key == "best_neural":
            neural_val, val_prob, neural_test, test_prob, neural_id = load_neural_source_predictions(
                args.output_dir,
                args.neural_model,
            )
            notes = f"{notes} Source checkpoint: {neural_id}."
            reg, rows = _evaluate_predictions(
                experiment_id=experiment_id,
                model=model,
                model_group=model_group,
                feature_set="best neural no-tp source-shift",
                validation_frame=neural_val,
                test_frame=neural_test,
                validation_prob=val_prob,
                test_prob=test_prob,
                regions=regions,
                args=args,
                artifact_root=artifact_root,
                notes=notes,
                feature_count=None,
            )
            registry_rows.append(reg)
            metric_rows.extend(rows)
            continue
        else:
            raise ValueError(f"Unknown representative model key: {key}")

        reg, rows = _evaluate_predictions(
            experiment_id=experiment_id,
            model=model,
            model_group=model_group,
            feature_set="full no-tp features",
            validation_frame=era5_val,
            test_frame=seas5_test,
            validation_prob=val_prob,
            test_prob=test_prob,
            regions=regions,
            args=args,
            artifact_root=artifact_root,
            notes=notes,
            feature_count=len(feature_columns),
        )
        registry_rows.append(reg)
        metric_rows.extend(rows)

    registry = pd.DataFrame(registry_rows)
    metrics = pd.DataFrame(metric_rows)
    registry.to_csv(artifact_root / f"{EXPERIMENT_STEM}_registry.csv", index=False)
    metrics.to_json(
        artifact_root / f"{EXPERIMENT_STEM}_metrics_wide.jsonl.gz",
        orient="records",
        lines=True,
        compression="gzip",
    )
    overall, yearly = tables_from_metrics(metric_rows)
    write_raw_table(args.output_dir, f"{EXPERIMENT_STEM}.csv", overall)
    write_raw_table(args.output_dir, f"{EXPERIMENT_STEM}_by_year.csv", yearly)
    plot_model_bars(overall, args.output_dir)
    write_compact_experiment(args.output_dir, overall, yearly)
    manifest = {
        "experiment": EXPERIMENT_STEM,
        "selected_models": MODEL_SPECS,
        "train_source": "ERA5",
        "validation_source": "ERA5",
        "test_source": "SEAS5/ECMWF",
        "drop_climate_variables": args.drop_climate_variables,
        "comparison_climate_variables": comparison_variables,
        "rows_before_coverage_filter": rows_before,
        "rows_available": len(seas5),
        "train_rows": int(masks["train"].sum()),
        "validation_rows": int(masks["validation"].sum()),
        "test_rows": int(masks["test"].sum()),
        "overall_rows": len(overall),
        "yearly_rows": len(yearly),
        "coverage_rows": coverage_rows,
    }
    write_json(artifact_root / f"{EXPERIMENT_STEM}_manifest.json", manifest)
    logging.info("Representative model ablation complete: %s", manifest)
    return manifest


def main() -> int:
    run(default_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
