from __future__ import annotations

import glob
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .stages import copy_if_exists
from .config import EvaluationConfig, NN_LABELS
from .artifacts import prune_empty_dirs
from .tabular import (
    LAT_COLUMN,
    LON_COLUMN,
    Region,
    compute_metric_errors,
    compute_metrics,
    load_regions,
)


LEGACY_FAILED_NN_IDS = ["lstm_mlp_full", "minimal_mlp_full", "ft_transformer_full"]
MAIN_FEATURE_SET = "global full NN features"
EVALUATION_TYPE = "legacy_sampled_case_control"


def import_neural_metrics(config: EvaluationConfig) -> None:
    out = config.output_dir
    main_path = out / "main_model_comparison.csv"
    neural_path = out / "embedding_fusion_ablation.csv"
    if not main_path.exists() or not neural_path.exists():
        print("Skipping NN metric import: flat result tables are not present.", flush=True)
        return

    main_table = pd.read_csv(main_path)
    by_year_path = out / "main_model_comparison_by_year.csv"
    main_by_year = pd.read_csv(by_year_path) if by_year_path.exists() else pd.DataFrame()
    neural_table = pd.read_csv(neural_path)
    neural_by_year_path = out / "embedding_fusion_ablation_by_year.csv"
    neural_by_year = pd.read_csv(neural_by_year_path) if neural_by_year_path.exists() else pd.DataFrame()
    registry_path = out / "experiment_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    regions = load_regions(config.regions_file)

    main_rows: list[dict[str, Any]] = []
    main_year_rows: list[dict[str, Any]] = []
    neural_rows: list[dict[str, Any]] = []
    neural_year_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    labels: list[str] = []
    ids: list[str] = []

    for metrics_file in sorted(Path(p) for p in glob.glob(config.nn_metrics_glob)):
        key = metrics_file.parent.name.removeprefix("nn_global_full_")
        if key not in config.new_nn_models:
            continue

        with metrics_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        label = NN_LABELS.get(key, f"{payload.get('architecture', key)} (global full)")
        exp_id = f"nn_global_full_{key}"
        threshold = payload.get("validation_threshold", (payload.get("test") or {}).get("threshold", 0.5))
        threshold = 0.5 if threshold is None else float(threshold)
        primary = payload.get("primary_full_grid_calibrated_metrics") or {}
        metric_rows = neural_metric_rows_from_artifacts(
            exp_id=exp_id,
            label=label,
            payload=payload,
            threshold=threshold,
            regions=regions,
            config=config,
        )
        if not metric_rows:
            raise RuntimeError(
                f"Neural metric import for {exp_id} did not produce regional rows. "
                "Check prediction_artifacts and prepared_data.npz instead of publishing a partial comparison."
            )

        labels.append(label)
        ids.append(exp_id)

        for row in metric_rows:
            if row["split"] == "test":
                main_rows.append(main_row_from_metric(row))
                neural_rows.append(neural_row_from_metric(row, payload, metrics_file))
            elif row["split"].startswith("test_"):
                main_year_rows.append(main_year_row_from_metric(row))
                neural_year_rows.append(neural_row_from_metric(row, payload, metrics_file))

        registry_rows.append(
            {
                "experiment_id": exp_id,
                "experiment_type": "main_model_comparison",
                "model": label,
                "feature_set": MAIN_FEATURE_SET,
                "status": "completed",
                "feature_count": None,
                "threshold": threshold,
                "threshold_source": "validation_f1_max",
                "validation_f1_at_threshold": payload.get("validation_best_f1"),
                "model_path": payload.get("model_path"),
                "prediction_paths": payload.get("prediction_artifacts"),
                "evaluation_type": primary.get("evaluation_type", EVALUATION_TYPE),
                "is_primary": bool(primary.get("status") == "completed"),
                "is_main_nn_model": key == config.main_nn_model,
                "notes": (
                    f"Imported from {metrics_file}. "
                    f"Primary full-grid status: {primary.get('status', 'not_available')}."
                ),
            }
        )
        copy_nn_artifacts(config, metrics_file, payload, exp_id)

    if not main_rows and not neural_rows:
        print("No NN metric files matched the evaluation config.", flush=True)
        return

    main_table = replace_rows(main_table, "Model", labels)
    main_table = pd.concat([main_table, pd.DataFrame(main_rows)], ignore_index=True)
    main_table = sort_if_possible(main_table, ["Region", "PR-AUC"], [True, False])
    main_table.to_csv(main_path, index=False)

    if main_year_rows:
        main_by_year = replace_rows(main_by_year, "model", labels)
        main_by_year = pd.concat([main_by_year, pd.DataFrame(main_year_rows)], ignore_index=True)
        main_by_year = sort_if_possible(main_by_year, ["model", "region_display", "period"], [True, True, True])
        main_by_year.to_csv(by_year_path, index=False)

    neural_table = replace_rows(neural_table, "experiment", labels)
    neural_table = pd.concat([neural_table, pd.DataFrame(neural_rows)], ignore_index=True)
    neural_table.to_csv(neural_path, index=False)

    if neural_year_rows:
        neural_by_year = replace_rows(neural_by_year, "experiment", labels)
        neural_by_year = pd.concat([neural_by_year, pd.DataFrame(neural_year_rows)], ignore_index=True)
        neural_by_year.to_csv(neural_by_year_path, index=False)

    if not registry.empty:
        registry = registry[~registry["experiment_id"].isin(ids + LEGACY_FAILED_NN_IDS)]
    registry = pd.concat([registry, pd.DataFrame(registry_rows)], ignore_index=True)
    registry.to_csv(registry_path, index=False)

    sync_legacy_model_tables(out, main_table, main_by_year)
    write_main_model_pr_plots(config)
    write_neural_plots(out, neural_table)
    import_neural_feature_ablation_metrics(config, regions)
    print(
        f"Imported {len(labels)} NN metric file(s); added {len(main_rows)} "
        "regional sampled/case-control NN row(s) to the main table.",
        flush=True,
    )


def neural_metric_rows_from_artifacts(
    *,
    exp_id: str,
    label: str,
    payload: dict[str, Any],
    threshold: float,
    regions: list[Region],
    config: EvaluationConfig,
    experiment_type: str = "main_model_comparison",
    feature_set: str = MAIN_FEATURE_SET,
) -> list[dict[str, Any]]:
    pred = read_prediction_artifact(payload, "test")
    frame = read_test_coordinate_frame(payload)
    if pred is None or frame is None or len(pred) != len(frame):
        return []

    y_true = pred["is_fire"].to_numpy(dtype=np.int8)
    y_prob = pred["prob_raw"].to_numpy(dtype=np.float32)
    years = frame["year"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []

    def add(split: str, region: str, display: str, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        metrics = compute_metrics(y_true[mask], y_prob[mask], threshold)
        metrics.update(
            compute_metric_errors(
                y_true[mask],
                y_prob[mask],
                threshold,
                trials=config.random_error_trials,
                sample_size=config.random_error_sample_size,
                seed=stable_seed(f"{exp_id}:{split}:{region}"),
            )
        )
        rows.append(
            {
                "experiment_id": exp_id,
                "experiment_type": experiment_type,
                "evaluation_type": EVALUATION_TYPE,
                "is_primary": False,
                "model": label,
                "feature_set": feature_set,
                "split": split,
                "region": region,
                "region_display": display,
                **metrics,
            }
        )

    add("test", "global", "Global", np.ones(len(frame), dtype=bool))
    for region in regions:
        add("test", region.name, region.display_name, region.mask(frame))

    period_specs: list[tuple[str, np.ndarray]] = [(str(year), years == year) for year in range(2021, 2026)]
    period_specs.extend(
        [
            ("2021-2023", (years >= 2021) & (years <= 2023)),
            ("2021-2025", (years >= 2021) & (years <= 2025)),
        ]
    )
    for period, period_mask in period_specs:
        if not period_mask.any():
            continue
        split = f"test_{period}"
        add(split, "global", "Global", period_mask)
        for region in regions:
            mask = period_mask & region.mask(frame)
            if mask.any():
                add(split, region.name, region.display_name, mask)
    return rows


def import_neural_feature_ablation_metrics(config: EvaluationConfig, regions: list[Region]) -> None:
    model_name = config.neural_feature_ablation_model or config.main_nn_model
    selected_variants = set(config.neural_feature_ablation_variants or [])
    metric_specs: list[tuple[Path, str, str]] = []

    baseline_path = next(
        (
            Path(p)
            for p in sorted(glob.glob(config.nn_metrics_glob))
            if Path(p).parent.name == f"nn_global_full_{model_name}"
        ),
        None,
    )
    if baseline_path is not None:
        metric_specs.append((baseline_path, "full", "Full neural feature set"))

    for metrics_file in sorted(Path(p) for p in glob.glob(config.nn_feature_ablation_metrics_glob)):
        with metrics_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ablation = payload.get("input_ablation") or {}
        base_model = str(ablation.get("base_model") or model_name)
        variant = str(ablation.get("variant") or metrics_file.parent.name.removeprefix(f"nn_feature_ablation_{base_model}_"))
        if base_model != model_name:
            continue
        if selected_variants and variant not in selected_variants:
            continue
        label = str(ablation.get("label") or payload.get("feature_set") or variant.replace("_", " ").title())
        metric_specs.append((metrics_file, variant, label))

    if not metric_specs:
        return

    out = config.output_dir
    table_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    ids: list[str] = []
    base_label = NN_LABELS.get(model_name, f"{model_name} (global full)")

    for metrics_file, variant, variant_label in metric_specs:
        with metrics_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        exp_id = f"nn_feature_ablation_{model_name}_{variant}"
        threshold = payload.get("validation_threshold", (payload.get("test") or {}).get("threshold", 0.5))
        threshold = 0.5 if threshold is None else float(threshold)
        metric_rows = neural_metric_rows_from_artifacts(
            exp_id=exp_id,
            label=base_label,
            payload=payload,
            threshold=threshold,
            regions=regions,
            config=config,
            experiment_type="neural_feature_ablation",
            feature_set=variant_label,
        )
        for row in metric_rows:
            out_row = neural_feature_ablation_row_from_metric(
                row,
                payload=payload,
                metrics_file=metrics_file,
                variant=variant,
                variant_label=variant_label,
            )
            if row["split"] == "test":
                table_rows.append(out_row)
            elif row["split"].startswith("test_"):
                year_rows.append(out_row)

        primary = payload.get("primary_full_grid_calibrated_metrics") or {}
        registry_rows.append(
            {
                "experiment_id": exp_id,
                "experiment_type": "neural_feature_ablation",
                "model": base_label,
                "feature_set": variant_label,
                "status": "completed" if metric_rows else "no_metric_rows",
                "feature_count": None,
                "threshold": threshold,
                "threshold_source": "validation_f1_max",
                "validation_f1_at_threshold": payload.get("validation_best_f1"),
                "model_path": payload.get("model_path"),
                "prediction_paths": payload.get("prediction_artifacts"),
                "evaluation_type": primary.get("evaluation_type", EVALUATION_TYPE),
                "is_primary": False,
                "is_main_nn_model": model_name == config.main_nn_model,
                "notes": f"Imported neural feature ablation variant '{variant}' from {metrics_file}.",
            }
        )
        ids.append(exp_id)
        copy_nn_artifacts(config, metrics_file, payload, exp_id)

    if table_rows:
        table = add_feature_ablation_deltas(pd.DataFrame(table_rows))
        table = sort_if_possible(table, ["region_display", "variant"], [True, True])
        table.to_csv(out / "neural_feature_ablation.csv", index=False)
    if year_rows:
        year_table = add_feature_ablation_deltas(pd.DataFrame(year_rows))
        year_table = sort_if_possible(year_table, ["region_display", "period", "variant"], [True, True, True])
        year_table.to_csv(out / "neural_feature_ablation_by_year.csv", index=False)

    registry_path = out / "experiment_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    if not registry.empty:
        registry = registry[~registry["experiment_id"].isin(ids)]
    registry = pd.concat([registry, pd.DataFrame(registry_rows)], ignore_index=True)
    registry.to_csv(registry_path, index=False)
    if table_rows:
        write_neural_feature_ablation_plots(out, pd.DataFrame(table_rows))


def neural_feature_ablation_row_from_metric(
    row: dict[str, Any],
    *,
    payload: dict[str, Any],
    metrics_file: Path,
    variant: str,
    variant_label: str,
) -> dict[str, Any]:
    return {
        "experiment_id": row["experiment_id"],
        "experiment": variant_label,
        "variant": variant,
        "model": row["model"],
        "feature_set": row["feature_set"],
        "region": row["region"],
        "region_display": row["region_display"],
        "period": row["split"].removeprefix("test_") if row["split"].startswith("test_") else "2021-2025",
        "support": row.get("support"),
        "positives": row.get("positives"),
        "positive_rate": row.get("positive_rate"),
        "predicted_positives": row.get("predicted_positives"),
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "f1": row.get("f1"),
        "f1_error": row.get("f1_error"),
        "average_precision": row.get("average_precision"),
        "average_precision_error": row.get("average_precision_error"),
        "roc_auc": row.get("roc_auc"),
        "brier_score": row.get("brier_score"),
        "threshold": row.get("threshold"),
        "validation_threshold": row.get("threshold"),
        "evaluation_type": row.get("evaluation_type", EVALUATION_TYPE),
        "is_primary": row.get("is_primary", False),
        "train_rows": (payload.get("split_sizes") or {}).get("train"),
        "architecture": payload.get("architecture"),
        "source_metrics": str(metrics_file),
    }


def add_feature_ablation_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta_average_precision_vs_full"] = np.nan
    out["delta_f1_vs_full"] = np.nan
    for _, group in out.groupby(["region", "period"], dropna=False):
        baseline = group[group["variant"].astype(str).eq("full")]
        if baseline.empty:
            continue
        full = baseline.iloc[0]
        idx = group.index
        out.loc[idx, "delta_average_precision_vs_full"] = (
            pd.to_numeric(full.get("average_precision"), errors="coerce")
            - pd.to_numeric(out.loc[idx, "average_precision"], errors="coerce")
        )
        out.loc[idx, "delta_f1_vs_full"] = (
            pd.to_numeric(full.get("f1"), errors="coerce")
            - pd.to_numeric(out.loc[idx, "f1"], errors="coerce")
        )
    return out


def read_prediction_artifact(payload: dict[str, Any], split: str) -> pd.DataFrame | None:
    artifacts = payload.get("prediction_artifacts") or {}
    path = Path(str(artifacts.get(split) or ""))
    if not path.is_file():
        return None
    pred = pd.read_parquet(path)
    required = {"is_fire", "prob_raw"}
    if not required.issubset(pred.columns):
        return None
    return pred


def read_test_coordinate_frame(payload: dict[str, Any]) -> pd.DataFrame | None:
    path = Path(str(payload.get("data_path") or ""))
    if not path.is_file():
        return None
    with np.load(path) as data:
        keys = set(data.files)
        if not {"split", "lat", "lon"}.issubset(keys):
            return None
        split = np.asarray(data["split"], dtype=np.int8)
        test_mask = split == 2
        if not test_mask.any():
            return None
        if "years" in keys:
            years = np.asarray(data["years"])[test_mask].astype(int)
        elif "dates" in keys:
            dates = pd.to_datetime(np.asarray(data["dates"])[test_mask], unit="D", errors="coerce")
            years = dates.year.to_numpy(dtype=int)
        else:
            return None
        return pd.DataFrame(
            {
                LAT_COLUMN: np.asarray(data["lat"])[test_mask].astype(float),
                LON_COLUMN: np.asarray(data["lon"])[test_mask].astype(float),
                "year": years,
            }
        )


def fallback_global_metric_rows(
    *,
    exp_id: str,
    label: str,
    payload: dict[str, Any],
    threshold: float,
    main_table: pd.DataFrame,
) -> list[dict[str, Any]]:
    test = payload.get("test") or {}
    support, positives = fallback_support(main_table)
    return [
        {
            "experiment_id": exp_id,
            "experiment_type": "main_model_comparison",
            "evaluation_type": EVALUATION_TYPE,
            "is_primary": False,
            "model": label,
            "feature_set": MAIN_FEATURE_SET,
            "split": "test",
            "region": "global",
            "region_display": "Global",
            "support": test.get("support", support),
            "positives": test.get("positives", positives),
            "positive_rate": safe_ratio(test.get("positives", positives), test.get("support", support)),
            "predicted_positives": None,
            "precision": test.get("precision"),
            "recall": test.get("recall"),
            "f1": test.get("f1"),
            "precision_error": test.get("precision_error"),
            "recall_error": test.get("recall_error"),
            "f1_error": test.get("f1_error"),
            "average_precision": test.get("average_precision", test.get("ap")),
            "average_precision_error": test.get("average_precision_error", test.get("ap_error")),
            "roc_auc": test.get("roc_auc"),
            "brier_score": test.get("brier_score"),
            "threshold": threshold,
        }
    ]


def fallback_support(main_table: pd.DataFrame) -> tuple[int | None, int | None]:
    if main_table.empty or "Region" not in main_table.columns:
        return None, None
    global_rows = main_table[main_table["Region"].astype(str).eq("Global")]
    if global_rows.empty:
        return None, None
    row = global_rows.iloc[0]
    support = int(row["support"]) if "support" in row and pd.notna(row["support"]) else None
    positives = int(row["positives"]) if "positives" in row and pd.notna(row["positives"]) else None
    return support, positives


def main_row_from_metric(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Model": row["model"],
        "Feature set": row["feature_set"],
        "Region": row["region_display"],
        "evaluation_type": row.get("evaluation_type", EVALUATION_TYPE),
        "is_primary": row.get("is_primary", False),
        "support": row.get("support"),
        "positives": row.get("positives"),
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "f1": row.get("f1"),
        "F1 error": row.get("f1_error"),
        "PR-AUC": row.get("average_precision"),
        "PR-AUC error": row.get("average_precision_error"),
        "ROC-AUC": row.get("roc_auc"),
        "Brier": row.get("brier_score"),
        "threshold": row.get("threshold"),
    }


def main_year_row_from_metric(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": row["model"],
        "feature_set": row["feature_set"],
        "region_display": row["region_display"],
        "period": row["split"].removeprefix("test_"),
        "support": row.get("support"),
        "positives": row.get("positives"),
        "positive_rate": row.get("positive_rate"),
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "f1": row.get("f1"),
        "f1_error": row.get("f1_error"),
        "average_precision": row.get("average_precision"),
        "average_precision_error": row.get("average_precision_error"),
        "roc_auc": row.get("roc_auc"),
        "brier_score": row.get("brier_score"),
        "threshold": row.get("threshold"),
    }


def neural_row_from_metric(row: dict[str, Any], payload: dict[str, Any], metrics_file: Path) -> dict[str, Any]:
    return {
        "experiment": row["model"],
        "model": "Neural",
        "feature_set": row["feature_set"],
        "region": row["region"],
        "region_display": row["region_display"],
        "period": row["split"].removeprefix("test_") if row["split"].startswith("test_") else "2021-2025",
        "support": row.get("support"),
        "positives": row.get("positives"),
        "negatives": None if row.get("support") is None or row.get("positives") is None else row["support"] - row["positives"],
        "positive_rate": row.get("positive_rate"),
        "predicted_positives": row.get("predicted_positives"),
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "f1": row.get("f1"),
        "f1_error": row.get("f1_error"),
        "average_precision": row.get("average_precision"),
        "average_precision_error": row.get("average_precision_error"),
        "roc_auc": row.get("roc_auc"),
        "brier_score": row.get("brier_score"),
        "threshold": row.get("threshold"),
        "validation_threshold": row.get("threshold"),
        "evaluation_type": row.get("evaluation_type", EVALUATION_TYPE),
        "is_primary": row.get("is_primary", False),
        "train_rows": (payload.get("split_sizes") or {}).get("train"),
        "architecture": payload.get("architecture"),
        "source_metrics": str(metrics_file),
    }


def replace_rows(df: pd.DataFrame, column: str, values: list[str]) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    return df[~df[column].isin(values)].copy()


def sort_if_possible(df: pd.DataFrame, columns: list[str], ascending: list[bool]) -> pd.DataFrame:
    if df.empty or not all(col in df.columns for col in columns):
        return df
    return df.sort_values(columns, ascending=ascending, na_position="last")


def sync_legacy_model_tables(output_dir: Path, main_table: pd.DataFrame, main_by_year: pd.DataFrame) -> None:
    for path in [
        output_dir / "legacy_sampled_model_comparison.csv",
        output_dir / "legacy_sampled_case_control" / "model_comparison.csv",
    ]:
        if path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            main_table.to_csv(path, index=False)
    if main_by_year.empty:
        return
    for path in [
        output_dir / "legacy_sampled_model_comparison_by_year.csv",
        output_dir / "legacy_sampled_case_control" / "model_comparison_by_year.csv",
    ]:
        if path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            main_by_year.to_csv(path, index=False)


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        denom = float(denominator)
        if denom == 0:
            return None
        return float(numerator) / denom
    except Exception:
        return None


def stable_seed(value: str) -> int:
    return int(sum((idx + 1) * ord(char) for idx, char in enumerate(value)) % (2**32 - 1))


def copy_nn_artifacts(config: EvaluationConfig, metrics_file: Path, payload: dict[str, Any], exp_id: str) -> None:
    out = config.output_dir
    copy_if_exists(metrics_file, out / "neural_model_metrics" / f"{exp_id}_metrics.json")

    config_path = Path(payload.get("config_path", ""))
    if config_path.is_file():
        copy_if_exists(config_path, out / "configs_used" / config_path.name)

    model_path = Path(payload.get("model_path", ""))
    if model_path.is_file():
        copy_if_exists(model_path, out / "models" / model_path.name)

    for split, pred_path in (payload.get("prediction_artifacts") or {}).items():
        src = Path(str(pred_path))
        if src.is_file():
            copy_if_exists(src, out / "predictions" / f"{exp_id}_{split}_legacy_predictions.parquet")


def write_neural_plots(output_dir: Path, neural_table: pd.DataFrame) -> None:
    plot_df = neural_table[
        neural_table["region"].astype(str).eq("global")
        & neural_table["period"].astype(str).eq("2021-2025")
    ].copy()

    for metric, stem in [("average_precision", "embedding_fusion_pr_auc"), ("f1", "embedding_fusion_f1")]:
        work = plot_df.dropna(subset=[metric]).sort_values(metric)
        if work.empty:
            continue
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(work) + 1.5)))
        ax.barh(work["experiment"], work[metric], color="#2563eb")
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
        fig.savefig(plot_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
    prune_empty_dirs(output_dir)


def write_neural_feature_ablation_plots(output_dir: Path, table: pd.DataFrame) -> None:
    plot_df = table[
        table["region"].astype(str).eq("global")
        & table["period"].astype(str).eq("2021-2025")
    ].copy()
    if plot_df.empty:
        return

    for metric, stem in [
        ("average_precision", "neural_feature_ablation_pr_auc"),
        ("f1", "neural_feature_ablation_f1"),
    ]:
        work = plot_df.dropna(subset=[metric]).sort_values(metric)
        if work.empty:
            continue
        colors = ["#6b7280" if str(v) == "full" else "#0f766e" for v in work["variant"]]
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(work) + 1.5)))
        ax.barh(work["experiment"], work[metric], color=colors)
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
        fig.savefig(plot_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
    prune_empty_dirs(output_dir)


def write_main_model_pr_plots(config: EvaluationConfig) -> None:
    from .probability_overlays import read_prediction_columns, find_prediction_file

    output_dir = config.output_dir
    main_table = read_result_table(output_dir, "main_model_comparison.csv")
    if main_table.empty:
        raise FileNotFoundError(
            f"Cannot write main PR curves because main_model_comparison.csv is missing under {output_dir}."
        )

    global_rows = main_table[main_table["Region"].astype(str).eq("Global")].copy()
    if global_rows.empty:
        raise ValueError("Cannot write main PR curves: main_model_comparison.csv has no Global rows.")
    global_rows["PR-AUC"] = pd.to_numeric(global_rows["PR-AUC"], errors="coerce")
    global_rows = global_rows.dropna(subset=["PR-AUC"]).sort_values("PR-AUC", ascending=False)

    label_to_id = main_model_prediction_ids(config)
    model_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    unknown: list[str] = []
    for _, row in global_rows.iterrows():
        label = str(row["Model"])
        exp_id = label_to_id.get(label)
        if exp_id is None:
            unknown.append(label)
            continue
        try:
            prediction_path = find_prediction_file(output_dir, exp_id, "legacy")
        except FileNotFoundError:
            missing.append(f"{label} ({exp_id})")
            continue
        frame = read_prediction_columns(prediction_path, "auto")
        if frame.empty:
            raise ValueError(f"Prediction file is empty for {label}: {prediction_path}")
        model_rows.append(
            {
                "label": label,
                "experiment_id": exp_id,
                "path": prediction_path,
                "frame": frame,
                "threshold": float(row.get("threshold", 0.5)) if pd.notna(row.get("threshold", np.nan)) else 0.5,
            }
        )
    if unknown:
        raise ValueError(
            "Main comparison contains model label(s) without prediction-id mapping: "
            + ", ".join(sorted(unknown))
        )
    if missing:
        raise FileNotFoundError(
            "Cannot write all-model PR curves because prediction files are missing for: "
            + ", ".join(missing)
        )
    if not model_rows:
        raise ValueError("No main model prediction files were available for PR-curve plotting.")

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    write_global_pr_curve(model_rows, plot_dir)
    write_regional_metric_heatmap(model_rows, config, plot_dir)
    prune_empty_dirs(output_dir)


def read_result_table(output_dir: Path, name: str) -> pd.DataFrame:
    stem = Path(name).with_suffix("").name
    candidates = [
        output_dir / name,
        output_dir / "shared_artifacts" / "raw_tables_jsonl" / f"{stem}.jsonl.gz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".gz":
            return pd.read_json(path, orient="records", lines=True, compression="gzip")
        return pd.read_csv(path)
    return pd.DataFrame()


def main_model_prediction_ids(config: EvaluationConfig) -> dict[str, str]:
    ids = {
        "Logistic Regression (linear SGD)": "logistic_regression_full",
        "Poisson Point-Process GLM": "poisson_point_process_full",
        "FWI-only CatBoost": "catboost_fwi_only",
        "Weather-only CatBoost": "catboost_weather_only",
        "CatBoost": "catboost_full",
        "Random Forest": "random_forest_full",
    }
    for key in config.new_nn_models:
        label = NN_LABELS.get(key, f"{key} (global full)")
        ids[label] = f"nn_global_full_{key}"
    return ids


def write_global_pr_curve(model_rows: list[dict[str, Any]], plot_dir: Path) -> None:
    fig, (ax, metric_ax) = plt.subplots(
        1,
        2,
        figsize=(18, 7.2),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    summary_rows: list[dict[str, Any]] = []
    for item in model_rows:
        frame = item["frame"]
        y_true = frame["is_fire"].to_numpy(dtype=np.int8)
        prob = frame["predicted_probability"].to_numpy(dtype=np.float32)
        if len(np.unique(y_true)) < 2:
            raise ValueError(f"Cannot plot PR curve for {item['label']}: only one target class is present.")
        precision, recall, _ = precision_recall_curve(y_true, prob)
        ap = float(average_precision_score(y_true, prob))
        threshold = float(item["threshold"])
        f1 = float(f1_score(y_true, prob >= threshold, zero_division=0))
        ax.plot(recall, precision, linewidth=1.8, label=item["label"])
        summary_rows.append({"method": item["label"], "average_precision": ap, "f1": f1})

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Global sampled precision-recall curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 0))

    summary_df = pd.DataFrame(summary_rows).sort_values("average_precision", ascending=True)
    y_pos = np.arange(len(summary_df))
    metric_ax.barh(
        y_pos - 0.18,
        summary_df["average_precision"].astype(float),
        height=0.35,
        color="#2563eb",
        label="PR-AUC",
    )
    metric_ax.barh(y_pos + 0.18, summary_df["f1"].astype(float), height=0.35, color="#f97316", label="F1")
    metric_ax.set_yticks(y_pos)
    metric_ax.set_yticklabels([wrap_plot_label(v, 24) for v in summary_df["method"]], fontsize=8)
    metric_ax.set_xlabel("Absolute test metric")
    metric_ax.set_xlim(0, 1)
    metric_ax.set_title("Global sampled metrics")
    metric_ax.grid(axis="x", alpha=0.25)
    metric_ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "pr_curves_global.png", dpi=240, bbox_inches="tight")
    fig.savefig(plot_dir / "pr_curves_global.pdf", bbox_inches="tight")
    plt.close(fig)


def write_regional_metric_heatmap(model_rows: list[dict[str, Any]], config: EvaluationConfig, plot_dir: Path) -> None:
    regions = load_regions(config.regions_file)
    region_specs: list[tuple[str, Region | None]] = [("Global", None)]
    region_specs.extend((region.display_name, region) for region in regions)
    method_order = [item["label"] for item in model_rows]
    regional_rows: list[dict[str, Any]] = []

    for item in model_rows:
        frame = item["frame"]
        y_true = frame["is_fire"].to_numpy(dtype=np.int8)
        prob = frame["predicted_probability"].to_numpy(dtype=np.float32)
        for region_label, region in region_specs:
            mask = np.ones(len(frame), dtype=bool) if region is None else region.mask(frame)
            y_region = y_true[mask]
            p_region = prob[mask]
            threshold = float(item["threshold"])
            regional_rows.append(
                {
                    "region": region_label,
                    "method": item["label"],
                    "average_precision": (
                        float(average_precision_score(y_region, p_region))
                        if len(y_region) and len(np.unique(y_region)) == 2
                        else np.nan
                    ),
                    "f1": float(f1_score(y_region, p_region >= threshold, zero_division=0)),
                }
            )

    regional_df = pd.DataFrame(regional_rows)
    fig, axes = plt.subplots(1, 2, figsize=(18, max(5.0, 0.58 * len(region_specs) + 2.0)), squeeze=False)
    for ax, metric, title in [
        (axes[0][0], "average_precision", "Regional PR-AUC"),
        (axes[0][1], "f1", "Regional F1"),
    ]:
        pivot = (
            regional_df.pivot(index="region", columns="method", values=metric)
            .reindex(index=[name for name, _ in region_specs], columns=method_order)
        )
        values = pivot.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)
        im = ax.imshow(masked, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(method_order)))
        ax.set_xticklabels([wrap_plot_label(v, 18) for v in method_order], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(str), fontsize=9)
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                if np.isfinite(value):
                    color = "white" if value >= 0.55 else "black"
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Regional sampled metric contrast", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / "pr_curves_regions.png", dpi=240, bbox_inches="tight")
    fig.savefig(plot_dir / "pr_curves_regions.pdf", bbox_inches="tight")
    plt.close(fig)


def wrap_plot_label(label: Any, width: int = 24) -> str:
    wrapped = textwrap.wrap(str(label), width=width, break_long_words=False)
    return "\n".join(wrapped) if wrapped else str(label)
