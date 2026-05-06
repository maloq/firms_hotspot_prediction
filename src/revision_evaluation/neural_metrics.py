from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .commands import copy_if_exists
from .config import EvaluationConfig, NN_LABELS


LEGACY_FAILED_NN_IDS = ["lstm_mlp_full", "minimal_mlp_full", "ft_transformer_full"]


def import_neural_metrics(config: EvaluationConfig) -> None:
    out = config.output_dir
    main_path = out / "main_model_comparison.csv"
    neural_path = out / "embedding_fusion_ablation.csv"
    if not main_path.exists() or not neural_path.exists():
        print("Skipping NN metric import: flat result tables are not present.", flush=True)
        return

    main_table = pd.read_csv(main_path)
    neural_table = pd.read_csv(neural_path)
    registry_path = out / "experiment_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()

    global_row = main_table[main_table["Region"].astype(str).eq("Global")].iloc[0]
    support = int(global_row["support"])
    positives = int(global_row["positives"])

    main_rows: list[dict[str, Any]] = []
    neural_rows: list[dict[str, Any]] = []
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
        test = payload["test"]
        threshold = payload.get("validation_threshold", test.get("threshold"))
        labels.append(label)
        ids.append(exp_id)

        main_rows.append(
            {
                "Model": label,
                "Feature set": "global full NN features",
                "Region": "Global",
                "support": support,
                "positives": positives,
                "precision": test.get("precision"),
                "recall": test.get("recall"),
                "f1": test.get("f1"),
                "PR-AUC": test.get("ap"),
                "ROC-AUC": None,
                "Brier": None,
                "threshold": threshold,
            }
        )
        neural_rows.append(
            {
                "experiment": label,
                "model": "Neural",
                "feature_set": "global full NN features",
                "region": "global",
                "region_display": "Global",
                "period": "2021-2025",
                "support": support,
                "positives": positives,
                "negatives": support - positives,
                "positive_rate": positives / support,
                "precision": test.get("precision"),
                "recall": test.get("recall"),
                "f1": test.get("f1"),
                "average_precision": test.get("ap"),
                "roc_auc": None,
                "brier_score": None,
                "threshold": threshold,
                "validation_threshold": threshold,
                "train_rows": payload.get("split_sizes", {}).get("train"),
                "architecture": payload.get("architecture"),
                "source_metrics": str(metrics_file),
            }
        )
        registry_rows.append(
            {
                "experiment_id": exp_id,
                "experiment_type": "main_model_comparison",
                "model": label,
                "feature_set": "global full NN features",
                "status": "completed",
                "feature_count": None,
                "threshold": threshold,
                "threshold_source": "validation_f1_max",
                "validation_f1_at_threshold": payload.get("validation_best_f1"),
                "model_path": payload.get("model_path"),
                "prediction_paths": None,
                "notes": f"Imported from {metrics_file}.",
            }
        )
        copy_nn_artifacts(config, metrics_file, payload, exp_id)

    if not main_rows:
        print("No NN metric files matched the evaluation config.", flush=True)
        return

    main_table = main_table[~main_table["Model"].isin(labels)]
    main_table = pd.concat([main_table, pd.DataFrame(main_rows)], ignore_index=True)
    main_table = main_table.sort_values(["Region", "PR-AUC"], ascending=[True, False], na_position="last")
    main_table.to_csv(main_path, index=False)

    neural_table = neural_table[~neural_table["experiment"].isin(labels)]
    neural_table = pd.concat([neural_table, pd.DataFrame(neural_rows)], ignore_index=True)
    neural_table.to_csv(neural_path, index=False)

    if not registry.empty:
        registry = registry[~registry["experiment_id"].isin(ids + LEGACY_FAILED_NN_IDS)]
    registry = pd.concat([registry, pd.DataFrame(registry_rows)], ignore_index=True)
    registry.to_csv(registry_path, index=False)
    write_neural_plots(out, neural_table)
    print(f"Imported {len(main_rows)} global NN model metric file(s).", flush=True)


def copy_nn_artifacts(config: EvaluationConfig, metrics_file: Path, payload: dict[str, Any], exp_id: str) -> None:
    out = config.output_dir
    copy_if_exists(metrics_file, out / "neural_model_metrics" / f"{exp_id}_metrics.json")

    config_path = Path(payload.get("config_path", ""))
    copy_if_exists(config_path, out / "configs_used" / config_path.name)

    model_path = Path(payload.get("model_path", ""))
    copy_if_exists(model_path, out / "models" / model_path.name)


def write_neural_plots(output_dir: Path, neural_table: pd.DataFrame) -> None:
    plot_df = neural_table[
        neural_table["region"].astype(str).eq("global")
        & neural_table["period"].astype(str).eq("2021-2025")
    ].copy()
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    for metric, stem in [("average_precision", "embedding_fusion_pr_auc"), ("f1", "embedding_fusion_f1")]:
        work = plot_df.dropna(subset=[metric]).sort_values(metric)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(work) + 1.5)))
        ax.barh(work["experiment"], work[metric], color="#2563eb")
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / f"{stem}.png", dpi=240, bbox_inches="tight")
        fig.savefig(output_dir / "plots" / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
