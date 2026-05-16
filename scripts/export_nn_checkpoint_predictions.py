from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neural_net.models.lightning import SequenceStaticLightningModule
from src.neural_net.train_nn import (
    calculate_metrics,
    choose_threshold_f1,
    load_prepared_training_arrays,
    predict_logits,
    save_sampled_prediction_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export validation/test prediction parquet files from a trained NN checkpoint."
    )
    parser.add_argument("--data-path", required=True, type=Path, help="Prepared NN .npz dataset.")
    parser.add_argument("--checkpoint-path", required=True, type=Path, help="Lightning checkpoint path.")
    parser.add_argument("--metrics-path", required=True, type=Path, help="Output metrics.json path.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional training config path to record.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--random-error-trials", type=int, default=5)
    parser.add_argument("--random-error-sample-size", type=int, default=50_000)
    return parser.parse_args()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def read_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading prepared data from {args.data_path}", flush=True)
    with np.load(args.data_path, allow_pickle=False) as data:
        arrays = load_prepared_training_arrays(data)

    print(f"Loading checkpoint from {args.checkpoint_path}", flush=True)
    model = SequenceStaticLightningModule.load_from_checkpoint(
        str(args.checkpoint_path),
        map_location=device,
    )
    model.to(device)
    model.eval()

    split_specs = {
        "validation": (
            arrays["x_dyn_val"],
            arrays["x_stat_val"],
            arrays["x_cat_val"],
            np.asarray(arrays["y_val"]).astype(int),
            18,
        ),
        "test": (
            arrays["x_dyn_test"],
            arrays["x_stat_test"],
            arrays["x_cat_test"],
            np.asarray(arrays["y_test"]).astype(int),
            19,
        ),
    }

    logits: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    for split_name, (x_dyn, x_stat, x_cat, _y, _seed) in split_specs.items():
        print(f"Predicting {split_name} ({len(x_dyn):,} rows)", flush=True)
        split_logits = predict_logits(
            model,
            x_dyn,
            x_stat,
            x_cat,
            batch_size=args.batch_size,
            device=device,
        )
        logits[split_name] = split_logits.astype(np.float32)
        probabilities[split_name] = (1.0 / (1.0 + np.exp(-split_logits))).astype(np.float32)

    threshold, best_val_f1 = choose_threshold_f1(
        split_specs["validation"][3],
        probabilities["validation"],
    )
    print(f"Validation F1 threshold: {threshold:.6f} (F1={best_val_f1:.4f})", flush=True)

    pred_dir = args.metrics_path.parent / "legacy_sampled_predictions"
    prediction_artifacts = {}
    split_metrics = {}
    for split_name, (_x_dyn, _x_stat, _x_cat, y_true, seed) in split_specs.items():
        prediction_artifacts[split_name] = save_sampled_prediction_table(
            str(pred_dir / f"{split_name}_predictions.parquet"),
            y_true,
            logits[split_name],
            split_name,
        )
        split_metrics[split_name] = calculate_metrics(
            y_true,
            probabilities[split_name],
            threshold=threshold,
            error_trials=args.random_error_trials,
            error_sample_size=args.random_error_sample_size,
            error_seed=seed,
        )

    config = read_config(args.config_path)
    metrics = {
        "config_path": str(args.config_path) if args.config_path else None,
        "data_path": str(args.data_path),
        "model_path": str(args.checkpoint_path),
        "architecture": model.hparams.get("model_name"),
        "model_config": model.hparams.get("model_config"),
        "loss_training": {
            "name": model.hparams.get("loss_name"),
            "focal_gamma": model.hparams.get("focal_gamma"),
            "focal_alpha": model.hparams.get("focal_alpha"),
        },
        "selection_metric": config.get("selection_metric"),
        "validation_threshold": threshold,
        "validation_best_f1": best_val_f1,
        "validation": split_metrics["validation"],
        "test": split_metrics["test"],
        "legacy_sampled_metrics": {
            "validation": split_metrics["validation"],
            "test": split_metrics["test"],
            "evaluation_type": "legacy_sampled_case_control",
            "note": "Metrics are computed on the undersampled/case-control neural dataset.",
        },
        "prediction_artifacts": {
            key: str(value) for key, value in prediction_artifacts.items()
        },
        "split_sizes": {
            "validation": int(len(split_specs["validation"][3])),
            "test": int(len(split_specs["test"][3])),
        },
        "exported_from_checkpoint": True,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2, default=json_default), encoding="utf-8")
    print(f"Saved metrics to {args.metrics_path}", flush=True)


if __name__ == "__main__":
    main()
