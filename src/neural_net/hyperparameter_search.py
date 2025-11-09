"""Hyperparameter search utilities for neural net training.

This script reuses the training pipeline defined in ``train_nn.py`` to evaluate
multiple hyperparameter configurations and keeps track of their validation
Average Precision (AP). By default, it performs random search over a small
grid of reasonable parameters, but a custom search space can be provided via
YAML.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import CSVLogger

# Ensure project root is on the path when the script is invoked directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.neural_net.train_nn import train_pipeline  # noqa: E402


DEFAULT_SEARCH_SPACE = {
    "nn_model": {
        "architecture": ["lstm_mlp", "lstm_early_fusion"],
        "params": {
            "lstm_units": [128, 192, 256],
            "dropout_lstm": [0.0, 0.1],
            "merge_units": [128, 256],
            "dropout_merged": [0.0, 0.2],
        },
        "lightning": {
            "learning_rate": [0.001, 0.003, 0.005],
            "l2": [0.0, 5e-4, 1e-3],
        },
    },
    "batch_size": [2048, 4096],
}


@dataclass
class TrialResult:
    """Container for trial metadata."""

    index: int
    assignments: Dict[str, Any]
    status: str
    duration_sec: float
    val_ap: float | None
    train_ap: float | None
    sel_ap: float | None
    val_loss: float | None
    error: str | None = None

    def to_row(self) -> Dict[str, Any]:
        row = {
            "trial": self.index,
            "status": self.status,
            "duration_sec": round(self.duration_sec, 2),
            "val_ap": self.val_ap if self.val_ap is not None else "",
            "train_ap": self.train_ap if self.train_ap is not None else "",
            "sel_ap": self.sel_ap if self.sel_ap is not None else "",
            "val_loss": self.val_loss if self.val_loss is not None else "",
        }
        if self.error:
            row["error"] = self.error
        for path, value in self.assignments.items():
            row[f"param:{path}"] = value
        return row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter search for neural net training.")
    parser.add_argument(
        "--config",
        default="configs/features_config_30d_LSTM_early_fusion.yaml",
        help="Path to the base YAML config used for training.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional path to prepared_data.npz. Overrides path derived from the config.",
    )
    parser.add_argument(
        "--search-space",
        default=None,
        help="YAML file with hyperparameter choices. Falls back to internal defaults.",
    )
    parser.add_argument(
        "--strategy",
        choices=("random", "grid"),
        default="random",
        help="Search strategy for sampling parameter combinations.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=40,
        help="Maximum number of trials to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed used for sampling and Lightning reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hparam_search",
        help="Directory to store search results.",
    )
    parser.add_argument(
        "--disable-cuda",
        action="store_true",
        help="Force training on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first failed trial.",
    )
    return parser.parse_args(argv)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def resolve_data_path(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    if args.data_path:
        return Path(args.data_path).expanduser().resolve()
    default_dir = config.get("output_train_data_dir", "data/saved_features/nn_train_data")
    return (Path(default_dir) / "prepared_data.npz").expanduser().resolve()


def load_dataset(npz_path: Path) -> Dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")
    with np.load(npz_path) as data:
        required_keys = [
            "x_dyn_train",
            "x_dyn_val",
            "x_stat_train",
            "x_stat_val",
            "y_train",
            "y_val",
        ]
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise KeyError(f"Dataset at {npz_path} is missing keys: {missing}")
        return {key: data[key] for key in data.files}


def flatten_search_space(space: Dict[str, Any], prefix: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], List[Any]]]:
    choices: List[Tuple[Tuple[str, ...], List[Any]]] = []
    for key, value in space.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            choices.extend(flatten_search_space(value, path))
        else:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"Expected list/tuple of choices at path {'.'.join(path)}; got {type(value)}")
            value_list = list(value)
            if not value_list:
                raise ValueError(f"Choice list for path {'.'.join(path)} is empty.")
            choices.append((path, value_list))
    return choices


def set_nested(config: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    target = config
    for key in path[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[path[-1]] = value


def assign_parameters(base_config: Dict[str, Any], assignments: Dict[Tuple[str, ...], Any]) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    for path, value in assignments.items():
        set_nested(config, path, value)
    return config


def grid_iterator(choices: List[Tuple[Tuple[str, ...], List[Any]]]) -> Iterable[Dict[Tuple[str, ...], Any]]:
    if not choices:
        yield {}
        return
    paths, values = zip(*choices)
    for combination in itertools.product(*values):
        yield {path: value for path, value in zip(paths, combination)}


def random_iterator(
    choices: List[Tuple[Tuple[str, ...], List[Any]]],
    rng: random.Random,
    num_samples: int,
) -> Iterable[Dict[Tuple[str, ...], Any]]:
    for _ in range(num_samples):
        yield {path: rng.choice(options) for path, options in choices}


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train_single_trial(
    trial_index: int,
    config: Dict[str, Any],
    dataset: Dict[str, np.ndarray],
    force_cpu: bool,
    log_dir: Path,
) -> Tuple[float | None, float | None, float | None, float | None]:
    nn_model_cfg = config.get("nn_model", {})
    model_name = nn_model_cfg.get("architecture", "lstm_mlp")
    model_params = nn_model_cfg.get("params", {}) or {}
    lightning_params = nn_model_cfg.get("lightning", {}) or {}
    trainer_params = nn_model_cfg.get("trainer", {}) or {}

    batch_size = int(config.get("batch_size", 4096))
    epochs = int(config.get("epochs", 10))
    num_workers = int(config.get("num_workers", 4))

    x_cat_train = dataset.get("x_cat_train")
    x_cat_val = dataset.get("x_cat_val")
    x_cat_test = dataset.get("x_cat_test")
    if x_cat_train is None:
        x_cat_train = np.zeros((dataset["x_dyn_train"].shape[0], 0), dtype=np.int64)
    if x_cat_val is None:
        x_cat_val = np.zeros((dataset["x_dyn_val"].shape[0], 0), dtype=np.int64)
    if x_cat_test is None:
        x_cat_test = np.zeros((dataset.get("x_dyn_test", np.zeros((0, 1, 1))).shape[0], x_cat_train.shape[1]), dtype=np.int64)

    # Disable Lightning loggers unless explicitly provided.
    trainer_params = dict(trainer_params)
    trainer_params.setdefault("enable_progress_bar", False)
    if "logger" not in trainer_params or trainer_params["logger"] in (True, None):
        ensure_output_dir(log_dir)
        trainer_params["logger"] = CSVLogger(
            save_dir=str(log_dir),
            name=f"trial_{trial_index:03d}",
        )
    if force_cpu:
        trainer_params["accelerator"] = "cpu"
        trainer_params["devices"] = 1

    selection_metric = str(config.get("selection_metric", "sel_ap"))
    results = train_pipeline(
        dataset["x_dyn_train"],
        dataset["x_stat_train"],
        dataset["y_train"],
        dataset["x_dyn_val"],
        dataset["x_stat_val"],
        dataset["y_val"],
        x_cat_train=x_cat_train,
        x_cat_val=x_cat_val,
        x_cat_test=x_cat_test,
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
        model_path=None,
        plot_path=None,
        model_name=model_name,
        model_config=model_params,
        lightning_config=lightning_params,
        trainer_kwargs=trainer_params,
        selection_metric=selection_metric,
    )

    history = results.get("history", {})
    val_ap = results.get("val_ap")
    train_ap = results.get("train_ap")
    sel_ap = results.get("sel_ap")
    val_loss = None
    if isinstance(history, dict):
        val_losses = history.get("val_loss") or []
        if val_losses:
            val_loss = float(val_losses[-1])
    return val_ap, train_ap, sel_ap, val_loss


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    base_config_path = Path(args.config).expanduser().resolve()
    if not base_config_path.exists():
        print(f"Config file not found: {base_config_path}", file=sys.stderr)
        return 1

    base_config = load_yaml(base_config_path)
    search_space_data = DEFAULT_SEARCH_SPACE
    if args.search_space:
        search_space_path = Path(args.search_space).expanduser().resolve()
        if not search_space_path.exists():
            print(f"Search space YAML not found: {search_space_path}", file=sys.stderr)
            return 1
        custom_space = load_yaml(search_space_path)
        if "search_space" in custom_space:
            custom_space = custom_space["search_space"]
        if not isinstance(custom_space, dict):
            print("Search space YAML must define a mapping.", file=sys.stderr)
            return 1
        search_space_data = custom_space

    choices = flatten_search_space(search_space_data)
    if not choices:
        print("Search space is empty; nothing to do.", file=sys.stderr)
        return 1

    dataset_path = resolve_data_path(args, base_config)
    try:
        dataset = load_dataset(dataset_path)
    except (FileNotFoundError, KeyError) as exc:
        print(f"Failed to load dataset: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)
    trials_csv_path = output_dir / "trials.csv"
    best_config_path = output_dir / "best_config.yaml"
    logs_dir = output_dir / "lightning_logs"

    pl.seed_everything(args.seed, workers=True)

    rng = random.Random(args.seed)
    results: List[TrialResult] = []
    best_sel = float("-inf")
    best_config = None
    best_metric_label = "sel_ap"

    if args.strategy == "grid":
        iterator = grid_iterator(choices)
    else:
        iterator = random_iterator(choices, rng, args.max_trials)

    for trial_index, assignments in enumerate(iterator, start=1):
        if args.strategy == "grid" and trial_index > args.max_trials:
            break
        if trial_index > args.max_trials:
            break

        trial_assignments_str = {".".join(path): value for path, value in assignments.items()}
        print(f"=== Trial {trial_index}/{args.max_trials} ===")
        for param, value in sorted(trial_assignments_str.items()):
            print(f"  {param}: {value}")

        trial_config = assign_parameters(base_config, assignments)

        pl.seed_everything(args.seed + trial_index, workers=True)
        start_time = time.time()
        status = "ok"
        val_ap = None
        train_ap = None
        sel_ap = None
        val_loss = None
        error_text = None

        try:
            val_ap, train_ap, sel_ap, val_loss = train_single_trial(
                trial_index,
                trial_config,
                dataset,
                force_cpu=args.disable_cuda,
                log_dir=logs_dir,
            )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            print(f"Trial {trial_index} failed: {error_text}", file=sys.stderr)
            if args.fail_fast:
                raise
        finally:
            duration = time.time() - start_time
            if (not args.disable_cuda) and torch.cuda.is_available():
                torch.cuda.empty_cache()

        trial_result = TrialResult(
            index=trial_index,
            assignments=trial_assignments_str,
            status=status,
            duration_sec=duration,
            val_ap=float(val_ap) if val_ap is not None else None,
            train_ap=float(train_ap) if train_ap is not None else None,
            sel_ap=float(sel_ap) if sel_ap is not None else None,
            val_loss=float(val_loss) if val_loss is not None else None,
            error=error_text,
        )
        results.append(trial_result)

        # Selection metric from config: 'sel_ap' (default) or 'val_ap'
        sel_metric_cfg = str(trial_config.get("selection_metric", "sel_ap")).lower()
        if sel_metric_cfg in {"val_ap", "ap", "validation_ap"}:
            metric_for_selection = val_ap
        else:
            metric_for_selection = sel_ap if sel_ap is not None else val_ap
        if status == "ok" and metric_for_selection is not None and float(metric_for_selection) > best_sel:
            best_sel = float(metric_for_selection)
            best_config = trial_config
            best_metric_label = "val_ap" if sel_metric_cfg in {"val_ap", "ap", "validation_ap"} else "sel_ap"
            if sel_metric_cfg in {"val_ap", "ap", "validation_ap"}:
                print(f"  -> New best val_ap: {best_sel:.4f}")
            else:
                if sel_ap is not None:
                    print(f"  -> New best sel_ap (train+val): {best_sel:.4f}")
                else:
                    print(f"  -> New best val_ap: {best_sel:.4f} (fallback)")
        elif status == "ok" and (sel_ap is not None or val_ap is not None):
            if sel_metric_cfg in {"val_ap", "ap", "validation_ap"}:
                print(f"  -> val_ap: {val_ap:.4f}")
            else:
                if sel_ap is not None:
                    print(f"  -> sel_ap (train+val): {sel_ap:.4f}")
                else:
                    print(f"  -> val_ap: {val_ap:.4f}")

        if status == "failed" and args.fail_fast:
            break

    if not results:
        print("No trials executed.", file=sys.stderr)
        return 1

    # Persist trial outcomes.
    fieldnames = sorted({key for result in results for key in result.to_row().keys()})
    with trials_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_row())
    print(f"Wrote trial summary to {trials_csv_path}")

    if best_config is not None:
        with best_config_path.open("w") as f:
            yaml.safe_dump(best_config, f, sort_keys=False)
        print(f"Best configuration saved to {best_config_path} ({best_metric_label}={best_sel:.4f})")
    else:
        print("No successful trials; best configuration not saved.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
