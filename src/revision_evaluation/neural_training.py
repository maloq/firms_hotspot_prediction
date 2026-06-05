"""Run the global full-data neural experiment set from revision config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import EvaluationConfig


DEFAULT_CONFIG = Path("configs/revision_evaluation_all_models_with_nns.yaml")

EXPERIMENTS = {
    "minimal_mlp": Path("configs/nn_global_full_minimal_mlp.yaml"),
    "minimal_mlp_fullgrid_opt": Path("configs/nn_global_full_minimal_mlp_fullgrid_opt.yaml"),
    "minimal_mlp_fullgrid_rank_opt": Path("configs/nn_global_full_minimal_mlp_fullgrid_rank_opt.yaml"),
    "ft_transformer": Path("configs/nn_global_full_ft_transformer.yaml"),
    "tsn": Path("configs/nn_global_full_tsn.yaml"),
    "tsn_embedding_fusion": Path("configs/nn_global_full_tsn_embedding_fusion.yaml"),
    "spatial_tsn": Path("configs/nn_global_full_spatial_tsn.yaml"),
    "spatial_tsn_embedding_fusion": Path("configs/nn_global_full_spatial_tsn_embedding_fusion.yaml"),
    "spatial_tsn_no_tp": Path("configs/nn_global_full_spatial_tsn_no_tp.yaml"),
    "spatial_tsn_ecmwf": Path("configs/nn_global_full_spatial_tsn_ecmwf.yaml"),
    "lstm_static_concat": Path("configs/nn_global_full_lstm_static_concat.yaml"),
    "lstm_embedding_fusion": Path("configs/nn_global_full_lstm_embedding_fusion.yaml"),
    "lstm_attention": Path("configs/nn_global_full_lstm_attention.yaml"),
    "lstm_gated_moe": Path("configs/nn_global_full_lstm_gated_moe.yaml"),
}

FEATURE_ABLATIONS = {
    "no_dynamic_sequence": {
        "label": "No dynamic weather sequence",
        "zero_dynamic": True,
        "zero_static": False,
        "zero_categorical": False,
    },
    "no_static_features": {
        "label": "No static features",
        "zero_dynamic": False,
        "zero_static": True,
        "zero_categorical": False,
    },
    "no_categorical_features": {
        "label": "No categorical features",
        "zero_dynamic": False,
        "zero_static": False,
        "zero_categorical": True,
    },
    "dynamic_sequence_only": {
        "label": "Dynamic sequence only",
        "zero_dynamic": False,
        "zero_static": True,
        "zero_categorical": True,
    },
}


@dataclass(frozen=True)
class NeuralTrainingConfig:
    models: list[str]
    output_dir: Path
    python: str = sys.executable
    data_path: Path | None = None
    dry_run: bool = False
    skip_existing_nn_models: bool = False
    run_legacy_sampled_evaluation: bool = True
    run_full_grid_evaluation: bool = True
    calibration_method: str = "platt_month"
    run_feature_ablation: bool = False
    feature_ablation_model: str = "tsn"
    feature_ablation_variants: list[str] | None = None
    parallel_jobs: int | str = 1
    parallel_devices: list[str] | str | None = "auto"


@dataclass(frozen=True)
class NeuralTrainingTask:
    name: str
    config_path: Path
    model_path: Path | None = None
    plot_path: Path | None = None
    metrics_path: Path | None = None


def neural_training_config(config: EvaluationConfig) -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        models=list(config.new_nn_models),
        output_dir=config.output_dir,
        python=config.python,
        data_path=config.nn_data_path,
        dry_run=config.nn_dry_run,
        skip_existing_nn_models=config.skip_existing_nn_models,
        run_legacy_sampled_evaluation=config.run_legacy_sampled_evaluation,
        run_full_grid_evaluation=config.run_full_grid_evaluation,
        calibration_method=config.calibration_method,
        run_feature_ablation=config.run_neural_feature_ablation,
        feature_ablation_model=config.neural_feature_ablation_model or config.main_nn_model,
        feature_ablation_variants=config.neural_feature_ablation_variants,
        parallel_jobs=config.nn_parallel_jobs,
        parallel_devices=config.nn_parallel_devices,
    )


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _update_metrics_payload(
    *,
    config_path: Path,
    output_dir: Path,
    run_legacy_sampled_evaluation: bool,
    run_full_grid_evaluation: bool,
    calibration_method: str,
) -> None:
    cfg = _load_yaml(config_path)
    metrics_path = Path(cfg.get("nn_metrics_path", ""))
    if not metrics_path.exists():
        return
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if run_legacy_sampled_evaluation:
        payload.setdefault(
            "legacy_sampled_metrics",
            {
                "train": payload.get("train"),
                "validation": payload.get("validation"),
                "test": payload.get("test"),
                "evaluation_type": "legacy_sampled_case_control",
                "note": "Undersampled/case-control neural evaluation retained as a legacy diagnostic.",
            },
        )
    if run_full_grid_evaluation:
        payload["primary_full_grid_calibrated_metrics"] = {
            "status": "adapter_required",
            "evaluation_type": "primary_full_grid_calibrated",
            "is_primary": False,
            "calibration_method": calibration_method,
            "note": (
                "Full-grid calibrated NN evaluation is part of revision_evaluation, "
                "but this architecture requires a deployment-grid tensor adapter "
                "before it can generate calibrated deployment predictions. "
                "Legacy sampled logits are saved separately and are not reported "
                "as deployment probabilities."
            ),
        }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out = output_dir / "neural_model_metrics"
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{metrics_path.parent.name}_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_one_training(
    *,
    python: str,
    config_path: Path,
    data_path: Path | None,
    dry_run: bool,
    model_path: Path | None = None,
    plot_path: Path | None = None,
    metrics_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    command = [python, "src/neural_net/train_nn.py", "--config-path", str(config_path)]
    if data_path is not None:
        command.extend(["--data-path", str(data_path)])
    if model_path is not None:
        command.extend(["--model-path", str(model_path)])
    if plot_path is not None:
        command.extend(["--plot-path", str(plot_path)])
    if metrics_path is not None:
        command.extend(["--metrics-path", str(metrics_path)])
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _training_artifacts_complete(config_path: Path, model_path: Path | None = None, metrics_path: Path | None = None) -> bool:
    cfg = _load_yaml(config_path)
    raw_metrics_path = metrics_path or cfg.get("nn_metrics_path")
    raw_model_path = model_path or cfg.get("nn_output_model_path")
    if not raw_metrics_path:
        return False
    resolved_metrics_path = Path(raw_metrics_path)
    if not resolved_metrics_path.exists():
        return False

    payload = _read_json_if_exists(resolved_metrics_path)
    payload_model_path = payload.get("model_path")
    if payload_model_path and Path(payload_model_path).exists():
        return True

    if raw_model_path and Path(raw_model_path).exists():
        return True

    if raw_model_path:
        configured = Path(raw_model_path)
        candidates = sorted(configured.parent.glob(f"{configured.stem}-*.ckpt"))
        if candidates:
            return True

    return False


def build_feature_ablation_config(
    *,
    base_config_path: Path,
    output_dir: Path,
    model_name: str,
    variant: str,
    spec: dict,
) -> tuple[Path, Path, Path, Path]:
    generated_dir = output_dir / "shared_artifacts" / "generated_configs"
    slug = f"nn_feature_ablation_{model_name}_{variant}"
    metrics_path = Path("outputs") / slug / "metrics.json"
    model_path = Path("models") / f"{slug}.ckpt"
    plot_path = Path("outputs") / "plots_nn" / slug
    payload = _load_yaml(base_config_path)
    payload["nn_metrics_path"] = str(metrics_path)
    payload["nn_output_model_path"] = str(model_path)
    payload["output_nn_plots_dir"] = str(plot_path)
    payload["input_ablation"] = {
        "enabled": True,
        "family": "neural_feature_ablation",
        "base_model": model_name,
        "variant": variant,
        **spec,
    }
    config_path = generated_dir / f"{slug}.yaml"
    _write_yaml(config_path, payload)
    return config_path, model_path, plot_path, metrics_path


def build_neural_feature_ablation_tasks(config: NeuralTrainingConfig) -> list[NeuralTrainingTask]:
    model_name = config.feature_ablation_model
    if model_name not in EXPERIMENTS:
        raise ValueError(f"Unknown neural feature-ablation model: {model_name}")
    variants = config.feature_ablation_variants or list(FEATURE_ABLATIONS)
    unknown = sorted(set(variants) - set(FEATURE_ABLATIONS))
    if unknown:
        raise ValueError(f"Unknown neural feature-ablation variant(s): {unknown}")

    base_config_path = EXPERIMENTS[model_name]
    tasks: list[NeuralTrainingTask] = []
    for variant in variants:
        spec = FEATURE_ABLATIONS[variant]
        ablation_config, model_path, plot_path, metrics_path = build_feature_ablation_config(
            base_config_path=base_config_path,
            output_dir=config.output_dir,
            model_name=model_name,
            variant=variant,
            spec=spec,
        )
        if config.skip_existing_nn_models and _training_artifacts_complete(
            ablation_config,
            model_path=model_path,
            metrics_path=metrics_path,
        ):
            print(f"\n=== Skipping existing neural feature ablation {model_name}/{variant} ===")
            continue
        tasks.append(
            NeuralTrainingTask(
                name=f"neural feature ablation {model_name}/{variant}",
                config_path=ablation_config,
                model_path=model_path,
                plot_path=plot_path,
                metrics_path=metrics_path,
            )
        )
    return tasks


def _visible_cuda_devices() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        raw = raw.strip()
        if not raw or raw == "-1":
            return []
        return [part.strip() for part in raw.split(",") if part.strip()]
    try:
        import torch

        return [str(idx) for idx in range(torch.cuda.device_count())]
    except Exception:
        return []


def _resolve_parallel_devices(config: NeuralTrainingConfig) -> list[str]:
    value = config.parallel_devices
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "cpu", "off"}:
            return []
        if normalized == "auto":
            return _visible_cuda_devices()
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _resolve_parallel_jobs(config: NeuralTrainingConfig, devices: list[str]) -> int:
    requested = config.parallel_jobs
    if isinstance(requested, str):
        normalized = requested.strip().lower()
        if normalized == "auto":
            return max(1, len(devices))
        if not normalized:
            return 1
        requested_value = int(normalized)
    else:
        requested_value = int(requested)
    return max(1, requested_value)


def _training_env(device: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("REVISION_EVALUATION_FAST_NN", "1")
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env


def _run_training_task(
    task: NeuralTrainingTask,
    config: NeuralTrainingConfig,
    *,
    device: str | None,
) -> None:
    device_note = f" [CUDA_VISIBLE_DEVICES={device}]" if device is not None else ""
    print(f"\n=== Running {task.name}{device_note} ===", flush=True)
    run_one_training(
        python=config.python,
        config_path=task.config_path,
        data_path=config.data_path,
        dry_run=config.dry_run,
        model_path=task.model_path,
        plot_path=task.plot_path,
        metrics_path=task.metrics_path,
        env=_training_env(device),
    )
    if not config.dry_run:
        _update_metrics_payload(
            config_path=task.config_path,
            output_dir=config.output_dir,
            run_legacy_sampled_evaluation=config.run_legacy_sampled_evaluation,
            run_full_grid_evaluation=config.run_full_grid_evaluation,
            calibration_method=config.calibration_method,
        )


def run_neural_training_tasks(tasks: list[NeuralTrainingTask], config: NeuralTrainingConfig) -> None:
    if not tasks:
        return
    devices = _resolve_parallel_devices(config)
    jobs = _resolve_parallel_jobs(config, devices)
    if jobs <= 1 or len(tasks) <= 1:
        for task in tasks:
            _run_training_task(task, config, device=devices[0] if devices else None)
        return

    print(
        f"\n=== Running {len(tasks)} neural training task(s) with {jobs} worker(s)"
        f"{' on devices ' + ','.join(devices) if devices else ''} ===",
        flush=True,
    )
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = []
        for index, task in enumerate(tasks):
            device = devices[index % len(devices)] if devices else None
            futures.append(executor.submit(_run_training_task, task, config, device=device))
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as exc:
                errors.append(exc)
                print(f"Neural training task failed: {exc}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} neural training task(s) failed.") from errors[0]


def run_neural_feature_ablation(config: NeuralTrainingConfig) -> None:
    run_neural_training_tasks(build_neural_feature_ablation_tasks(config), config)


def build_neural_training_tasks(config: NeuralTrainingConfig) -> list[NeuralTrainingTask]:
    selected = config.models or list(EXPERIMENTS)
    unknown = sorted(set(selected) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown neural model(s): {unknown}")

    tasks: list[NeuralTrainingTask] = []
    for name in selected:
        config_path = EXPERIMENTS[name]
        if config.skip_existing_nn_models and _training_artifacts_complete(config_path):
            print(f"\n=== Skipping existing {name} ===")
            continue
        tasks.append(NeuralTrainingTask(name=name, config_path=config_path))
    if config.run_feature_ablation:
        tasks.extend(build_neural_feature_ablation_tasks(config))
    return tasks


def run_neural_training(config: NeuralTrainingConfig) -> None:
    run_neural_training_tasks(build_neural_training_tasks(config), config)


def run_from_evaluation_config(config: EvaluationConfig) -> None:
    run_neural_training(neural_training_config(config))
    if not config.nn_dry_run and config.import_nn_metrics:
        from .neural_metrics import import_neural_metrics

        import_neural_metrics(config, refresh_main_plots=False)


def main(
    config_path: Path = DEFAULT_CONFIG,
    *,
    models: list[str] | None = None,
    output_dir: Path | None = None,
    skip_existing: bool | None = None,
    dry_run: bool | None = None,
) -> int:
    config = EvaluationConfig.from_yaml(config_path)
    if models:
        config.new_nn_models = list(models)
    if output_dir is not None:
        config.output_dir = output_dir
    if skip_existing is not None:
        config.skip_existing_nn_models = bool(skip_existing)
    if dry_run is not None:
        config.nn_dry_run = bool(dry_run)
    run_from_evaluation_config(config)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Run only these registered neural model keys, e.g. tsn_embedding_fusion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the evaluation output directory used for metric import and organized tables.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose configured checkpoint/metrics artifacts already exist.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print training commands without running them.")
    args = parser.parse_args()
    raise SystemExit(
        main(
            args.config,
            models=args.models,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing if args.skip_existing else None,
            dry_run=args.dry_run if args.dry_run else None,
        )
    )
