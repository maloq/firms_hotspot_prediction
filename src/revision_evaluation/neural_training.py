"""Run the global full-data neural experiment set from revision config."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import EvaluationConfig


DEFAULT_CONFIG = Path("configs/revision_evaluation_all_models_with_nns.yaml")

EXPERIMENTS = {
    "minimal_mlp": Path("configs/nn_global_full_minimal_mlp.yaml"),
    "minimal_mlp_fullgrid_opt": Path("configs/nn_global_full_minimal_mlp_fullgrid_opt.yaml"),
    "minimal_mlp_fullgrid_rank_opt": Path("configs/nn_global_full_minimal_mlp_fullgrid_rank_opt.yaml"),
    "ft_transformer": Path("configs/nn_global_full_ft_transformer.yaml"),
    "tsn": Path("configs/nn_global_full_tsn.yaml"),
    "spatial_tsn": Path("configs/nn_global_full_spatial_tsn.yaml"),
    "spatial_tsn_no_tp": Path("configs/nn_global_full_spatial_tsn_no_tp.yaml"),
    "spatial_tsn_ecmwf": Path("configs/nn_global_full_spatial_tsn_ecmwf.yaml"),
    "lstm_static_concat": Path("configs/nn_global_full_lstm_static_concat.yaml"),
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
    run_legacy_sampled_evaluation: bool = True
    run_full_grid_evaluation: bool = True
    calibration_method: str = "platt_month"
    run_feature_ablation: bool = False
    feature_ablation_model: str = "tsn"
    feature_ablation_variants: list[str] | None = None


def neural_training_config(config: EvaluationConfig) -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        models=list(config.new_nn_models),
        output_dir=config.output_dir,
        python=config.python,
        data_path=config.nn_data_path,
        dry_run=config.nn_dry_run,
        run_legacy_sampled_evaluation=config.run_legacy_sampled_evaluation,
        run_full_grid_evaluation=config.run_full_grid_evaluation,
        calibration_method=config.calibration_method,
        run_feature_ablation=config.run_neural_feature_ablation,
        feature_ablation_model=config.neural_feature_ablation_model or config.main_nn_model,
        feature_ablation_variants=config.neural_feature_ablation_variants,
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
        subprocess.run(command, check=True)


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


def run_neural_feature_ablation(config: NeuralTrainingConfig) -> None:
    model_name = config.feature_ablation_model
    if model_name not in EXPERIMENTS:
        raise ValueError(f"Unknown neural feature-ablation model: {model_name}")
    variants = config.feature_ablation_variants or list(FEATURE_ABLATIONS)
    unknown = sorted(set(variants) - set(FEATURE_ABLATIONS))
    if unknown:
        raise ValueError(f"Unknown neural feature-ablation variant(s): {unknown}")

    base_config_path = EXPERIMENTS[model_name]
    for variant in variants:
        spec = FEATURE_ABLATIONS[variant]
        ablation_config, model_path, plot_path, metrics_path = build_feature_ablation_config(
            base_config_path=base_config_path,
            output_dir=config.output_dir,
            model_name=model_name,
            variant=variant,
            spec=spec,
        )
        print(f"\n=== Running neural feature ablation {model_name}/{variant} ===")
        run_one_training(
            python=config.python,
            config_path=ablation_config,
            data_path=config.data_path,
            dry_run=config.dry_run,
            model_path=model_path,
            plot_path=plot_path,
            metrics_path=metrics_path,
        )
        if not config.dry_run:
            _update_metrics_payload(
                config_path=ablation_config,
                output_dir=config.output_dir,
                run_legacy_sampled_evaluation=config.run_legacy_sampled_evaluation,
                run_full_grid_evaluation=config.run_full_grid_evaluation,
                calibration_method=config.calibration_method,
            )


def run_neural_training(config: NeuralTrainingConfig) -> None:
    selected = config.models or list(EXPERIMENTS)
    unknown = sorted(set(selected) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown neural model(s): {unknown}")

    for name in selected:
        config_path = EXPERIMENTS[name]
        print(f"\n=== Running {name} ===")
        run_one_training(
            python=config.python,
            config_path=config_path,
            data_path=config.data_path,
            dry_run=config.dry_run,
        )
        if not config.dry_run:
            _update_metrics_payload(
                config_path=config_path,
                output_dir=config.output_dir,
                run_legacy_sampled_evaluation=config.run_legacy_sampled_evaluation,
                run_full_grid_evaluation=config.run_full_grid_evaluation,
                calibration_method=config.calibration_method,
            )
    if config.run_feature_ablation:
        run_neural_feature_ablation(config)


def run_from_evaluation_config(config: EvaluationConfig) -> None:
    run_neural_training(neural_training_config(config))


def main(config_path: Path = DEFAULT_CONFIG) -> int:
    run_from_evaluation_config(EvaluationConfig.from_yaml(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
