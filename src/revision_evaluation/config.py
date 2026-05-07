from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ALL_NN_MODELS = [
    "minimal_mlp",
    "ft_transformer",
    "tsn",
    "lstm_static_concat",
    "lstm_attention",
    "lstm_gated_moe",
]

NN_LABELS = {
    "minimal_mlp": "Minimal MLP (global full)",
    "ft_transformer": "FT-Transformer (global full)",
    "tsn": "TemporalConvNet / TSN-MLP (global full)",
    "lstm_static_concat": "LSTM static concat (global full)",
    "lstm_attention": "LSTM attention (global full)",
    "lstm_gated_moe": "LSTM gated MoE (global full)",
}


@dataclass
class EvaluationConfig:
    output_dir: Path = Path("results/revision_experiments_complete")
    python: str = sys.executable
    overwrite_output_dir: bool = False

    run_main_tabular: bool = True
    run_followups: bool = True
    run_new_nn_models: bool = True
    import_nn_metrics: bool = True
    run_organizer: bool = True

    features_path: Path = Path("data/saved_features_boost/train_test_features_30d_all_extended_north_14cb.parquet")
    feature_config: Path = Path("configs/features_config_30d.yaml")
    target_config: Path = Path("configs/target_config.yaml")
    catboost_config: Path = Path("configs/catboost_train_config.yaml")
    regions_file: Path = Path("configs/regions_example.yaml")
    era5_dir: Path = Path("/home/ids/vmorozov/era5")

    catboost_iterations: int = 450
    followup_catboost_iterations: int = 260
    catboost_task_type: str = "GPU"
    random_error_trials: int = 5
    random_error_sample_size: int = 50_000
    permutation_trials: int = 5
    permutation_sample_size: int = 50_000
    new_nn_models: list[str] = field(default_factory=lambda: list(ALL_NN_MODELS))
    nn_metrics_glob: str = "outputs/nn_global_full_*/metrics.json"

    @classmethod
    def from_yaml(cls, path: Path) -> "EvaluationConfig":
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping in {path}")

        data: dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in raw:
                data[field_name] = raw[field_name]

        for key in [
            "output_dir",
            "features_path",
            "feature_config",
            "target_config",
            "catboost_config",
            "regions_file",
            "era5_dir",
        ]:
            if key in data and data[key] is not None:
                data[key] = Path(data[key])
        if data.get("python") in {None, ""}:
            data["python"] = sys.executable
        return cls(**data)
