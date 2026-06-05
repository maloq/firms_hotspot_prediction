from __future__ import annotations

from pathlib import Path

from src.revision_evaluation.config import EvaluationConfig
from src.revision_evaluation import neural_metrics
from src.revision_evaluation.neural_training import (
    NeuralTrainingConfig,
    _resolve_parallel_jobs,
    build_neural_training_tasks,
    run_from_evaluation_config,
)


def test_build_neural_training_tasks_includes_generated_feature_ablation(tmp_path: Path) -> None:
    config = NeuralTrainingConfig(
        models=["minimal_mlp"],
        output_dir=tmp_path,
        dry_run=True,
        run_feature_ablation=True,
        feature_ablation_model="tsn",
        feature_ablation_variants=["no_static_features"],
    )

    tasks = build_neural_training_tasks(config)

    assert [task.name for task in tasks] == [
        "minimal_mlp",
        "neural feature ablation tsn/no_static_features",
    ]
    ablation_task = tasks[1]
    assert ablation_task.config_path == (
        tmp_path
        / "shared_artifacts"
        / "generated_configs"
        / "nn_feature_ablation_tsn_no_static_features.yaml"
    )
    assert ablation_task.config_path.exists()
    assert ablation_task.model_path == Path("models/nn_feature_ablation_tsn_no_static_features.ckpt")


def test_parallel_jobs_auto_uses_visible_device_count(tmp_path: Path) -> None:
    config = NeuralTrainingConfig(
        models=[],
        output_dir=tmp_path,
        parallel_jobs="auto",
        parallel_devices=["0", "1", "2"],
    )

    assert _resolve_parallel_jobs(config, ["0", "1", "2"]) == 3


def test_run_from_evaluation_config_imports_metrics_after_training(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    def fake_train(config: NeuralTrainingConfig) -> None:
        calls.append(("train", config.models))

    def fake_import(config: EvaluationConfig, **kwargs: object) -> None:
        calls.append(("import", (config.output_dir, kwargs)))

    monkeypatch.setattr("src.revision_evaluation.neural_training.run_neural_training", fake_train)
    monkeypatch.setattr(neural_metrics, "import_neural_metrics", fake_import)

    config = EvaluationConfig(
        output_dir=tmp_path,
        new_nn_models=["tsn_embedding_fusion"],
        import_nn_metrics=True,
    )

    run_from_evaluation_config(config)

    assert calls == [
        ("train", ["tsn_embedding_fusion"]),
        ("import", (tmp_path, {"refresh_main_plots": False})),
    ]
