from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import EvaluationConfig


def run_command(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def prepare_output_dir(config: EvaluationConfig) -> None:
    if config.overwrite_output_dir and config.output_dir.exists():
        shutil.rmtree(config.output_dir)


def run_main_tabular(config: EvaluationConfig) -> None:
    run_command(
        [
            config.python,
            "-m",
            "src.revision_evaluation.tabular",
            "--features-path",
            str(config.features_path),
            "--feature-config",
            str(config.feature_config),
            "--target-config",
            str(config.target_config),
            "--catboost-config",
            str(config.catboost_config),
            "--regions-file",
            str(config.regions_file),
            "--output-dir",
            str(config.output_dir),
            "--era5-dir",
            str(config.era5_dir),
            "--catboost-iterations",
            str(config.catboost_iterations),
            "--catboost-task-type",
            config.catboost_task_type,
        ]
    )


def run_followups(config: EvaluationConfig) -> None:
    run_command(
        [
            config.python,
            "-m",
            "src.revision_evaluation.followups",
            "--output-dir",
            str(config.output_dir),
            "--catboost-iterations",
            str(config.followup_catboost_iterations),
            "--catboost-task-type",
            config.catboost_task_type,
        ]
    )


def run_new_nn_models(config: EvaluationConfig) -> None:
    for model in config.new_nn_models:
        run_command([config.python, "-m", "src.revision_evaluation.neural_training", "--only", model])


def run_organizer(config: EvaluationConfig) -> None:
    run_command([config.python, "-m", "src.revision_evaluation.result_library", "--root", str(config.output_dir)])


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
