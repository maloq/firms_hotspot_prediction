from __future__ import annotations

from .commands import (
    prepare_output_dir,
    run_followups,
    run_main_tabular,
    run_new_nn_models,
    run_organizer,
)
from .config import EvaluationConfig
from .neural_metrics import import_neural_metrics


def run_evaluation(config: EvaluationConfig) -> None:
    prepare_output_dir(config)

    if config.run_main_tabular:
        run_main_tabular(config)
    if config.run_followups:
        run_followups(config)
    if config.run_new_nn_models:
        run_new_nn_models(config)
    if config.import_nn_metrics:
        import_neural_metrics(config)
    if config.run_organizer:
        run_organizer(config)
