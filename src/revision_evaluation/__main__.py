from __future__ import annotations

import argparse
from pathlib import Path

from .config import EvaluationConfig
from .workflow import run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete revision evaluation workflow.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/revision_evaluation_all_models.yaml"),
        help="Evaluation YAML config.",
    )
    args = parser.parse_args()
    run_evaluation(EvaluationConfig.from_yaml(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
