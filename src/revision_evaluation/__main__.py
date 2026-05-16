from __future__ import annotations

from pathlib import Path

from .config import EvaluationConfig
from .workflow import run_evaluation


DEFAULT_CONFIG = Path("configs/revision_evaluation_all_models_with_nns.yaml")


def main(config_path: Path = DEFAULT_CONFIG) -> int:
    run_evaluation(EvaluationConfig.from_yaml(config_path))
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the revision evaluation suite.")
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    raise SystemExit(main(args.config))
