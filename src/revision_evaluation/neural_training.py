"""Run the global full-data LSTM experiment set sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    "minimal_mlp": Path("configs/nn_global_full_minimal_mlp.yaml"),
    "ft_transformer": Path("configs/nn_global_full_ft_transformer.yaml"),
    "tsn": Path("configs/nn_global_full_tsn.yaml"),
    "lstm_static_concat": Path("configs/nn_global_full_lstm_static_concat.yaml"),
    "lstm_attention": Path("configs/nn_global_full_lstm_attention.yaml"),
    "lstm_gated_moe": Path("configs/nn_global_full_lstm_gated_moe.yaml"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=sorted(EXPERIMENTS),
        action="append",
        help="Run only selected experiment(s). Repeat to select multiple.",
    )
    parser.add_argument("--data-path", help="Override prepared_data.npz path for every run.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected = args.only or list(EXPERIMENTS)

    for name in selected:
        config_path = EXPERIMENTS[name]
        command = [
            sys.executable,
            "src/neural_net/train_nn.py",
            "--config-path",
            str(config_path),
        ]
        if args.data_path:
            command.extend(["--data-path", args.data_path])

        print("\n=== Running", name, "===")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
