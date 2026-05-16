from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neural_net.grouped_thresholds import (
    apply_threshold_policy,
    evaluate_threshold_policy,
    fit_threshold_policy,
    json_default,
    load_metrics_payload,
    load_prediction_frames,
    load_regions,
    resolve_data_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune neural-network decision thresholds by season, region, or region-season "
            "using validation predictions, then apply the policy to validation/test artifacts."
        )
    )
    parser.add_argument(
        "--metrics-path",
        required=True,
        type=Path,
        help="Path to neural metrics.json containing prediction_artifacts and data_path.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Prepared NN .npz path. Defaults to data_path from metrics.json.",
    )
    parser.add_argument(
        "--regions-file",
        type=Path,
        default=Path("configs/regions_example.yaml"),
        help="YAML file with region bounds. Unmatched rows are assigned to region 'other'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <metrics-dir>/grouped_thresholding.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["season", "region", "region_season"],
        choices=["global", "season", "region", "region_season"],
        help="Threshold grouping strategies to evaluate.",
    )
    parser.add_argument("--min-val-rows", type=int, default=1000)
    parser.add_argument("--min-val-positives", type=int, default=20)
    parser.add_argument("--min-val-negatives", type=int, default=20)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument(
        "--update-metrics",
        action="store_true",
        help="Append grouped_thresholding artifact references and global metrics to metrics.json.",
    )
    return parser.parse_args()


def write_predictions(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics_path
    payload = load_metrics_payload(metrics_path)
    data_path = resolve_data_path(payload, metrics_path, args.data_path)
    output_dir = args.output_dir or (metrics_path.parent / "grouped_thresholding")
    regions = load_regions(args.regions_file)

    print(f"Loading prediction artifacts from {metrics_path}", flush=True)
    frames = load_prediction_frames(payload, data_path, regions)
    print(
        "Loaded splits: "
        + ", ".join(f"{name}={len(frame):,}" for name, frame in frames.items()),
        flush=True,
    )

    summary: dict[str, object] = {
        "metrics_path": str(metrics_path),
        "data_path": str(data_path),
        "regions_file": str(args.regions_file),
        "strategies": {},
    }
    metrics_update: dict[str, object] = {}

    for strategy in args.strategies:
        print(f"Tuning {strategy} thresholds on validation split", flush=True)
        policy = fit_threshold_policy(
            frames["validation"],
            strategy=strategy,
            min_val_rows=args.min_val_rows,
            min_val_positives=args.min_val_positives,
            min_val_negatives=args.min_val_negatives,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )

        strategy_dir = output_dir / strategy
        threshold_path = strategy_dir / "thresholds.json"
        write_json(threshold_path, policy)

        split_metrics: dict[str, object] = {}
        group_frames: list[pd.DataFrame] = []
        prediction_paths: dict[str, str] = {}
        for split_name, frame in frames.items():
            adjusted = apply_threshold_policy(frame, policy)
            prediction_path = strategy_dir / f"{split_name}_predictions.parquet"
            write_predictions(prediction_path, adjusted)
            prediction_paths[split_name] = str(prediction_path)

            overall, by_group = evaluate_threshold_policy(frame, policy, split_name=split_name)
            split_metrics[split_name] = overall
            group_frames.append(by_group)

        by_group_path = strategy_dir / "metrics_by_group.csv"
        pd.concat(group_frames, ignore_index=True).to_csv(by_group_path, index=False)
        metrics_path_out = strategy_dir / "metrics.json"
        metrics_payload = {
            "strategy": strategy,
            "thresholds_path": str(threshold_path),
            "prediction_artifacts": prediction_paths,
            "global_metrics": split_metrics,
            "metrics_by_group_path": str(by_group_path),
        }
        write_json(metrics_path_out, metrics_payload)

        summary["strategies"][strategy] = {
            "thresholds_path": str(threshold_path),
            "metrics_path": str(metrics_path_out),
            "metrics_by_group_path": str(by_group_path),
            "prediction_artifacts": prediction_paths,
            "validation": split_metrics["validation"],
            "test": split_metrics["test"],
        }
        metrics_update[strategy] = summary["strategies"][strategy]

        val_f1 = split_metrics["validation"].get("f1")
        test_f1 = split_metrics["test"].get("f1")
        print(f"{strategy}: validation F1={val_f1:.4f}, test F1={test_f1:.4f}", flush=True)

    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)

    if args.update_metrics:
        payload["grouped_thresholding"] = {
            "summary_path": str(summary_path),
            "strategies": metrics_update,
        }
        metrics_path.write_text(
            json.dumps(payload, indent=2, default=json_default),
            encoding="utf-8",
        )
        print(f"Updated {metrics_path} with grouped_thresholding artifact references", flush=True)

    print(f"Saved grouped threshold summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
