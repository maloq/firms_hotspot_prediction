from __future__ import annotations

import pandas as pd

from src.revision_evaluation.full_grid_evaluation import _upsert_csv


def test_upsert_csv_hydrates_and_syncs_organized_raw_jsonl(tmp_path):
    output_dir = tmp_path
    primary = output_dir / "primary_full_grid_calibrated"
    raw_dir = output_dir / "shared_artifacts" / "raw_tables_jsonl"
    schema_dir = output_dir / "shared_artifacts" / "raw_table_schemas"
    raw_dir.mkdir(parents=True)
    schema_dir.mkdir(parents=True)

    original = pd.DataFrame(
        [
            {"model_name": "CatBoost", "region": "global", "split": "test", "average_precision": 0.4},
        ]
    )
    raw_path = raw_dir / "primary_full_grid_calibrated_model_comparison.jsonl.gz"
    original.to_json(raw_path, orient="records", lines=True, compression="gzip")

    _upsert_csv(
        primary / "model_comparison.csv",
        pd.DataFrame(
            [
                {
                    "model_name": "TemporalConvNet embedding fusion (global full)",
                    "region": "global",
                    "split": "test",
                    "average_precision": 0.2,
                }
            ]
        ),
        ["model_name", "region", "split"],
    )

    live = pd.read_csv(primary / "model_comparison.csv")
    archived = pd.read_json(raw_path, orient="records", lines=True, compression="gzip")

    assert live["model_name"].tolist() == [
        "CatBoost",
        "TemporalConvNet embedding fusion (global full)",
    ]
    assert archived["model_name"].tolist() == live["model_name"].tolist()
    assert (schema_dir / "primary_full_grid_calibrated_model_comparison.schema.json").exists()
