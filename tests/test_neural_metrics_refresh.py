from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.revision_evaluation.neural_metrics import refresh_neural_embedding_experiment_tables


def test_refresh_neural_embedding_experiment_tables_writes_global_table(tmp_path: Path) -> None:
    neural_table = pd.DataFrame(
        [
            {
                "experiment": "TemporalConvNet embedding fusion (global full)",
                "region": "global",
                "region_display": "Global",
                "period": "2021-2025",
                "f1": 0.51,
                "f1_error": 0.002,
                "average_precision": 0.46,
                "average_precision_error": 0.003,
                "threshold": 0.25,
            },
            {
                "experiment": "LSTM embedding fusion (global full)",
                "region": "global",
                "region_display": "Global",
                "period": "2021-2025",
                "f1": 0.49,
                "f1_error": 0.004,
                "average_precision": 0.45,
                "average_precision_error": 0.005,
                "threshold": 0.23,
            },
            {
                "experiment": "TemporalConvNet embedding fusion (global full)",
                "region": "central_asia",
                "region_display": "Central Asia",
                "period": "2021-2025",
                "f1": 0.30,
                "f1_error": 0.006,
                "average_precision": 0.22,
                "average_precision_error": 0.007,
                "threshold": 0.25,
            },
        ]
    )

    refresh_neural_embedding_experiment_tables(tmp_path, neural_table, pd.DataFrame())

    global_path = (
        tmp_path
        / "experiments"
        / "12_neural_embedding_fusion_global"
        / "tables"
        / "global_neural_fusion_metrics.csv"
    )
    regional_path = (
        tmp_path
        / "experiments"
        / "13_neural_embedding_fusion_by_region"
        / "tables"
        / "regional_neural_fusion_metrics.csv"
    )

    global_table = pd.read_csv(global_path)
    regional_table = pd.read_csv(regional_path)

    assert global_table["Variant"].tolist() == [
        "LSTM embedding fusion (global full)",
        "TemporalConvNet embedding fusion (global full)",
    ]
    assert "TemporalConvNet embedding fusion (global full)" in regional_table["Variant"].tolist()
