from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import EvaluationConfig, NN_LABELS
from .probability_overlays import load_neural_schema, neural_data_paths, safe_slug


MAIN_FEATURE_SET = "global full NN features"
TABLE_NAME = "neural_feature_importance.csv"


def run_neural_feature_importance(config: EvaluationConfig) -> None:
    output_dir = config.output_dir
    main = read_result_table(output_dir, "main_model_comparison.csv")
    if main.empty:
        raise FileNotFoundError(
            f"Cannot compute neural feature importance because main_model_comparison.csv "
            f"is missing under {output_dir} or shared_artifacts/raw_tables_jsonl."
        )

    exp_id, label = select_best_neural_model(main, output_dir=output_dir, config=config)
    metrics_path = find_metrics_file(output_dir, exp_id, config)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    prediction_path = find_prediction_file(output_dir, exp_id, payload)
    data_path = find_data_file(payload, prediction_path)
    model_path = find_model_file(output_dir, exp_id, payload)

    if config.neural_importance_device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.neural_importance_device

    x_dyn, x_static, x_cat, y_true = load_test_tensors(data_path, payload)
    x_dyn, x_static, x_cat, y_true = sample_tensors(
        x_dyn,
        x_static,
        x_cat,
        y_true,
        sample_size=int(config.neural_importance_sample_size),
        seed=int(config.seed),
    )
    if len(np.unique(y_true)) < 2:
        raise ValueError("Neural feature importance sample has only one target class.")

    from src.neural_net.models.lightning import SequenceStaticLightningModule

    model = SequenceStaticLightningModule.load_from_checkpoint(str(model_path), map_location=device)
    model.to(device)
    model.eval()

    probs = predict_probs(model, x_dyn, x_static, x_cat, batch_size=int(config.neural_importance_batch_size), device=device)
    baseline_ap = float(average_precision_score(y_true, probs))
    threshold, baseline_f1 = best_f1_threshold(y_true, probs)

    schema = load_neural_schema(
        output_dir,
        prediction_path,
        config.probability_overlay_dense_neural_training_features,
    )
    coordinate_columns = spatial_coordinate_columns(payload)
    if coordinate_columns:
        schema = dict(schema)
        static_columns = [str(col) for col in schema.get("static_columns", [])]
        if len(static_columns) + len(coordinate_columns) == x_static.shape[1]:
            schema["static_columns"] = [*static_columns, *coordinate_columns]
    dynamic_columns, static_columns, categorical_columns = feature_names_from_schema(schema, x_dyn, x_static, x_cat)

    rng = np.random.default_rng(int(config.seed))
    rows: list[dict[str, Any]] = []
    if x_dyn.ndim == 3:
        n_steps, n_channels = x_dyn.shape[1], x_dyn.shape[2]
        for step in range(n_steps):
            for channel in range(n_channels):
                flat_idx = step * n_channels + channel
                order = rng.permutation(len(x_dyn))
                original = x_dyn[:, step, channel].copy()
                x_dyn[:, step, channel] = original[order]
                try:
                    rows.append(
                        importance_row(
                            model=model,
                            x_dyn=x_dyn,
                            x_static=x_static,
                            x_cat=x_cat,
                            y_true=y_true,
                            batch_size=int(config.neural_importance_batch_size),
                            device=device,
                            model_id=exp_id,
                            model_label=label,
                            feature=dynamic_columns[flat_idx],
                            feature_type="dynamic_sequence",
                            baseline_ap=baseline_ap,
                            baseline_f1=baseline_f1,
                            threshold=threshold,
                        )
                    )
                finally:
                    x_dyn[:, step, channel] = original
    elif x_dyn.ndim == 5:
        for channel, feature in enumerate(dynamic_channel_names(dynamic_columns, x_dyn.shape[-1])):
            order = rng.permutation(len(x_dyn))
            original = x_dyn[..., channel].copy()
            x_dyn[..., channel] = original[order, ...]
            try:
                rows.append(
                    importance_row(
                        model=model,
                        x_dyn=x_dyn,
                        x_static=x_static,
                        x_cat=x_cat,
                        y_true=y_true,
                        batch_size=int(config.neural_importance_batch_size),
                        device=device,
                        model_id=exp_id,
                        model_label=label,
                        feature=feature,
                        feature_type="dynamic_spatial_channel",
                        baseline_ap=baseline_ap,
                        baseline_f1=baseline_f1,
                        threshold=threshold,
                    )
                )
            finally:
                x_dyn[..., channel] = original
    else:
        raise ValueError(f"Unsupported dynamic tensor shape for neural importance: {x_dyn.shape}")

    for idx, feature in enumerate(static_columns):
        order = rng.permutation(len(x_static))
        original = x_static[:, idx].copy()
        x_static[:, idx] = original[order]
        try:
            rows.append(
                importance_row(
                    model=model,
                    x_dyn=x_dyn,
                    x_static=x_static,
                    x_cat=x_cat,
                    y_true=y_true,
                    batch_size=int(config.neural_importance_batch_size),
                    device=device,
                    model_id=exp_id,
                    model_label=label,
                    feature=feature,
                    feature_type="static",
                    baseline_ap=baseline_ap,
                    baseline_f1=baseline_f1,
                    threshold=threshold,
                )
            )
        finally:
            x_static[:, idx] = original

    for idx, feature in enumerate(categorical_columns):
        order = rng.permutation(len(x_cat))
        original = x_cat[:, idx].copy()
        x_cat[:, idx] = original[order]
        try:
            rows.append(
                importance_row(
                    model=model,
                    x_dyn=x_dyn,
                    x_static=x_static,
                    x_cat=x_cat,
                    y_true=y_true,
                    batch_size=int(config.neural_importance_batch_size),
                    device=device,
                    model_id=exp_id,
                    model_label=label,
                    feature=feature,
                    feature_type="categorical",
                    baseline_ap=baseline_ap,
                    baseline_f1=baseline_f1,
                    threshold=threshold,
                )
            )
        finally:
            x_cat[:, idx] = original

    table = pd.DataFrame(rows)
    table = table.sort_values("delta_average_precision", ascending=False, na_position="last").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table["sample_size"] = int(len(y_true))
    table["positives"] = int(np.sum(y_true))
    table["metrics_path"] = str(metrics_path)
    table["model_path"] = str(model_path)
    table["prediction_path"] = str(prediction_path)
    table["data_path"] = str(data_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / TABLE_NAME, index=False)
    write_importance_plot(output_dir, table)
    print(
        f"Wrote neural feature importance for {label}: {len(table)} features, "
        f"sample_size={len(y_true)}, baseline PR-AUC={baseline_ap:.4f}.",
        flush=True,
    )


def read_result_table(output_dir: Path, name: str) -> pd.DataFrame:
    stem = Path(name).with_suffix("").name
    candidates = [
        output_dir / name,
        output_dir / "shared_artifacts" / "raw_tables_jsonl" / f"{stem}.jsonl.gz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".gz":
            return pd.read_json(path, orient="records", lines=True, compression="gzip")
        return pd.read_csv(path)
    return pd.DataFrame()


def select_best_neural_model(main: pd.DataFrame, *, output_dir: Path, config: EvaluationConfig) -> tuple[str, str]:
    label_to_key = {label: key for key, label in NN_LABELS.items()}
    work = main.copy()
    if "Region" in work.columns:
        work = work[work["Region"].astype(str).eq("Global")]
    if "Model" not in work.columns or "PR-AUC" not in work.columns:
        raise ValueError("main_model_comparison.csv must contain Model and PR-AUC columns.")
    work = work[work["Model"].astype(str).isin(label_to_key)]
    if work.empty:
        raise ValueError("main_model_comparison.csv does not contain global neural model rows.")
    work["PR-AUC"] = pd.to_numeric(work["PR-AUC"], errors="coerce")
    work = work.dropna(subset=["PR-AUC"]).sort_values("PR-AUC", ascending=False)
    if work.empty:
        raise ValueError("Global neural model rows have no finite PR-AUC values.")
    skipped: list[str] = []
    for _, row in work.iterrows():
        label = str(row["Model"])
        exp_id = f"nn_global_full_{label_to_key[label]}"
        try:
            metrics_path = find_metrics_file(output_dir, exp_id, config)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            prediction_path = find_prediction_file(output_dir, exp_id, payload)
            find_data_file(payload, prediction_path)
            return exp_id, label
        except Exception as exc:
            skipped.append(f"{label}: {exc}")
    raise ValueError(
        "No global neural model has a compatible checkpoint/prediction/data triplet for feature importance. "
        "Skipped candidates: " + "; ".join(skipped)
    )


def find_metrics_file(output_dir: Path, exp_id: str, config: EvaluationConfig) -> Path:
    candidates = [
        output_dir / "neural_model_metrics" / f"{exp_id}_metrics.json",
        output_dir / "shared_artifacts" / "neural_model_metrics" / f"{exp_id}_metrics.json",
    ]
    candidates.extend(Path(p) for p in sorted(glob.glob(config.nn_metrics_glob)) if Path(p).parent.name == exp_id)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find metrics JSON for {exp_id}.")


def find_prediction_file(output_dir: Path, exp_id: str, payload: dict[str, Any]) -> Path:
    candidates = [
        output_dir / "predictions" / f"{exp_id}_test_legacy_predictions.parquet",
        output_dir / "shared_artifacts" / "predictions" / f"{exp_id}_test_legacy_predictions.parquet",
    ]
    artifacts = payload.get("prediction_artifacts") or {}
    if artifacts.get("test"):
        candidates.append(Path(str(artifacts["test"])))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find test prediction parquet for {exp_id}.")


def find_data_file(payload: dict[str, Any], prediction_path: Path) -> Path:
    candidates: list[Path] = []
    if payload.get("data_path"):
        candidates.append(Path(str(payload["data_path"])))
    candidates.extend(neural_data_paths(prediction_path))
    expected_rows = expected_prediction_rows(prediction_path)
    for path in candidates:
        if importance_data_is_compatible(path, payload, expected_rows):
            return path
    raise FileNotFoundError(f"Could not find prepared NN tensor data for {prediction_path}.")


def expected_prediction_rows(prediction_path: Path) -> int | None:
    try:
        return int(len(pd.read_parquet(prediction_path, columns=["is_fire"])))
    except Exception:
        return None


def importance_data_is_compatible(path: Path, payload: dict[str, Any], expected_rows: int | None) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            if not {"x_dyn", "split"}.issubset(set(data.files)):
                return False
            split = np.asarray(data["split"], dtype=np.int8)
            test_rows = int(np.sum(split == 2))
            if expected_rows is not None and test_rows != int(expected_rows):
                return False
            x_dyn_ndim = int(data["x_dyn"].ndim)
    except Exception:
        return False

    architecture = str(payload.get("architecture") or "").lower()
    expects_spatial = "spatial" in architecture
    return x_dyn_ndim == 5 if expects_spatial else x_dyn_ndim == 3


def find_model_file(output_dir: Path, exp_id: str, payload: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    if payload.get("model_path"):
        candidates.append(Path(str(payload["model_path"])))
    candidates.extend(sorted((output_dir / "models").glob(f"{exp_id}*.ckpt")))
    candidates.extend(sorted((output_dir / "shared_artifacts" / "models").glob(f"{exp_id}*.ckpt")))
    candidates.extend(sorted(Path("models").glob(f"{exp_id}*.ckpt")))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find neural checkpoint for {exp_id}.")


def spatial_coordinate_columns(payload: dict[str, Any]) -> list[str]:
    coordinate_info = ((payload.get("spatial_training") or {}).get("coordinate_features") or {})
    if not coordinate_info.get("enabled", False):
        return []
    columns = coordinate_info.get("columns") or []
    return [str(col) for col in columns]


def build_spatial_coordinate_features(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat, dtype=np.float32).reshape(-1)
    lon = np.asarray(lon, dtype=np.float32).reshape(-1)
    lat_clean = np.nan_to_num(lat, nan=0.0, posinf=90.0, neginf=-90.0)
    lon_clean = np.nan_to_num(lon, nan=0.0, posinf=180.0, neginf=-180.0)
    lat_rad = np.deg2rad(lat_clean).astype(np.float32)
    lon_rad = np.deg2rad(lon_clean).astype(np.float32)
    return np.column_stack(
        [
            np.clip(lat_clean / 90.0, -1.0, 1.0),
            np.clip(lon_clean / 180.0, -1.0, 1.0),
            np.sin(lat_rad),
            np.cos(lat_rad),
            np.sin(lon_rad),
            np.cos(lon_rad),
        ]
    ).astype(np.float32)


def load_test_tensors(data_path: Path, payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(data_path) as data:
        required = {"x_dyn", "x_static", "x_cat", "y", "split"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{data_path} is missing required arrays: {', '.join(sorted(missing))}")
        mask = np.asarray(data["split"], dtype=np.int8) == 2
        if not mask.any():
            raise ValueError(f"{data_path} has no test split rows.")
        x_static = np.asarray(data["x_static"][mask], dtype=np.float32)
        if spatial_coordinate_columns(payload):
            if "lat" not in data.files or "lon" not in data.files:
                raise KeyError(f"{data_path} is missing lat/lon arrays required by spatial coordinate features.")
            coords = build_spatial_coordinate_features(data["lat"][mask], data["lon"][mask])
            x_static = np.concatenate([x_static, coords], axis=1).astype(np.float32)
        return (
            np.asarray(data["x_dyn"][mask], dtype=np.float32),
            x_static,
            np.asarray(data["x_cat"][mask], dtype=np.int64),
            (np.asarray(data["y"][mask]) > 0).astype(np.int8),
        )


def sample_tensors(
    x_dyn: np.ndarray,
    x_static: np.ndarray,
    x_cat: np.ndarray,
    y_true: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sample_size <= 0 or sample_size >= len(y_true):
        return x_dyn, x_static, x_cat, y_true
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y_true), size=sample_size, replace=False)
    indices.sort()
    return x_dyn[indices], x_static[indices], x_cat[indices], y_true[indices]


def feature_names_from_schema(
    schema: dict[str, Any],
    x_dyn: np.ndarray,
    x_static: np.ndarray,
    x_cat: np.ndarray,
) -> tuple[list[str], list[str], list[str]]:
    dynamic = [str(col) for col in schema.get("dynamic_source_columns", [])]
    static = [str(col) for col in schema.get("static_columns", [])]
    categorical = [str(col) for col in schema.get("categorical_columns", [])]

    expected_dynamic = int(np.prod(x_dyn.shape[1:]))
    if len(dynamic) != expected_dynamic:
        raise ValueError(f"Schema has {len(dynamic)} dynamic columns; tensor requires {expected_dynamic}.")
    if len(static) != x_static.shape[1]:
        raise ValueError(f"Schema has {len(static)} static columns; tensor requires {x_static.shape[1]}.")
    if len(categorical) != x_cat.shape[1]:
        raise ValueError(f"Schema has {len(categorical)} categorical columns; tensor requires {x_cat.shape[1]}.")
    return dynamic, static, categorical


def dynamic_channel_names(dynamic_columns: list[str], n_channels: int) -> list[str]:
    names: list[str] = []
    for column in dynamic_columns:
        name = str(column).split("_day_", 1)[0]
        if name not in names:
            names.append(name)
        if len(names) == n_channels:
            break
    if len(names) != n_channels:
        names = [f"dynamic_channel_{idx}" for idx in range(n_channels)]
    return [f"{name} spatial sequence" for name in names]


def predict_probs(model: Any, x_dyn: np.ndarray, x_static: np.ndarray, x_cat: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    import torch

    probs: list[np.ndarray] = []
    model.eval()
    batch_size = max(1, int(batch_size))
    with torch.inference_mode():
        for start in range(0, len(x_dyn), batch_size):
            end = min(start + batch_size, len(x_dyn))
            dyn = torch.as_tensor(x_dyn[start:end], dtype=torch.float32, device=device)
            stat = torch.as_tensor(x_static[start:end], dtype=torch.float32, device=device)
            cat = torch.as_tensor(x_cat[start:end], dtype=torch.long, device=device)
            logits = model(dyn, stat, cat if cat.numel() else None)
            probs.append(torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(probs).astype(np.float32) if probs else np.zeros((0,), dtype=np.float32)


def best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    f1 = f1[:-1]
    idx = int(np.nanargmax(f1)) if f1.size else 0
    threshold = float(thresholds[idx]) if thresholds.size else 0.5
    return threshold, float(f1_score(y_true, probs >= threshold, zero_division=0))


def importance_row(
    *,
    model: Any,
    x_dyn: np.ndarray,
    x_static: np.ndarray,
    x_cat: np.ndarray,
    y_true: np.ndarray,
    batch_size: int,
    device: str,
    model_id: str,
    model_label: str,
    feature: str,
    feature_type: str,
    baseline_ap: float,
    baseline_f1: float,
    threshold: float,
) -> dict[str, Any]:
    probs = predict_probs(model, x_dyn, x_static, x_cat, batch_size=batch_size, device=device)
    permuted_ap = float(average_precision_score(y_true, probs))
    permuted_f1 = float(f1_score(y_true, probs >= threshold, zero_division=0))
    return {
        "model_id": model_id,
        "model": model_label,
        "feature": feature,
        "feature_slug": safe_slug(feature),
        "feature_type": feature_type,
        "baseline_average_precision": baseline_ap,
        "permuted_average_precision": permuted_ap,
        "delta_average_precision": baseline_ap - permuted_ap,
        "baseline_f1": baseline_f1,
        "permuted_f1": permuted_f1,
        "delta_f1": baseline_f1 - permuted_f1,
        "threshold": threshold,
    }


def write_importance_plot(output_dir: Path, table: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    top = table.sort_values("delta_average_precision", ascending=False).head(30).iloc[::-1]
    if top.empty:
        return
    colors = top["feature_type"].map(
        {
            "dynamic_sequence": "#2563eb",
            "static": "#0f766e",
            "categorical": "#9333ea",
        }
    ).fillna("#6b7280")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(top) + 1.5)))
    ax.barh(top["feature"], top["delta_average_precision"], color=colors)
    ax.set_xlabel("PR-AUC drop after permutation")
    ax.set_title("Neural feature importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "neural_feature_importance_top30.png", dpi=240, bbox_inches="tight")
    fig.savefig(plot_dir / "neural_feature_importance_top30.pdf", bbox_inches="tight")
    plt.close(fig)
