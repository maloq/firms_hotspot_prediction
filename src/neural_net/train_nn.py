import argparse
import json
import math
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from tqdm.auto import tqdm
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.neural_net.models.lightning import SequenceStaticLightningModule
from src.utils.prediction_adjustments import adjust_probabilities_for_prior

torch.set_float32_matmul_precision('high')

DEFAULT_SPATIAL_STATIC_PREFIXES = [
    "coord_",
    "elevation_",
    "road_",
    "distance_to_road",
    "night_light_",
    "light_density_",
    "distance_to_light",
    "fire_index_",
    "lai_",
    "population",
    "anor",
    "isor",
    "z",
    "lsm",
    "slor",
    "sdfor",
    "sdor",
]

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(f"GPU {idx}: {props.name}, memory {props.total_memory / (1024 ** 3):.1f} GB")


class FireDataset(Dataset):
    """Dataset that returns model inputs, hard labels, loss targets, and sample weights."""

    def __init__(self, x_dyn, x_stat, x_cat, y, sample_weight=None, loss_target=None):
        self.x_dyn = torch.as_tensor(x_dyn)
        if not torch.is_floating_point(self.x_dyn):
            self.x_dyn = self.x_dyn.float()
        self.x_stat = torch.as_tensor(x_stat, dtype=torch.float32)
        if x_cat is None:
            x_cat = np.zeros((self.x_dyn.shape[0], 0), dtype=np.int64)
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1)
        if loss_target is None:
            loss_target = y
        self.loss_target = torch.as_tensor(loss_target, dtype=torch.float32).view(-1)
        if sample_weight is None:
            self.sample_weight = torch.ones_like(self.y)
        else:
            self.sample_weight = torch.as_tensor(sample_weight, dtype=torch.float32).view(-1)

    def __len__(self):
        return self.x_dyn.shape[0]

    def __getitem__(self, idx):
        return (
            self.x_dyn[idx].float(),
            self.x_stat[idx],
            self.x_cat[idx],
            self.y[idx],
            self.loss_target[idx],
            self.sample_weight[idx],
        )

class HistoryCallback(pl.Callback):
    """Collects epoch-level metrics for later plotting."""

    def __init__(self):
        super().__init__()
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_ap": [],
            "val_f1": [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        if train_loss is not None:
            self.history["train_loss"].append(float(train_loss.cpu().item()))

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        if val_loss is not None:
            self.history["val_loss"].append(float(val_loss.cpu().item()))
        val_ap = metrics.get("val_ap")
        if val_ap is not None:
            value = float(val_ap.cpu().item()) if isinstance(val_ap, torch.Tensor) else float(val_ap)
            self.history["val_ap"].append(value)
        val_f1 = metrics.get("val_f1")
        if val_f1 is not None:
            value = float(val_f1.cpu().item()) if isinstance(val_f1, torch.Tensor) else float(val_f1)
            self.history["val_f1"].append(value)


def compute_class_weight(y_train):
    classes = np.unique(y_train)
    cw = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    sample_w = np.array([cw_dict[int(v)] for v in y_train], dtype=float)
    return cw_dict, sample_w


def neural_training_sample_weight(
    *,
    y_train: np.ndarray,
    class_weight_sample: np.ndarray,
    prepared_sample_weight: np.ndarray | None,
    config: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build optional deployment-distribution loss weights for NN training."""

    cfg = dict(config or {})
    base = np.asarray(class_weight_sample, dtype=np.float32).reshape(-1)
    info: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "used_sample_weight": False,
        "base_class_weight_mean": float(np.mean(base)) if base.size else None,
    }
    if not cfg.get("enabled", False):
        return base, info

    source = str(cfg.get("source", "sample_weight"))
    if source != "sample_weight":
        raise ValueError("training_sample_weight.source must be 'sample_weight'.")
    if prepared_sample_weight is None:
        raise ValueError(
            "training_sample_weight.enabled is true, but prepared_data.npz has no sample_weight array."
        )

    deploy = np.asarray(prepared_sample_weight, dtype=np.float32).reshape(-1)
    if deploy.shape[0] != base.shape[0]:
        raise ValueError(
            "training_sample_weight row mismatch: "
            f"sample_weight={deploy.shape[0]}, y_train={base.shape[0]}."
        )

    invalid = ~np.isfinite(deploy) | (deploy <= 0)
    if invalid.any():
        deploy = deploy.copy()
        deploy[invalid] = 1.0

    raw_min = float(np.min(deploy)) if deploy.size else None
    raw_max = float(np.max(deploy)) if deploy.size else None
    raw_mean = float(np.mean(deploy)) if deploy.size else None

    power = float(cfg.get("power", 1.0))
    if power <= 0.0:
        raise ValueError("training_sample_weight.power must be positive.")
    if power != 1.0:
        deploy = np.power(deploy, power, dtype=np.float32)

    cap_value = cfg.get("cap_value")
    cap_quantile = cfg.get("cap_quantile")
    if cap_value is None and cap_quantile is not None:
        q = float(cap_quantile)
        if not 0.0 < q <= 1.0:
            raise ValueError("training_sample_weight.cap_quantile must be in (0, 1].")
        cap_value = float(np.quantile(deploy, q))
    if cap_value is not None:
        cap = float(cap_value)
        if cap > 0.0 and np.isfinite(cap):
            deploy = np.minimum(deploy, cap).astype(np.float32)

    if bool(cfg.get("normalize", True)) and deploy.size:
        mean = float(np.mean(deploy))
        if np.isfinite(mean) and mean > 0.0:
            deploy = (deploy / mean).astype(np.float32)

    multiply_class_weights = bool(cfg.get("multiply_class_weights", True))
    combined = (base * deploy) if multiply_class_weights else deploy
    combined = np.asarray(combined, dtype=np.float32)
    combined[~np.isfinite(combined) | (combined <= 0)] = 1.0
    if bool(cfg.get("normalize_after_multiply", True)) and combined.size:
        mean = float(np.mean(combined))
        if np.isfinite(mean) and mean > 0.0:
            combined = (combined / mean).astype(np.float32)

    y_arr = np.asarray(y_train).reshape(-1)
    pos = y_arr == 1
    neg = y_arr == 0
    info.update(
        {
            "enabled": True,
            "used_sample_weight": True,
            "source": source,
            "power": power,
            "cap_quantile": None if cap_quantile is None else float(cap_quantile),
            "cap_value": None if cap_value is None else float(cap_value),
            "normalize": bool(cfg.get("normalize", True)),
            "multiply_class_weights": multiply_class_weights,
            "normalize_after_multiply": bool(cfg.get("normalize_after_multiply", True)),
            "invalid_weight_rows": int(invalid.sum()),
            "raw_weight_min": raw_min,
            "raw_weight_max": raw_max,
            "raw_weight_mean": raw_mean,
            "deployment_weight_mean": float(np.mean(deploy)) if deploy.size else None,
            "final_weight_min": float(np.min(combined)) if combined.size else None,
            "final_weight_max": float(np.max(combined)) if combined.size else None,
            "final_weight_mean": float(np.mean(combined)) if combined.size else None,
            "positive_final_weight_mean": float(np.mean(combined[pos])) if pos.any() else None,
            "negative_final_weight_mean": float(np.mean(combined[neg])) if neg.any() else None,
        }
    )
    return combined, info


def plot_loss(history, out_path=None):
    if hasattr(history, "history"):
        hist = history.history
    else:
        hist = history
    train_loss = hist.get("train_loss") or hist.get("loss", [])
    val_loss = hist.get("val_loss", [])
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.title("Loss")
    plt.legend()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.close()


def plot_recall_precision(y_true, y_scores, out_path=None, title="PR curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(12, 10))
    plt.step(recall, precision, where="post", label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AP={ap:.4f})")
    plt.legend()
    plt.grid(True)
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.close()
    return ap


def choose_threshold_f1(y_true, y_scores, sample_weight=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores, sample_weight=sample_weight)
    if thresholds.size == 0:
        return 0.5, None
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    f1 = f1[:-1]
    if f1.size == 0:
        return 0.5, None
    idx = np.nanargmax(f1)
    return float(thresholds[idx]), float(f1[idx])


def infer_categorical_embeddings(x_cat: np.ndarray) -> list[dict[str, int]]:
    """Infer compact embedding metadata from integer categorical IDs."""

    if x_cat is None or x_cat.ndim != 2 or x_cat.shape[1] == 0:
        return []

    embeddings: list[dict[str, int]] = []
    for idx in range(x_cat.shape[1]):
        values = x_cat[:, idx]
        cardinality = int(np.nanmax(values)) + 1 if values.size else 0
        cardinality = max(cardinality, 1)
        embedding_dim = min(16, max(2, int(round(cardinality ** 0.25 * 4)))) if cardinality > 1 else 1
        embeddings.append(
            {
                "name": f"cat_{idx}",
                "cardinality": cardinality,
                "embedding_dim": embedding_dim,
            }
        )
    return embeddings


def predict_logits(
    model,
    x_dyn,
    x_stat,
    x_cat=None,
    batch_size=256,
    device=None,
    show_progress: bool = False,
    desc: str | None = None,
):
    if len(x_dyn) == 0:
        return np.zeros((0,), dtype=np.float32)
    model.eval()
    device = device or next(model.parameters()).device
    if x_cat is None:
        x_cat = np.zeros((len(x_dyn), 0), dtype=np.int64)
    dataset = FireDataset(x_dyn, x_stat, x_cat, np.zeros(len(x_dyn), dtype=np.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            desc=desc or "Predict",
            dynamic_ncols=True,
            leave=True,
        )
    preds = []
    with torch.no_grad():
        for dyn, stat, cat, *_ in iterator:
            dyn = dyn.to(device)
            stat = stat.to(device)
            cat = cat.to(device)
            logits = model(dyn, stat, cat if cat.numel() else None)
            preds.append(logits.cpu().numpy())
    return np.concatenate(preds).reshape(-1)


def logits_to_probs(logits):
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))


def predict_probs(
    model,
    x_dyn,
    x_stat,
    x_cat=None,
    batch_size=256,
    device=None,
    show_progress: bool = False,
    desc: str | None = None,
):
    logits = predict_logits(
        model,
        x_dyn,
        x_stat,
        x_cat=x_cat,
        batch_size=batch_size,
        device=device,
        show_progress=show_progress,
        desc=desc,
    )
    return logits_to_probs(logits)


def save_sampled_prediction_table(path, y_true, logits, split_name):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    logits = np.asarray(logits, dtype=float).reshape(-1)
    prob = logits_to_probs(logits)
    frame = pd.DataFrame(
        {
            "split_name": split_name,
            "is_fire": y_true,
            "raw_score": logits.astype(np.float32),
            "prob_raw": prob.astype(np.float32),
            "raw_score_source": "neural_logit_before_sigmoid",
            "evaluation_type": "legacy_sampled_case_control",
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def train_pipeline(
    x_dyn_train,
    x_stat_train,
    y_train,
    x_dyn_val,
    x_stat_val,
    y_val,
    *,
    x_cat_train=None,
    x_cat_val=None,
    x_cat_test=None,
    batch_size=1024,
    epochs=50,
    model_path=None,
    plot_path=None,
    num_workers=4,
    model_name="lstm_mlp",
    model_config=None,
    lightning_config=None,
    trainer_kwargs=None,
    class_weights=None,
    prepared_sample_weight_train=None,
    training_sample_weight_config=None,
    selection_metric: str = "sel_ap",
    compute_train_predictions: bool = True,
    y_loss_train=None,
    y_loss_val=None,
    loss_config=None,
):
    n_train = x_dyn_train.shape[0]
    steps_per_epoch = math.ceil(n_train / batch_size)
    lr_decay_epochs = 5
    decay_steps = lr_decay_epochs * steps_per_epoch

    if x_cat_train is None:
        x_cat_train = np.zeros((x_dyn_train.shape[0], 0), dtype=np.int64)
    if x_cat_val is None:
        x_cat_val = np.zeros((x_dyn_val.shape[0], 0), dtype=np.int64)
    if x_cat_test is None:
        cat_dim = x_cat_train.shape[1] if x_cat_train.ndim == 2 else 0
        x_cat_test = np.zeros((0, cat_dim), dtype=np.int64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Handle class weights: use config if provided, otherwise calculate
    if class_weights is not None:
        # Config-provided weights
        cw_dict = dict(class_weights)
        sample_w_train = np.array([cw_dict.get(int(v), 1.0) for v in y_train], dtype=float)
        # Print in requested format
        weight_0 = cw_dict.get(0, 1.0)
        weight_1 = cw_dict.get(1, 1.0)
        print(f"Class weights (from config): 1:{weight_1/weight_0:.4f} [actual]")
        print(f"Class weights (calculated): 1:{cw_dict.get(1, 1.0)/cw_dict.get(0, 1.0):.4f}")
    else:
        # Calculate balanced weights
        cw_dict, sample_w_train = compute_class_weight(y_train)
        # Print in requested format
        weight_0 = 1
        weight_1 = 1
        print(f"Class weights default: 1:{weight_1/weight_0:.4f}")

    sample_w_train, training_sample_weight_info = neural_training_sample_weight(
        y_train=y_train,
        class_weight_sample=sample_w_train,
        prepared_sample_weight=prepared_sample_weight_train,
        config=training_sample_weight_config,
    )
    if training_sample_weight_info.get("used_sample_weight"):
        print("[training_sample_weight]", json.dumps(training_sample_weight_info, sort_keys=True))

    if y_loss_train is None:
        y_loss_train = y_train
    if y_loss_val is None:
        y_loss_val = y_val

    model_kwargs = dict(model_config or {})
    if x_dyn_train.ndim == 3:
        seq_len = int(x_dyn_train.shape[1])
        n_channels = int(x_dyn_train.shape[2])
    elif x_dyn_train.ndim == 5:
        seq_len = int(x_dyn_train.shape[1])
        n_channels = int(x_dyn_train.shape[4])
        model_kwargs.setdefault("spatial_height", int(x_dyn_train.shape[2]))
        model_kwargs.setdefault("spatial_width", int(x_dyn_train.shape[3]))
    else:
        raise ValueError(
            "Expected dynamic training tensor shaped (N, T, C) or (N, T, H, W, C), "
            f"got {x_dyn_train.shape}."
        )
    model_kwargs.setdefault("seq_len", seq_len)
    model_kwargs.setdefault("n_channels", n_channels)
    model_kwargs.setdefault("n_static", x_stat_train.shape[1])
    model_kwargs.setdefault("n_categorical", x_cat_train.shape[1] if x_cat_train.ndim == 2 else 0)
    if (
        str(model_kwargs.get("categorical_mode", "auto")).lower()
        in {"embedding", "embeddings", "learned_embedding", "learned_embeddings"}
        and not model_kwargs.get("categorical_embeddings")
        and model_kwargs["n_categorical"] > 0
    ):
        model_kwargs["categorical_embeddings"] = infer_categorical_embeddings(x_cat_train)
        print("Inferred categorical embeddings:", model_kwargs["categorical_embeddings"])
    print(f"Using architecture: {model_name} with params: {model_kwargs}")

    lightning_kwargs = dict(lightning_config or {})
    loss_config = dict(loss_config or {})
    if loss_config:
        loss_name = loss_config.get("name", loss_config.get("type"))
        if loss_name is not None:
            lightning_kwargs["loss_name"] = loss_name
        if "focal_gamma" in loss_config:
            lightning_kwargs["focal_gamma"] = loss_config["focal_gamma"]
        if "focal_alpha" in loss_config:
            lightning_kwargs["focal_alpha"] = loss_config["focal_alpha"]
    # Support alias `weight_decay` in config; map to module arg `l2`
    if "weight_decay" in lightning_kwargs and "l2" not in lightning_kwargs:
        try:
            lightning_kwargs["l2"] = float(lightning_kwargs.pop("weight_decay"))
        except Exception:
            lightning_kwargs["l2"] = lightning_kwargs.pop("weight_decay")
    lightning_kwargs.setdefault("decay_steps", decay_steps)

    # Learning rate scheduler configuration
    # Supported schedulers:
    # - "lambda": Exponential decay (default)
    # - "step": Step decay
    # - "cosine": Cosine annealing
    # - "reduce_on_plateau": Reduce LR on plateau
    scheduler_config = lightning_kwargs.pop("scheduler_config", None)
    scheduler_alias = lightning_kwargs.pop("scheduler", None)
    if scheduler_alias is not None:
        if scheduler_config is not None:
            raise ValueError(
                "Provide only one of 'scheduler' or 'scheduler_config' in lightning configuration."
            )
        scheduler_config = scheduler_alias

    if scheduler_config is None:
        scheduler_config = {
            "type": "lambda",
            "interval": "step",
            "frequency": 1,
            "params": {
                "decay_rate": lightning_kwargs.get("decay_rate", 0.5),
                "decay_steps": lightning_kwargs.get("decay_steps", decay_steps),
            },
        }
    else:
        if not isinstance(scheduler_config, dict):
            raise ValueError("Expected scheduler configuration to be a mapping")
        scheduler_config = dict(scheduler_config)
        scheduler_config.setdefault("params", {})
        scheduler_type = str(scheduler_config.get("type", "")).lower()
        scheduler_config.setdefault("interval", "epoch" if scheduler_type == "reduce_on_plateau" else "step")
        scheduler_config.setdefault("frequency", 1)
        if scheduler_type == "reduce_on_plateau":
            scheduler_config.setdefault("monitor", "val_ap")
            scheduler_config["interval"] = "epoch"

    lightning_kwargs["scheduler_config"] = scheduler_config

    # Resolve which metric to monitor for checkpointing/early stopping
    metric_alias = str(selection_metric or "").lower()
    if metric_alias in {"sel_ap", "sum", "sum_ap", "train_plus_val", "train+val"}:
        monitor_metric = "sel_ap"
    elif metric_alias in {"val_ap", "ap", "validation_ap"}:
        monitor_metric = "val_ap"
    elif metric_alias in {"val_f1", "f1", "validation_f1"}:
        monitor_metric = "val_f1"
    else:
        print(f"Unknown selection_metric '{selection_metric}', defaulting to 'sel_ap'.")
        monitor_metric = "sel_ap"

    # Check for empty datasets
    if len(x_dyn_train) == 0:
        raise ValueError(f"Training set is empty! x_dyn_train shape: {x_dyn_train.shape}")
    if len(x_dyn_val) == 0:
        raise ValueError(f"Validation set is empty! x_dyn_val shape: {x_dyn_val.shape}")
    
    print(f"Dataset sizes - Train: {len(x_dyn_train)}, Val: {len(x_dyn_val)}")
    
    train_dataset = FireDataset(
        x_dyn_train,
        x_stat_train,
        x_cat_train,
        y_train,
        sample_weight=sample_w_train,
        loss_target=y_loss_train,
    )
    val_dataset = FireDataset(x_dyn_val, x_stat_val, x_cat_val, y_val, loss_target=y_loss_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = SequenceStaticLightningModule(
        model_name=model_name,
        model_config=model_kwargs,
        **lightning_kwargs,
    )

    callbacks = []
    history_cb = HistoryCallback()
    callbacks.append(history_cb)
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    checkpoint_path = None
    if model_path is not None:
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        safe_arch = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_name)
        metric_tag = monitor_metric
        filename_tmpl = f"{base_name}-{safe_arch}-{{epoch:02d}}-{{{metric_tag}:.4f}}"
        checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            filename=filename_tmpl,
            monitor=monitor_metric,
            mode="max",
            save_top_k=1,
        )
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None

    trainer_params = dict(trainer_kwargs or {})
    early_stopping_cfg = trainer_params.pop("early_stopping", None)
    if early_stopping_cfg is None:
        early_stopping_cfg = {}
    elif not isinstance(early_stopping_cfg, dict):
        raise ValueError("Expected nn_model.trainer.early_stopping to be a mapping")
    early_stopping_cfg = dict(early_stopping_cfg)
    if early_stopping_cfg.pop("enabled", True):
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping_cfg.pop("monitor", monitor_metric),
                patience=int(early_stopping_cfg.pop("patience", 10)),
                mode=early_stopping_cfg.pop("mode", "max"),
                min_delta=float(early_stopping_cfg.pop("min_delta", 0.0)),
                verbose=bool(early_stopping_cfg.pop("verbose", True)),
                **early_stopping_cfg,
            )
        )

    default_trainer_params = {
        "max_epochs": epochs,
        "callbacks": callbacks,
        "accelerator": "auto",
        "devices": "auto",
        "log_every_n_steps": 50,
        "precision": "16-mixed",
        "enable_checkpointing": checkpoint_callback is not None,
    }
    default_trainer_params.update(trainer_params)
    prediction_progress_bar = bool(default_trainer_params.get("enable_progress_bar", True))

    trainer = pl.Trainer(**default_trainer_params)

    trainer.fit(model, train_loader, val_loader)

    best_model = model
    if checkpoint_callback and checkpoint_callback.best_model_path:
        print("Loaded best model from checkpoint:", checkpoint_callback.best_model_path)
        best_model = SequenceStaticLightningModule.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            model_name=model_name,
            model_config=model_kwargs,
            **lightning_kwargs,
        )
        checkpoint_path = checkpoint_callback.best_model_path
    else:
        print("No checkpoint saved; using the last trained weights.")

    best_model.eval()
    best_model.to(device)

    results = {
        "history": history_cb.history,
        "model": best_model,
        "class_weight": cw_dict,
        "sample_w_train": sample_w_train,
        "training_sample_weight": training_sample_weight_info,
        "checkpoint_path": checkpoint_path,
        "model_architecture": model_name,
        "model_config": model_kwargs,
    }

    if plot_path is not None:
        os.makedirs(plot_path, exist_ok=True)
        plot_loss(history_cb.history, out_path=os.path.join(plot_path, "loss.png"))

    # Compute validation scores and, when needed, train scores to report selection metrics.
    y_val_logits = predict_logits(
        best_model,
        x_dyn_val,
        x_stat_val,
        x_cat_val,
        batch_size=batch_size,
        device=device,
        show_progress=prediction_progress_bar,
        desc="Predict validation",
    )
    y_val_probs = logits_to_probs(y_val_logits)
    val_ap = plot_recall_precision(
        y_val,
        y_val_probs,
        out_path=os.path.join(plot_path, "pr_validation.png") if plot_path is not None else None,
        title="Validation PR",
    )
    train_ap = float("nan")
    y_train_logits_for_sel = None
    y_train_probs_for_sel = None
    sel_ap = float(val_ap)
    if compute_train_predictions or monitor_metric == "sel_ap":
        y_train_logits_for_sel = predict_logits(
            best_model,
            x_dyn_train,
            x_stat_train,
            x_cat_train,
            batch_size=batch_size,
            device=device,
            show_progress=prediction_progress_bar,
            desc="Predict train",
        )
        y_train_probs_for_sel = logits_to_probs(y_train_logits_for_sel)
        train_ap = float(average_precision_score(y_train, y_train_probs_for_sel))
        sel_ap = float(val_ap) + float(train_ap)
    results.update({
        "y_val_logits": y_val_logits,
        "y_val_probs": y_val_probs,
        "y_train_logits": y_train_logits_for_sel,
        "y_train_probs": y_train_probs_for_sel,
        "val_ap": float(val_ap),
        "train_ap": train_ap,
        "sel_ap": sel_ap,
        "selection_metric": monitor_metric,
    })

    return results


def calculate_metric_errors(
    y_true,
    y_probs,
    threshold=0.5,
    sample_weight=None,
    trials: int = 5,
    sample_size: int = 50_000,
    seed: int = 17,
):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)
    if trials <= 1 or len(y_true) <= 1:
        return {
            "precision_error": None,
            "recall_error": None,
            "f1_error": None,
            "ap_error": None,
        }
    sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    n = min(len(y_true), int(sample_size)) if sample_size and sample_size > 0 else len(y_true)
    if n <= 1:
        return {
            "precision_error": None,
            "recall_error": None,
            "f1_error": None,
            "ap_error": None,
        }

    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    pos_frac = len(pos) / len(y_true) if len(y_true) else 0.0
    rng = np.random.default_rng(seed)
    values = {"precision": [], "recall": [], "f1": [], "ap": []}
    for _ in range(trials):
        if len(pos) and len(neg):
            n_pos = min(max(1, int(round(n * pos_frac))), n - 1)
            idx = np.concatenate(
                [
                    rng.choice(pos, size=n_pos, replace=True),
                    rng.choice(neg, size=n - n_pos, replace=True),
                ]
            )
            rng.shuffle(idx)
        else:
            idx = rng.choice(np.arange(len(y_true)), size=n, replace=True)
        sw = None if sample_weight is None else sample_weight[idx]
        y_pred = (y_probs[idx] >= threshold).astype(int)
        scores = {
            "precision": precision_score(y_true[idx], y_pred, zero_division=0, sample_weight=sw),
            "recall": recall_score(y_true[idx], y_pred, zero_division=0, sample_weight=sw),
            "f1": f1_score(y_true[idx], y_pred, zero_division=0, sample_weight=sw),
            "ap": average_precision_score(y_true[idx], y_probs[idx], sample_weight=sw),
        }
        for key, value in scores.items():
            if np.isfinite(float(value)):
                values[key].append(float(value))
    return {
        f"{key}_error": float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        for key, vals in values.items()
    }


def calculate_metrics(
    y_true,
    y_probs,
    threshold=0.5,
    sample_weight=None,
    error_trials: int = 5,
    error_sample_size: int = 50_000,
    error_seed: int = 17,
):
    y_pred = (y_probs >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
    recall = recall_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
    f1 = f1_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
    ap = average_precision_score(y_true, y_probs, sample_weight=sample_weight)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": ap,
        "threshold": threshold,
    }
    metrics.update(
        calculate_metric_errors(
            y_true,
            y_probs,
            threshold=threshold,
            sample_weight=sample_weight,
            trials=error_trials,
            sample_size=error_sample_size,
            seed=error_seed,
        )
    )
    return metrics


def print_metrics(metrics_train, metrics_val, metrics_test):
    print("\nМетрики классификации:")
    print(
        f"Train: Precision = {metrics_train['precision']:.4f}, Recall = {metrics_train['recall']:.4f}, F1 = {metrics_train['f1']:.4f}"
    )
    print(
        f"Val:   Precision = {metrics_val['precision']:.4f}, Recall = {metrics_val['recall']:.4f}, F1 = {metrics_val['f1']:.4f}"
    )
    print(
        f"Test:  Precision = {metrics_test['precision']:.4f}, Recall = {metrics_test['recall']:.4f}, F1 = {metrics_test['f1']:.4f}"
    )

    print("\nAUC-PR:")
    print(f"Train: {metrics_train['ap']:.4f}")
    print(f"Val:   {metrics_val['ap']:.4f}")
    print(f"Test:  {metrics_test['ap']:.4f}")


def load_prepared_training_arrays(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    """Load either split-array or full-array prepared NN data."""

    keys = set(data.files)
    if {"x_dyn_train", "x_dyn_val", "x_stat_train", "x_stat_val", "y_train", "y_val"}.issubset(keys):
        x_dyn_train = data["x_dyn_train"]
        x_dyn_val = data["x_dyn_val"]
        x_dyn_test = data["x_dyn_test"] if "x_dyn_test" in keys else np.zeros((0, *x_dyn_train.shape[1:]), dtype=x_dyn_train.dtype)
        x_stat_train = data["x_stat_train"]
        x_stat_val = data["x_stat_val"]
        x_stat_test = data["x_stat_test"] if "x_stat_test" in keys else np.zeros((0, x_stat_train.shape[1]), dtype=x_stat_train.dtype)
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"] if "y_test" in keys else np.zeros((0,), dtype=y_train.dtype)
        if "x_cat_train" in keys:
            x_cat_train = data["x_cat_train"].astype(np.int64)
            x_cat_val = data["x_cat_val"].astype(np.int64)
            x_cat_test = data["x_cat_test"].astype(np.int64) if "x_cat_test" in keys else np.zeros((0, x_cat_train.shape[1]), dtype=np.int64)
        else:
            x_cat_train = np.zeros((x_dyn_train.shape[0], 0), dtype=np.int64)
            x_cat_val = np.zeros((x_dyn_val.shape[0], 0), dtype=np.int64)
            x_cat_test = np.zeros((x_dyn_test.shape[0], 0), dtype=np.int64)
        out = {
            "format": "split",
            "x_dyn_train": x_dyn_train,
            "x_dyn_val": x_dyn_val,
            "x_dyn_test": x_dyn_test,
            "x_stat_train": x_stat_train,
            "x_stat_val": x_stat_val,
            "x_stat_test": x_stat_test,
            "x_cat_train": x_cat_train,
            "x_cat_val": x_cat_val,
            "x_cat_test": x_cat_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }
        for base_name in ("soft_label", "sample_weight"):
            for split_name, fallback_len in (
                ("train", len(y_train)),
                ("val", len(y_val)),
                ("test", len(y_test)),
            ):
                key = f"{base_name}_{split_name}"
                if key in keys:
                    out[key] = data[key]
                else:
                    out[key] = None
        for base_name in ("lat", "lon"):
            for split_name in ("train", "val", "test"):
                key = f"{base_name}_{split_name}"
                out[key] = data[key] if key in keys else None
        return out

    if {"x_dyn", "y", "split"}.issubset(keys) and ("x_static" in keys or "x_stat" in keys):
        x_dyn = np.asarray(data["x_dyn"])
        if not np.issubdtype(x_dyn.dtype, np.floating):
            x_dyn = x_dyn.astype(np.float32)
        x_stat = np.asarray(data["x_static" if "x_static" in keys else "x_stat"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        split = np.asarray(data["split"], dtype=np.int8)
        x_cat = np.asarray(data["x_cat"], dtype=np.int64) if "x_cat" in keys else np.zeros((x_dyn.shape[0], 0), dtype=np.int64)

        if not (x_dyn.shape[0] == x_stat.shape[0] == y.shape[0] == split.shape[0] == x_cat.shape[0]):
            raise ValueError(
                "Full-array prepared data has inconsistent row counts: "
                f"x_dyn={x_dyn.shape[0]}, x_stat={x_stat.shape[0]}, x_cat={x_cat.shape[0]}, "
                f"y={y.shape[0]}, split={split.shape[0]}."
            )

        masks = {
            "train": split == 0,
            "val": split == 1,
            "test": split == 2,
        }
        if masks["train"].sum() == 0 or masks["val"].sum() == 0:
            raise ValueError("Full-array prepared data must contain split codes 0=train and 1=validation.")

        out = {
            "format": "full",
            "x_dyn_train": x_dyn[masks["train"]],
            "x_dyn_val": x_dyn[masks["val"]],
            "x_dyn_test": x_dyn[masks["test"]],
            "x_stat_train": x_stat[masks["train"]],
            "x_stat_val": x_stat[masks["val"]],
            "x_stat_test": x_stat[masks["test"]],
            "x_cat_train": x_cat[masks["train"]],
            "x_cat_val": x_cat[masks["val"]],
            "x_cat_test": x_cat[masks["test"]],
            "y_train": y[masks["train"]],
            "y_val": y[masks["val"]],
            "y_test": y[masks["test"]],
        }
        optional_specs = {
            "soft_label": (np.float32, None),
            "sample_weight": (np.float32, None),
            "lat": (np.float32, None),
            "lon": (np.float32, None),
        }
        for base_name, (dtype, fill_value) in optional_specs.items():
            if base_name in keys:
                values = np.asarray(data[base_name], dtype=dtype)
            elif fill_value is None:
                values = None
            else:
                values = np.full(y.shape[0], fill_value, dtype=dtype)
            for split_name, mask in masks.items():
                key = f"{base_name}_{split_name}"
                out[key] = None if values is None else values[mask]
        return out

    raise KeyError(
        "prepared_data.npz must contain either split arrays "
        "(x_dyn_train/x_stat_train/y_train) or full arrays (x_dyn/x_static/y/split)."
    )


def build_loss_targets(
    *,
    y_train: np.ndarray,
    y_val: np.ndarray,
    soft_label_train: np.ndarray | None,
    soft_label_val: np.ndarray | None,
    loss_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cfg = dict(loss_config or {})
    info: dict[str, Any] = {"enabled": bool(cfg)}

    y_loss_train = np.asarray(y_train, dtype=np.float32)
    y_loss_val = np.asarray(y_val, dtype=np.float32)
    if cfg.get("use_soft_labels", False):
        if soft_label_train is None:
            raise ValueError("loss.use_soft_labels is true, but prepared_data.npz has no soft_label array.")
        y_loss_train = np.maximum(
            np.asarray(soft_label_train, dtype=np.float32),
            np.asarray(y_train, dtype=np.float32),
        ).clip(0.0, 1.0)
        if soft_label_val is not None:
            y_loss_val = np.maximum(
                np.asarray(soft_label_val, dtype=np.float32),
                np.asarray(y_val, dtype=np.float32),
            ).clip(0.0, 1.0)
        info["soft_labels"] = {
            "enabled": True,
            "train_mean": float(np.mean(y_loss_train)) if len(y_loss_train) else None,
            "train_soft_negative_rows": int(((np.asarray(y_train) == 0) & (y_loss_train > 0)).sum()),
        }
    else:
        info["soft_labels"] = {"enabled": False}

    return y_loss_train, y_loss_val, info


def build_spatial_coordinate_features(lat: np.ndarray | None, lon: np.ndarray | None) -> np.ndarray | None:
    if lat is None or lon is None:
        return None
    lat = np.asarray(lat, dtype=np.float32).reshape(-1)
    lon = np.asarray(lon, dtype=np.float32).reshape(-1)
    if lat.shape[0] != lon.shape[0]:
        raise ValueError(f"Latitude/longitude row mismatch: {lat.shape[0]} vs {lon.shape[0]}")
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


def append_spatial_coordinate_features(
    *,
    config: dict[str, Any],
    x_stat_train: np.ndarray,
    x_stat_val: np.ndarray,
    x_stat_test: np.ndarray,
    lat_train: np.ndarray | None,
    lon_train: np.ndarray | None,
    lat_val: np.ndarray | None,
    lon_val: np.ndarray | None,
    lat_test: np.ndarray | None,
    lon_test: np.ndarray | None,
    static_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    cfg = dict(config.get("spatial_coordinate_features") or {})
    if not cfg.get("enabled", False):
        return x_stat_train, x_stat_val, x_stat_test, static_columns, {"enabled": False}

    train_features = build_spatial_coordinate_features(lat_train, lon_train)
    val_features = build_spatial_coordinate_features(lat_val, lon_val)
    test_features = build_spatial_coordinate_features(lat_test, lon_test)
    if train_features is None or val_features is None or test_features is None:
        raise ValueError(
            "spatial_coordinate_features.enabled is true, but prepared_data.npz does not contain lat/lon arrays."
        )

    names = [
        "coord_lat_scaled",
        "coord_lon_scaled",
        "coord_lat_sin",
        "coord_lat_cos",
        "coord_lon_sin",
        "coord_lon_cos",
    ]
    info = {
        "enabled": True,
        "columns": names,
        "description": "Latitude/longitude coordinate encodings appended to x_static at train time.",
    }
    return (
        np.concatenate([x_stat_train, train_features], axis=1).astype(np.float32),
        np.concatenate([x_stat_val, val_features], axis=1).astype(np.float32),
        np.concatenate([x_stat_test, test_features], axis=1).astype(np.float32),
        [*static_columns, *names],
        info,
    )


def resolve_spatial_static_indices(
    model_params: dict[str, Any],
    static_columns: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = dict(model_params)
    explicit_indices = params.pop("spatial_static_indices", None)
    explicit_columns = params.pop("spatial_static_columns", None)
    prefixes = params.pop("spatial_static_prefixes", None)
    auto = bool(params.pop("auto_spatial_static", explicit_indices is None and explicit_columns is None))

    selected: set[int] = set()
    if explicit_indices is not None:
        selected.update(int(idx) for idx in explicit_indices)

    column_to_idx = {name: idx for idx, name in enumerate(static_columns)}
    if explicit_columns:
        missing = [str(name) for name in explicit_columns if str(name) not in column_to_idx]
        if missing:
            raise ValueError(f"spatial_static_columns not found in prepared static columns: {missing}")
        selected.update(column_to_idx[str(name)] for name in explicit_columns)

    prefix_values = list(prefixes or (DEFAULT_SPATIAL_STATIC_PREFIXES if auto else []))
    for idx, name in enumerate(static_columns):
        if any(name == prefix or name.startswith(str(prefix)) for prefix in prefix_values):
            selected.add(idx)

    indices = sorted(selected)
    if not indices:
        raise ValueError(
            "Spatial TSN requires at least one spatial static feature. "
            "Set nn_model.params.spatial_static_columns, spatial_static_prefixes, "
            "or enable auto_spatial_static."
        )
    invalid = [idx for idx in indices if idx < 0 or idx >= len(static_columns)]
    if invalid:
        raise ValueError(f"spatial_static_indices outside static feature range: {invalid}")

    params["spatial_static_indices"] = indices
    info = {
        "enabled": True,
        "count": len(indices),
        "indices": indices,
        "columns": [static_columns[idx] for idx in indices],
        "prefixes": prefix_values,
        "auto_spatial_static": auto,
    }
    return params, info


def zero_like(array: np.ndarray) -> np.ndarray:
    return np.zeros_like(array).astype(array.dtype, copy=False)


def apply_input_ablation(
    *,
    config: dict[str, Any],
    x_dyn_train: np.ndarray,
    x_dyn_val: np.ndarray,
    x_dyn_test: np.ndarray,
    x_stat_train: np.ndarray,
    x_stat_val: np.ndarray,
    x_stat_test: np.ndarray,
    x_cat_train: np.ndarray,
    x_cat_val: np.ndarray,
    x_cat_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ablation = dict(config.get("input_ablation") or {})
    if not ablation.get("enabled", False):
        return (
            x_dyn_train,
            x_dyn_val,
            x_dyn_test,
            x_stat_train,
            x_stat_val,
            x_stat_test,
            x_cat_train,
            x_cat_val,
            x_cat_test,
            ablation,
        )

    if ablation.get("zero_dynamic", False):
        x_dyn_train = zero_like(x_dyn_train)
        x_dyn_val = zero_like(x_dyn_val)
        x_dyn_test = zero_like(x_dyn_test)
    if ablation.get("zero_static", False):
        x_stat_train = zero_like(x_stat_train)
        x_stat_val = zero_like(x_stat_val)
        x_stat_test = zero_like(x_stat_test)
    if ablation.get("zero_categorical", False):
        x_cat_train = zero_like(x_cat_train)
        x_cat_val = zero_like(x_cat_val)
        x_cat_test = zero_like(x_cat_test)

    print("[input_ablation]", json.dumps(ablation, sort_keys=True))
    return (
        x_dyn_train,
        x_dyn_val,
        x_dyn_test,
        x_stat_train,
        x_stat_val,
        x_stat_test,
        x_cat_train,
        x_cat_val,
        x_cat_test,
        ablation,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a global LSTM-based fire-risk neural net.")
    parser.add_argument("--config-path", default="configs/features_config_30d_LSTM_early_fusion.yaml")
    parser.add_argument("--data-dir", help="Directory containing prepared_data.npz.")
    parser.add_argument("--data-path", help="Explicit path to prepared_data.npz.")
    parser.add_argument("--model-path", help="Output Lightning checkpoint path.")
    parser.add_argument("--plot-path", help="Directory for training plots.")
    parser.add_argument("--metrics-path", help="Optional JSON path for final train/val/test metrics.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    CONFIG_PATH = args.config_path

    # Load configuration
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    DATA_DIR = args.data_dir or config.get("output_train_data_dir", "data/saved_features/nn_train_data")
    DATA_PATH = args.data_path or os.path.join(DATA_DIR, "prepared_data.npz")
    MODEL_PATH = args.model_path or config.get("nn_output_model_path", "models/lstm_ml_fire.ckpt")
    PLOT_PATH = args.plot_path or config.get("output_nn_plots_dir", "outputs/plots_nn")
    METRICS_PATH = args.metrics_path or config.get("nn_metrics_path")
    # Deterministic seeding for reproducibility across runs
    # Set `seed:` in YAML to override (defaults to 17)
    try:
        pl.seed_everything(int(config.get("seed", 17)), workers=True)
    except Exception:
        pl.seed_everything(17, workers=True)

    metadata_path = os.path.join(DATA_DIR, "prepared_metadata.json")
    metadata: dict[str, Any] = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception as exc:
            print(f"Warning: failed to load metadata from '{metadata_path}': {exc}")
            metadata = {}
    else:
        print(
            f"Warning: metadata file '{metadata_path}' not found; categorical embeddings and resplitting may be unavailable."
        )

    data = np.load(DATA_PATH)
    prepared_arrays = load_prepared_training_arrays(data)
    data_format = prepared_arrays.pop("format")
    x_dyn_train = prepared_arrays["x_dyn_train"]
    x_dyn_val = prepared_arrays["x_dyn_val"]
    x_dyn_test = prepared_arrays["x_dyn_test"]
    x_stat_train = prepared_arrays["x_stat_train"]
    x_stat_val = prepared_arrays["x_stat_val"]
    x_stat_test = prepared_arrays["x_stat_test"]
    x_cat_train = prepared_arrays["x_cat_train"]
    x_cat_val = prepared_arrays["x_cat_val"]
    x_cat_test = prepared_arrays["x_cat_test"]
    y_train = prepared_arrays["y_train"]
    y_val = prepared_arrays["y_val"]
    y_test = prepared_arrays["y_test"]
    soft_label_train = prepared_arrays.get("soft_label_train")
    soft_label_val = prepared_arrays.get("soft_label_val")
    sample_weight_train = prepared_arrays.get("sample_weight_train")
    lat_train = prepared_arrays.get("lat_train")
    lon_train = prepared_arrays.get("lon_train")
    lat_val = prepared_arrays.get("lat_val")
    lon_val = prepared_arrays.get("lon_val")
    lat_test = prepared_arrays.get("lat_test")
    lon_test = prepared_arrays.get("lon_test")
    print(f"Loaded prepared NN data from {DATA_PATH} using '{data_format}' format.")

    static_columns = list(metadata.get("static_columns") or [])
    if len(static_columns) != x_stat_train.shape[1]:
        static_columns = [f"static_{idx}" for idx in range(x_stat_train.shape[1])]

    train_end_cfg = config.get("train_end")
    val_end_cfg = config.get("val_end")
    if data_format == "split" and train_end_cfg is not None and val_end_cfg is not None:
        def _load_indices_from_npz(key: str) -> np.ndarray:
            if key not in data:
                return np.empty(0, dtype=int)
            return np.asarray(data[key], dtype=int)

        # Preserve original references in case we need to fall back.
        orig_x_dyn_train, orig_x_dyn_val, orig_x_dyn_test = x_dyn_train, x_dyn_val, x_dyn_test
        orig_x_stat_train, orig_x_stat_val, orig_x_stat_test = x_stat_train, x_stat_val, x_stat_test
        orig_x_cat_train, orig_x_cat_val, orig_x_cat_test = x_cat_train, x_cat_val, x_cat_test
        orig_y_train, orig_y_val, orig_y_test = y_train, y_val, y_test

        train_idx_orig = _load_indices_from_npz("train_idx")
        val_idx_orig = _load_indices_from_npz("val_idx")
        test_idx_orig = _load_indices_from_npz("test_idx")

        def _attempt_proportional_fallback():
            """Fallback: Re-split by retaining the original proportions and ordering."""
            split_specs = [
                ("train", train_idx_orig, orig_x_dyn_train, orig_x_stat_train, orig_x_cat_train, orig_y_train),
                ("val", val_idx_orig, orig_x_dyn_val, orig_x_stat_val, orig_x_cat_val, orig_y_val),
                ("test", test_idx_orig, orig_x_dyn_test, orig_x_stat_test, orig_x_cat_test, orig_y_test),
            ]
            dyn_parts, stat_parts, cat_parts, target_parts, idx_parts = [], [], [], [], []
            total_samples = 0
            for _, indices, dyn, stat, cat, target in split_specs:
                if dyn.shape[0] == 0:
                    continue
                total_samples += dyn.shape[0]
                dyn_parts.append(dyn)
                stat_parts.append(stat)
                cat_parts.append(cat)
                target_parts.append(target)
                idx_parts.append(indices)

            if total_samples == 0:
                return None

            dyn_all = np.concatenate(dyn_parts, axis=0)
            stat_all = np.concatenate(stat_parts, axis=0)
            cat_all = np.concatenate(cat_parts, axis=0) if cat_parts else np.zeros((total_samples, 0), dtype=np.int64)
            target_all = np.concatenate(target_parts, axis=0)

            combined_idx = np.concatenate(idx_parts, axis=0) if idx_parts else np.arange(total_samples)
            if combined_idx.size == total_samples:
                order = np.argsort(combined_idx)
                dyn_all = dyn_all[order]
                stat_all = stat_all[order]
                cat_all = cat_all[order]
                target_all = target_all[order]

            train_count = min(orig_x_dyn_train.shape[0], total_samples)
            val_count = min(orig_x_dyn_val.shape[0], total_samples - train_count)
            test_count = total_samples - train_count - val_count

            if train_count == 0 or val_count == 0:
                return None

            cut1 = train_count
            cut2 = train_count + val_count

            return {
                "x_dyn_train": dyn_all[:cut1],
                "x_dyn_val": dyn_all[cut1:cut2],
                "x_dyn_test": dyn_all[cut2:],
                "x_stat_train": stat_all[:cut1],
                "x_stat_val": stat_all[cut1:cut2],
                "x_stat_test": stat_all[cut2:],
                "x_cat_train": cat_all[:cut1],
                "x_cat_val": cat_all[cut1:cut2],
                "x_cat_test": cat_all[cut2:],
                "y_train": target_all[:cut1],
                "y_val": target_all[cut1:cut2],
                "y_test": target_all[cut2:],
                "message": (
                    "Applied proportional fallback split based on stored split sizes "
                    f"(train={train_count}, val={val_count}, test={test_count})."
                ),
            }

        new_splits = None

        try:
            if not metadata:
                raise ValueError("Prepared metadata not available for dataset re-splitting.")
            dataset_meta = metadata.get("dataset") or {}
            coord_meta = (metadata.get("coordinates") or {}).get("columns") or []

            datetime_col_name = dataset_meta.get("datetime_column", "datetime")
            datetime_meta = next(
                (item for item in coord_meta if item.get("name") == datetime_col_name),
                None,
            )

            datetime_series = None
            if datetime_meta:
                npz_key = datetime_meta.get("npz_key")
                if npz_key and npz_key in data:
                    unit = datetime_meta.get("unit", "ns")
                    raw_values = np.asarray(data[npz_key], dtype=np.int64)
                    datetime_series = pd.to_datetime(raw_values, unit=unit, errors="coerce")
                    if datetime_series.isna().any():
                        raise ValueError(
                            f"Unable to parse datetime values using metadata column '{datetime_col_name}'."
                        )

            if datetime_series is None:
                raise ValueError(
                    "Datetime information not available in metadata; cannot rebuild dataset splits."
                )

            datetime_values = datetime_series.to_numpy(copy=False)
            if datetime_values.size == 0:
                raise ValueError("No datetime values available; cannot rebuild dataset splits.")

            split_specs = [
                ("train", train_idx_orig, orig_x_dyn_train, orig_x_stat_train, orig_x_cat_train, orig_y_train),
                ("val", val_idx_orig, orig_x_dyn_val, orig_x_stat_val, orig_x_cat_val, orig_y_val),
                ("test", test_idx_orig, orig_x_dyn_test, orig_x_stat_test, orig_x_cat_test, orig_y_test),
            ]

            idx_parts, dyn_parts, stat_parts, cat_parts, target_parts = [], [], [], [], []
            for split_name, indices, dyn, stat, cat, target in split_specs:
                if dyn.shape[0] == 0:
                    continue
                if indices.size != dyn.shape[0]:
                    raise ValueError(
                        f"Split '{split_name}' index size ({indices.size}) does not match feature rows ({dyn.shape[0]})."
                    )
                idx_parts.append(indices)
                dyn_parts.append(dyn)
                stat_parts.append(stat)
                cat_parts.append(cat)
                target_parts.append(target)

            if not idx_parts:
                raise ValueError("No samples available across stored splits; cannot rebuild dataset.")

            combined_idx = np.concatenate(idx_parts, axis=0)
            if np.unique(combined_idx).size != combined_idx.size:
                raise ValueError("Duplicate sample indices detected across stored splits.")

            if combined_idx.size and (
                combined_idx.min() < 0 or combined_idx.max() >= datetime_values.shape[0]
            ):
                raise ValueError(
                    "Original split indices fall outside valid range for datetime reconstruction."
                )

            order = np.argsort(combined_idx)
            sorted_indices = combined_idx[order]

            dyn_all = np.concatenate(dyn_parts, axis=0)[order]
            stat_all = np.concatenate(stat_parts, axis=0)[order]
            cat_all = np.concatenate(cat_parts, axis=0)[order] if cat_parts else np.zeros((order.size, 0), dtype=np.int64)
            target_all = np.concatenate(target_parts, axis=0)[order]

            datetime_subset = datetime_values[sorted_indices]

            train_cutoff = pd.to_datetime(train_end_cfg)
            val_cutoff = pd.to_datetime(val_end_cfg)
            if val_cutoff < train_cutoff:
                raise ValueError("val_end must be on or after train_end.")

            train_cutoff_np = np.datetime64(train_cutoff)
            val_cutoff_np = np.datetime64(val_cutoff)

            train_mask = datetime_subset <= train_cutoff_np
            val_mask = (datetime_subset > train_cutoff_np) & (datetime_subset <= val_cutoff_np)
            test_mask = datetime_subset > val_cutoff_np

            new_train_idx = np.where(train_mask)[0]
            new_val_idx = np.where(val_mask)[0]
            new_test_idx = np.where(test_mask)[0]

            if new_train_idx.size == 0:
                raise ValueError(
                    f"Resplit produced empty training set using train_end={train_end_cfg}."
                )
            if new_val_idx.size == 0:
                raise ValueError(
                    f"Resplit produced empty validation set using val_end={val_end_cfg}."
                )
            if new_test_idx.size == 0:
                print(
                    f"Warning: Resplit produced empty test set with val_end={val_end_cfg}; continuing without test samples."
                )

            new_splits = {
                "x_dyn_train": dyn_all[new_train_idx],
                "x_dyn_val": dyn_all[new_val_idx],
                "x_dyn_test": dyn_all[new_test_idx],
                "x_stat_train": stat_all[new_train_idx],
                "x_stat_val": stat_all[new_val_idx],
                "x_stat_test": stat_all[new_test_idx],
                "x_cat_train": cat_all[new_train_idx],
                "x_cat_val": cat_all[new_val_idx],
                "x_cat_test": cat_all[new_test_idx],
                "y_train": target_all[new_train_idx],
                "y_val": target_all[new_val_idx],
                "y_test": target_all[new_test_idx],
                "message": (
                    f"Re-split dataset using train_end={train_end_cfg} and val_end={val_end_cfg}. "
                    f"New split sizes -> train: {new_train_idx.size}, val: {new_val_idx.size}, test: {new_test_idx.size}"
                ),
            }

            samples_available = datetime_values.shape[0]
            if samples_available != sorted_indices.size:
                print(
                    f"Warning: Rebuild covered {sorted_indices.size}/{samples_available} samples present in metadata; "
                    "proceeding with resplit on available subset."
                )

        except Exception as exc:
            print(f"Warning: Failed to re-split dataset using config boundaries: {exc}")
            proportional = _attempt_proportional_fallback()
            if proportional is not None:
                new_splits = proportional
            else:
                print("Falling back to original splits stored in prepared_data.npz.")

        if new_splits is not None:
            x_dyn_train = new_splits["x_dyn_train"]
            x_dyn_val = new_splits["x_dyn_val"]
            x_dyn_test = new_splits["x_dyn_test"]
            x_stat_train = new_splits["x_stat_train"]
            x_stat_val = new_splits["x_stat_val"]
            x_stat_test = new_splits["x_stat_test"]
            x_cat_train = new_splits["x_cat_train"]
            x_cat_val = new_splits["x_cat_val"]
            x_cat_test = new_splits["x_cat_test"]
            y_train = new_splits["y_train"]
            y_val = new_splits["y_val"]
            y_test = new_splits["y_test"]
            soft_label_train = None
            soft_label_val = None
            sample_weight_train = None
            lat_train = lon_train = lat_val = lon_val = lat_test = lon_test = None
            print(new_splits["message"])

    (
        x_stat_train,
        x_stat_val,
        x_stat_test,
        static_columns,
        spatial_coordinate_info,
    ) = append_spatial_coordinate_features(
        config=config,
        x_stat_train=x_stat_train,
        x_stat_val=x_stat_val,
        x_stat_test=x_stat_test,
        lat_train=lat_train,
        lon_train=lon_train,
        lat_val=lat_val,
        lon_val=lon_val,
        lat_test=lat_test,
        lon_test=lon_test,
        static_columns=static_columns,
    )
    if spatial_coordinate_info.get("enabled"):
        print("[spatial_coordinate_features]", json.dumps(spatial_coordinate_info, sort_keys=True))

    (
        x_dyn_train,
        x_dyn_val,
        x_dyn_test,
        x_stat_train,
        x_stat_val,
        x_stat_test,
        x_cat_train,
        x_cat_val,
        x_cat_test,
        input_ablation,
    ) = apply_input_ablation(
        config=config,
        x_dyn_train=x_dyn_train,
        x_dyn_val=x_dyn_val,
        x_dyn_test=x_dyn_test,
        x_stat_train=x_stat_train,
        x_stat_val=x_stat_val,
        x_stat_test=x_stat_test,
        x_cat_train=x_cat_train,
        x_cat_val=x_cat_val,
        x_cat_test=x_cat_test,
    )

    print("Shapes:")
    print("x_dyn_train:", x_dyn_train.shape)
    print("x_dyn_val:", x_dyn_val.shape)
    print("x_dyn_test:", x_dyn_test.shape)
    print("x_stat_train:", x_stat_train.shape)
    print("x_stat_val:", x_stat_val.shape)
    print("x_stat_test:", x_stat_test.shape)
    print("y_train mean:", float(y_train.mean()), "shape:", y_train.shape)
    print("y_val mean:", float(y_val.mean()), "shape:", y_val.shape)
    print("y_test mean:", float(y_test.mean()), "shape:", y_test.shape)

    nn_model_cfg = config.get("nn_model") or {}
    if nn_model_cfg and not isinstance(nn_model_cfg, dict):
        raise ValueError("Expected 'nn_model' section in config to be a mapping")

    model_name = nn_model_cfg.get("architecture", "lstm_mlp")
    model_params = dict(nn_model_cfg.get("params") or {})
    lightning_params = nn_model_cfg.get("lightning") or {}
    trainer_params = nn_model_cfg.get("trainer") or {}

    spatial_branch_info: dict[str, Any] = {"enabled": False}
    spatial_architectures = {"spatial_tsn_mlp", "tsn_spatial_mlp"}
    if str(model_name).lower() in spatial_architectures:
        model_params, spatial_branch_info = resolve_spatial_static_indices(model_params, static_columns)
        print(
            "[spatial_static_branch]",
            json.dumps(
                {
                    "enabled": True,
                    "count": spatial_branch_info["count"],
                    "columns": spatial_branch_info["columns"],
                },
                sort_keys=True,
            ),
        )

    categorical_embeddings_meta = metadata.get("categorical_embeddings") or []
    if categorical_embeddings_meta:
        model_params["categorical_embeddings"] = categorical_embeddings_meta

    if not isinstance(model_params, dict):
        raise ValueError("Expected 'nn_model.params' to be a mapping")
    if not isinstance(lightning_params, dict):
        raise ValueError("Expected 'nn_model.lightning' to be a mapping")
    if not isinstance(trainer_params, dict):
        raise ValueError("Expected 'nn_model.trainer' to be a mapping")

    # Extract class weights from config if provided
    class_weights_cfg = config.get('class_weights', None)
    loss_cfg = dict(config.get("loss") or {})
    y_loss_train, y_loss_val, loss_training_info = build_loss_targets(
        y_train=y_train,
        y_val=y_val,
        soft_label_train=soft_label_train,
        soft_label_val=soft_label_val,
        loss_config=loss_cfg,
    )
    if loss_training_info.get("enabled"):
        print("[loss_training]", json.dumps(loss_training_info, sort_keys=True))
    
    # Selection metric can be configured in YAML: selection_metric: "sel_ap" or "val_ap"
    selection_metric = str(config.get("selection_metric", "sel_ap"))
    compute_train_metrics = bool(config.get("compute_train_metrics", True))
    results = train_pipeline(
        x_dyn_train,
        x_stat_train,
        y_train,
        x_dyn_val,
        x_stat_val,
        y_val,
        x_cat_train=x_cat_train,
        x_cat_val=x_cat_val,
        x_cat_test=x_cat_test,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        num_workers=config['num_workers'],
        model_path=MODEL_PATH,
        plot_path=PLOT_PATH,
        model_name=model_name,
        model_config=model_params,
        lightning_config=lightning_params,
        trainer_kwargs=trainer_params,
        class_weights=class_weights_cfg,
        prepared_sample_weight_train=sample_weight_train,
        training_sample_weight_config=config.get("training_sample_weight"),
        selection_metric=selection_metric,
        compute_train_predictions=compute_train_metrics,
        y_loss_train=y_loss_train,
        y_loss_val=y_loss_val,
        loss_config=loss_cfg,
    )

    model = results["model"]
    sample_w_train = results.get("sample_w_train", None)
    device = next(model.parameters()).device
    progress_bar_enabled = bool(trainer_params.get("enable_progress_bar", True))

    y_train_probs = results.get("y_train_probs")
    y_train_logits = results.get("y_train_logits")
    if compute_train_metrics and y_train_probs is None:
        y_train_logits = predict_logits(
            model,
            x_dyn_train,
            x_stat_train,
            x_cat_train,
            batch_size=config['batch_size'],
            device=device,
            show_progress=progress_bar_enabled,
            desc="Predict train",
        )
        y_train_probs = logits_to_probs(y_train_logits)
    y_val_probs = results["y_val_probs"]
    y_val_logits = results.get("y_val_logits")
    y_test_logits = predict_logits(
        model,
        x_dyn_test,
        x_stat_test,
        x_cat_test,
        batch_size=config['batch_size'],
        device=device,
        show_progress=progress_bar_enabled,
        desc="Predict test",
    )
    y_test_probs = logits_to_probs(y_test_logits)

    thr, best_f1_val = choose_threshold_f1(y_val, y_val_probs)
    print("Val threshold for F1:", thr, "val_f1:", best_f1_val)
    random_error_trials = int(config.get("random_error_trials", 5))
    random_error_sample_size = int(config.get("random_error_sample_size", 50_000))

    metrics_train = (
        calculate_metrics(
            y_train,
            y_train_probs,
            threshold=thr,
            sample_weight=sample_w_train,
            error_trials=random_error_trials,
            error_sample_size=random_error_sample_size,
            error_seed=17,
        )
        if y_train_probs is not None
        else {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "ap": float("nan"),
            "precision_error": None,
            "recall_error": None,
            "f1_error": None,
            "ap_error": None,
            "threshold": thr,
        }
    )
    metrics_val = calculate_metrics(
        y_val,
        y_val_probs,
        threshold=thr,
        error_trials=random_error_trials,
        error_sample_size=random_error_sample_size,
        error_seed=18,
    )
    metrics_test = calculate_metrics(
        y_test,
        y_test_probs,
        threshold=thr,
        error_trials=random_error_trials,
        error_sample_size=random_error_sample_size,
        error_seed=19,
    )

    print_metrics(metrics_train, metrics_val, metrics_test)

    if y_train_probs is not None:
        plot_recall_precision(
            y_train,
            y_train_probs,
            out_path=os.path.join(PLOT_PATH, "pr_train.png"),
            title="Train PR",
        )

    plot_recall_precision(
        y_test,
        y_test_probs,
        out_path=os.path.join(PLOT_PATH, "pr_test.png"),
        title="Test PR",
    )

    if METRICS_PATH:
        prediction_artifacts = {}
        pred_dir = os.path.join(os.path.dirname(METRICS_PATH), "legacy_sampled_predictions")
        if y_val_logits is None:
            y_val_logits = predict_logits(
                model,
                x_dyn_val,
                x_stat_val,
                x_cat_val,
                batch_size=config['batch_size'],
                device=device,
                show_progress=progress_bar_enabled,
                desc="Save validation logits",
            )
        prediction_artifacts["validation"] = save_sampled_prediction_table(
            os.path.join(pred_dir, "validation_predictions.parquet"),
            y_val,
            y_val_logits,
            "validation",
        )
        prediction_artifacts["test"] = save_sampled_prediction_table(
            os.path.join(pred_dir, "test_predictions.parquet"),
            y_test,
            y_test_logits,
            "test",
        )
        if y_train_probs is not None:
            if y_train_logits is None:
                y_train_logits = predict_logits(
                    model,
                    x_dyn_train,
                    x_stat_train,
                    x_cat_train,
                    batch_size=config['batch_size'],
                    device=device,
                    show_progress=progress_bar_enabled,
                    desc="Save train logits",
                )
            prediction_artifacts["train"] = save_sampled_prediction_table(
                os.path.join(pred_dir, "train_predictions.parquet"),
                y_train,
                y_train_logits,
                "train",
            )
        metrics_out = {
            "config_path": CONFIG_PATH,
            "data_path": DATA_PATH,
            "model_path": results.get("checkpoint_path") or MODEL_PATH,
            "architecture": results.get("model_architecture"),
            "model_config": results.get("model_config"),
            "input_ablation": input_ablation,
            "loss_training": loss_training_info,
            "training_sample_weight": results.get("training_sample_weight", {"enabled": False}),
            "spatial_training": {
                "coordinate_features": spatial_coordinate_info,
                "static_branch": spatial_branch_info,
            },
            "feature_set": input_ablation.get("label") if input_ablation else None,
            "selection_metric": results.get("selection_metric"),
            "validation_threshold": thr,
            "validation_best_f1": best_f1_val,
            "train": metrics_train,
            "validation": metrics_val,
            "test": metrics_test,
            "legacy_sampled_metrics": {
                "train": metrics_train,
                "validation": metrics_val,
                "test": metrics_test,
                "evaluation_type": "legacy_sampled_case_control",
                "note": "Metrics are computed on the undersampled/case-control neural dataset.",
            },
            "primary_full_grid_calibrated_metrics": {
                "status": "not_run",
                "evaluation_type": "primary_full_grid_calibrated",
                "note": (
                    "The revision_evaluation wrapper records the dedicated calibrated deployment-grid "
                    "status after training. Raw neural logits are saved in legacy_sampled_predictions."
                ),
            },
            "prediction_artifacts": {
                key: str(value) for key, value in prediction_artifacts.items()
            },
            "split_sizes": {
                "train": int(len(y_train)),
                "validation": int(len(y_val)),
                "test": int(len(y_test)),
            },
        }
        metrics_dir = os.path.dirname(METRICS_PATH)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(METRICS_PATH, "w", encoding="utf-8") as handle:
            json.dump(
                metrics_out,
                handle,
                indent=2,
                default=lambda value: float(value) if isinstance(value, np.generic) else str(value),
            )
        print(f"Saved metrics to {METRICS_PATH}")

    # Optional: prior-shift correction to reduce validation→test gap
    # Configure in YAML:
    # prior_adjustment:
    #   enabled: true
    #   # For offline evaluation, use ground-truth test prior:
    #   use_true_test_prior: true
    #   # Or set an explicit expected deploy prior (e.g., 0.005):
    #   # deploy_prior: 0.005
    prior_cfg = dict(config.get("prior_adjustment", {}) or {})
    if prior_cfg.get("enabled", False):
        train_prior = float(np.clip(y_train.mean(), 1e-9, 1 - 1e-9))
        deploy_prior = prior_cfg.get("deploy_prior")
        if deploy_prior is None and bool(prior_cfg.get("use_true_test_prior", False)):
            deploy_prior = float(np.clip(y_test.mean(), 1e-9, 1 - 1e-9))
        if deploy_prior is not None:
            deploy_prior = float(np.clip(deploy_prior, 1e-9, 1 - 1e-9))

        if deploy_prior is None:
            print("[prior_adjustment] Skipped: no deploy_prior provided and use_true_test_prior is False.")
        else:
            # Adjust both validation and test to the same target prior
            y_val_probs_adj = adjust_probabilities_for_prior(
                y_val_probs, train_prior=train_prior, deploy_prior=deploy_prior
            )
            y_test_probs_adj = adjust_probabilities_for_prior(
                y_test_probs, train_prior=train_prior, deploy_prior=deploy_prior
            )

            thr_adj, best_f1_val_adj = choose_threshold_f1(y_val, y_val_probs_adj)
            print("[prior_adjustment] Deploy prior:", deploy_prior)
            print("[prior_adjustment] Val threshold for F1 (adjusted):", thr_adj, "val_f1:", best_f1_val_adj)

            metrics_val_adj = calculate_metrics(y_val, y_val_probs_adj, threshold=thr_adj)
            metrics_test_adj = calculate_metrics(y_test, y_test_probs_adj, threshold=thr_adj)

            print("\n[prior_adjustment] Metrics with prior shift correction:")
            print_metrics(metrics_train, metrics_val_adj, metrics_test_adj)

            plot_recall_precision(
                y_val,
                y_val_probs_adj,
                out_path=os.path.join(PLOT_PATH, "pr_validation_adjusted.png"),
                title="Validation PR (adjusted)",
            )
            plot_recall_precision(
                y_test,
                y_test_probs_adj,
                out_path=os.path.join(PLOT_PATH, "pr_test_adjusted.png"),
                title="Test PR (adjusted)",
            )
