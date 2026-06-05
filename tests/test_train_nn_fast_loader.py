from __future__ import annotations

import numpy as np
import torch

from src.neural_net import train_nn
from src.neural_net.train_nn import (
    FireDataset,
    make_fire_dataloader,
    resolve_training_data_device,
    train_pipeline,
)


def toy_arrays(rows: int = 6):
    x_dyn = np.arange(rows * 2 * 3, dtype=np.float32).reshape(rows, 2, 3)
    x_stat = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
    x_cat = np.arange(rows * 2, dtype=np.int64).reshape(rows, 2)
    y = np.arange(rows, dtype=np.float32) % 2
    return x_dyn, x_stat, x_cat, y


def test_fire_dataset_batched_getitems_returns_one_batch() -> None:
    x_dyn, x_stat, x_cat, y = toy_arrays()
    dataset = FireDataset(x_dyn, x_stat, x_cat, y)

    batch = dataset.__getitems__([3, 1, 0])

    assert isinstance(batch, tuple)
    assert len(batch) == 6
    assert batch[0].shape == (3, 2, 3)
    assert batch[1].shape == (3, 4)
    assert batch[2].dtype == torch.long
    assert batch[3].tolist() == [1.0, 1.0, 0.0]
    np.testing.assert_allclose(batch[0][0].numpy(), x_dyn[3])


def test_make_fire_dataloader_uses_fast_batched_fetch() -> None:
    x_dyn, x_stat, x_cat, y = toy_arrays(rows=5)
    loader = make_fire_dataloader(
        x_dyn=x_dyn,
        x_stat=x_stat,
        x_cat=x_cat,
        y=y,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        training_device="cpu",
        data_device="cpu",
    )

    first = next(iter(loader))

    assert isinstance(first, tuple)
    assert first[0].shape == (2, 2, 3)
    assert first[1].shape == (2, 4)
    assert first[3].tolist() == [0.0, 1.0]


def test_resolve_training_data_device_auto_respects_vram_budget(monkeypatch) -> None:
    gib = 1024**3
    monkeypatch.setattr(train_nn.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_nn.torch.cuda, "mem_get_info", lambda: (10 * gib, 16 * gib))

    device, info = resolve_training_data_device(
        "auto",
        tensor_bytes=2 * gib,
        max_fraction=0.5,
        min_free_gb=1.0,
    )
    assert device == "cuda"
    assert info["reason"] == "fits_auto_budget"

    device, info = resolve_training_data_device(
        "auto",
        tensor_bytes=6 * gib,
        max_fraction=0.5,
        min_free_gb=1.0,
    )
    assert device == "cpu"
    assert info["reason"] == "too_large_for_auto_budget"


def test_train_pipeline_smoke_with_fast_loader_on_cpu() -> None:
    x_dyn, x_stat, x_cat, y = toy_arrays(rows=12)
    train_idx = np.arange(8)
    val_idx = np.arange(8, 12)

    results = train_pipeline(
        x_dyn[train_idx],
        x_stat[train_idx],
        y[train_idx],
        x_dyn[val_idx],
        x_stat[val_idx],
        y[val_idx],
        x_cat_train=x_cat[train_idx],
        x_cat_val=x_cat[val_idx],
        batch_size=4,
        epochs=1,
        num_workers=0,
        model_name="minimal_mlp",
        model_config={
            "hidden_units": 8,
            "dropout": 0.0,
            "dynamic_mode": "flatten",
            "categorical_mode": "ignore",
        },
        lightning_config={"learning_rate": 0.001, "l2": 0.0},
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
            "enable_progress_bar": False,
            "logger": False,
            "early_stopping": {"enabled": False},
            "data_loader": {"data_device": "cpu"},
        },
        class_weights={0: 1.0, 1: 1.0},
        selection_metric="val_ap",
        compute_train_predictions=False,
    )

    assert results["y_val_probs"].shape == (4,)
    assert results["y_train_probs"] is None
    assert np.isfinite(results["val_ap"])
