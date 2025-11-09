import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
import yaml
from matplotlib import pyplot as plt

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.neural_net.models import SequenceStaticLightningModule
from src.neural_net.train_nn import predict_probs


def load_checkpoint(model_path: str, device: torch.device) -> SequenceStaticLightningModule:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Neural network checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    hyperparams = checkpoint.get("hyper_parameters") or {}
    model_name = hyperparams.get("model_name", "lstm_mlp")
    model_config = hyperparams.get("model_config") or {}

    lightning_defaults = {
        key: hyperparams.get(key)
        for key in ("learning_rate", "decay_rate", "decay_steps", "l2", "clip_gradient_norm")
        if key in hyperparams
    }
    del checkpoint

    model = SequenceStaticLightningModule.load_from_checkpoint(
        model_path,
        map_location=device,
        model_name=model_name,
        model_config=model_config,
        **lightning_defaults,
    )
    model.eval()
    model.to(device)
    return model


def compute_grad_importance(
    model: SequenceStaticLightningModule,
    x_dyn: np.ndarray,
    x_stat: np.ndarray,
    x_cat: np.ndarray | None,
    batch_size: int = 1024,
    use_grad_x_input: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dyn_importance_per_channel, static_importance_per_feature).

    Importance is mean absolute gradient w.r.t. input. If use_grad_x_input is True,
    multiply gradients by inputs before taking absolute value.
    """

    device = next(model.parameters()).device
    model.eval()

    lstm_modules: list[tuple[nn.Module, bool]] = []
    for module in model.modules():
        if isinstance(module, nn.LSTM):
            lstm_modules.append((module, module.training))
            module.train()

    n_samples, n_days, n_channels = x_dyn.shape
    n_static = x_stat.shape[1] if x_stat is not None and x_stat.ndim == 2 else 0

    dyn_acc = np.zeros((n_channels,), dtype=np.float64)
    stat_acc = np.zeros((n_static,), dtype=np.float64) if n_static else None
    count = 0

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        dyn_batch = torch.as_tensor(x_dyn[start:end], dtype=torch.float32, device=device)
        dyn_batch.requires_grad_(True)
        if n_static:
            stat_batch = torch.as_tensor(x_stat[start:end], dtype=torch.float32, device=device)
            stat_batch.requires_grad_(True)
        else:
            stat_batch = torch.zeros((end - start, 0), dtype=torch.float32, device=device)
            stat_batch.requires_grad_(True)

        if x_cat is not None and x_cat.shape[1] > 0:
            cat_batch = torch.as_tensor(x_cat[start:end], dtype=torch.long, device=device)
        else:
            cat_batch = torch.zeros((end - start, 0), dtype=torch.long, device=device)

        logits = model(dyn_batch, stat_batch, cat_batch if cat_batch.numel() else None)
        if logits.dim() != 1:
            logits = logits.view(-1)

        # Backprop sum of logits to get per-input gradients
        loss = logits.sum()
        grads = torch.autograd.grad(
            loss,
            (dyn_batch, stat_batch) if n_static else (dyn_batch,),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        if n_static:
            g_dyn, g_stat = grads
        else:
            (g_dyn,) = grads

        g_dyn = g_dyn  # (B, T, C)
        if use_grad_x_input:
            g_dyn = g_dyn * dyn_batch
        g_dyn = g_dyn.abs()  # (B, T, C)
        # Mean over time and batch to per-channel score
        dyn_scores = g_dyn.mean(dim=(0, 1)).detach().cpu().numpy()
        dyn_acc += dyn_scores.astype(np.float64)

        if n_static:
            if g_stat is None:
                raise RuntimeError("Failed to compute gradient for static inputs")
            if use_grad_x_input:
                g_stat = g_stat * stat_batch
            g_stat = g_stat.abs()
            stat_scores = g_stat.mean(dim=0).detach().cpu().numpy()
            stat_acc += stat_scores.astype(np.float64)

        count += 1

        # free tensors explicitly
        del dyn_batch, stat_batch, logits, loss, grads, g_dyn
        if n_static:
            del g_stat
        torch.cuda.empty_cache() if device.type == "cuda" else None

    dyn_importance = (dyn_acc / max(count, 1)).astype(np.float32)
    if n_static:
        stat_importance = (stat_acc / max(count, 1)).astype(np.float32)
    else:
        stat_importance = np.zeros((0,), dtype=np.float32)

    for module, state in lstm_modules:
        module.train(state)

    return dyn_importance, stat_importance


def permutation_importance(
    model: SequenceStaticLightningModule,
    x_dyn: np.ndarray,
    x_stat: np.ndarray,
    x_cat: np.ndarray | None,
    y_true: np.ndarray,
    batch_size: int = 1024,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Permutation importance measured by drop in AP on validation set.

    Returns (dyn_channel_importance, static_feature_importance, base_ap)
    where importance = base_ap - permuted_ap (higher is more important).
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    base_probs = predict_probs(model, x_dyn, x_stat, x_cat, batch_size=batch_size, device=device)
    base_ap = float(average_precision_score(y_true, base_probs))

    n_channels = x_dyn.shape[2]
    n_static = x_stat.shape[1] if x_stat is not None and x_stat.ndim == 2 else 0

    dyn_importance = np.zeros((n_channels,), dtype=np.float32)
    stat_importance = np.zeros((n_static,), dtype=np.float32) if n_static else np.zeros((0,), dtype=np.float32)

    # Dynamic channels
    for c in range(n_channels):
        x_dyn_perm = x_dyn.copy()
        idx = rng.permutation(x_dyn_perm.shape[0])
        x_dyn_perm[:, :, c] = x_dyn_perm[idx, :, c]
        probs = predict_probs(model, x_dyn_perm, x_stat, x_cat, batch_size=batch_size, device=device)
        ap = float(average_precision_score(y_true, probs))
        dyn_importance[c] = base_ap - ap

    # Static features
    if n_static:
        for j in range(n_static):
            x_stat_perm = x_stat.copy()
            idx = rng.permutation(x_stat_perm.shape[0])
            x_stat_perm[:, j] = x_stat_perm[idx, j]
            probs = predict_probs(model, x_dyn, x_stat_perm, x_cat, batch_size=batch_size, device=device)
            ap = float(average_precision_score(y_true, probs))
            stat_importance[j] = base_ap - ap

    return dyn_importance, stat_importance, base_ap


def save_bar_plot(names, values, title, path, top_k: int | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    values = np.asarray(values)
    names = list(names)

    order = np.argsort(values)[::-1]
    if top_k is not None:
        order = order[:top_k]
    values = values[order]
    names = [names[i] for i in order]

    plt.figure(figsize=(12, max(4, 0.3 * len(names))))
    plt.barh(range(len(names)), values[::-1])
    plt.yticks(range(len(names)), names[::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_interpretation(config_path: str, method: str = "grad", output_dir: str | None = None) -> Dict:
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    data_dir = config.get("output_train_data_dir", "data/saved_features/nn_train_data")
    npz_path = os.path.join(data_dir, "prepared_data.npz")
    meta_path = os.path.join(data_dir, "prepared_metadata.json")
    model_path = config.get("nn_model_path") or config.get("model_path")
    if not model_path:
        raise KeyError("Config must provide 'nn_model_path' or 'model_path' pointing to a trained checkpoint.")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Prepared data not found at {npz_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as meta_file:
        meta = json.load(meta_file)
    features_mask = (meta.get("features") or {}).get("mask") or {}
    dyn_channel_names = features_mask.get("dyn_channels") or []
    static_feature_names = features_mask.get("static") or []

    data = np.load(npz_path)
    x_dyn_val = data["x_dyn_val"]
    x_stat_val = data["x_stat_val"]
    y_val = data["y_val"].astype(np.float32)
    if "x_cat_val" in data.files:
        x_cat_val = data["x_cat_val"].astype(np.int64)
    else:
        x_cat_val = np.zeros((x_dyn_val.shape[0], 0), dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(model_path, device)

    method = method.strip().lower()
    results: Dict[str, np.ndarray | float] = {}

    if output_dir is None:
        output_dir = "outputs/nn_importance"
    os.makedirs(output_dir, exist_ok=True)

    if method in {"grad", "gradient", "saliency"}:
        dyn_imp, stat_imp = compute_grad_importance(
            model,
            x_dyn_val,
            x_stat_val,
            x_cat_val,
            batch_size=1024,
            use_grad_x_input=True,
        )
        results.update({
            "method": "grad_x_input",
            "dyn_importance": dyn_imp,
            "static_importance": stat_imp,
        })
    elif method in {"perm", "permutation"}:
        dyn_imp, stat_imp, base_ap = permutation_importance(
            model,
            x_dyn_val,
            x_stat_val,
            x_cat_val,
            y_val,
            batch_size=1024,
            seed=42,
        )
        results.update({
            "method": "permutation",
            "dyn_importance": dyn_imp,
            "static_importance": stat_imp,
            "base_ap": base_ap,
        })
    else:
        raise ValueError("Unknown method. Use 'grad' or 'permutation'.")

    # Save CSVs
    dyn_csv = os.path.join(output_dir, f"dynamic_importance_{results['method']}.csv")
    with open(dyn_csv, "w", encoding="utf-8") as fh:
        fh.write("feature,importance\n")
        for name, val in zip(dyn_channel_names or [f"dyn_ch_{i}" for i in range(len(results["dyn_importance"]))], results["dyn_importance"]):
            fh.write(f"{name},{float(val):.8f}\n")

    if results["static_importance"].size:
        stat_csv = os.path.join(output_dir, f"static_importance_{results['method']}.csv")
        with open(stat_csv, "w", encoding="utf-8") as fh:
            fh.write("feature,importance\n")
            for name, val in zip(static_feature_names or [f"stat_{i}" for i in range(len(results["static_importance"]))], results["static_importance"]):
                fh.write(f"{name},{float(val):.8f}\n")

    # Save plots
    save_bar_plot(
        dyn_channel_names or [f"dyn_ch_{i}" for i in range(len(results["dyn_importance"]))],
        results["dyn_importance"],
        title=f"Dynamic channels importance ({results['method']})",
        path=os.path.join(output_dir, f"dynamic_importance_{results['method']}.png"),
        top_k=40,
    )
    if results["static_importance"].size:
        save_bar_plot(
            static_feature_names or [f"stat_{i}" for i in range(len(results["static_importance"]))],
            results["static_importance"],
            title=f"Static features importance ({results['method']})",
            path=os.path.join(output_dir, f"static_importance_{results['method']}.png"),
            top_k=40,
        )

    print(
        f"Saved NN feature importance results to {output_dir} using method={results['method']}"
    )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute feature importance for trained NN model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/features_config_30d_nn.yaml",
        help="Path to features/training config used for NN.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="grad",
        choices=["grad", "permutation"],
        help="Importance method: gradient-based or permutation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/nn_importance",
        help="Directory to store importance CSVs and plots.",
    )

    args = parser.parse_args()
    run_interpretation(args.config, method=args.method, output_dir=args.output)
