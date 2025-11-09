
from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.append(os.getcwd())



logger = logging.getLogger("firecast")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y‑%m‑%d %H:%M:%S",
)

def quantile_normalize_predictions(
    preds: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    """
    Match *distribution* of predictions to ground truth via quantile mapping.

    Useful when you **do** have contemporaneous truth data (e.g. hindcasts).
    For real forecasts keep raw probabilities instead.
    """
    preds_f = preds.ravel()
    truth_f = truth.ravel()

    # short‑circuit: no fires in truth → use 99‑th percentile as cut‑off
    if truth_f.max() == 0:
        threshold = np.percentile(preds_f, 99)
        return (preds > threshold).astype(float)

    order_pred = np.argsort(preds_f)
    order_truth = np.argsort(truth_f)

    new_vals = np.empty_like(preds_f)
    new_vals[order_pred] = np.sort(truth_f)[
        np.linspace(0, len(truth_f) - 1, len(preds_f)).astype(int)
    ]
    return new_vals.reshape(preds.shape)


def calibrate_threshold(preds: np.ndarray, truth: np.ndarray) -> float:
    """
    Pick the probability threshold that gives the same fire/no‑fire ratio
    in *preds* as observed in *truth*.
    """
    truth_bin = (truth > 0).astype(int) if truth.max() > 1 else truth
    fire_ratio = truth_bin.mean()
    if fire_ratio == 0:
        return 0.999  # extremely rare event
    if fire_ratio == 1:
        return 0.001
    quantile = 1 - fire_ratio
    return np.quantile(preds, quantile)


# plotting helpers ────────────────────────────────────────────────────────────
PLOT_KW = dict(figsize=(16, 12), shading="auto", vmin=0, vmax=1)


def _pcolormesh(
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
    cmap: str | ListedColormap,
    *,
    title: str,
    save_to: Path,
) -> None:
    plt.figure(**PLOT_KW)
    plt.pcolormesh(lons, lats, data, cmap=cmap, **PLOT_KW)
    plt.colorbar(label="Prediction probability")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.axis("equal")
    os.makedirs(save_to.parent, exist_ok=True)
    plt.savefig(save_to, dpi=300, bbox_inches="tight")
    plt.close()
    logger.debug("Saved plot → %s", save_to)


def plot_probability_map(
    lats: np.ndarray,
    lons: np.ndarray,
    probs: np.ndarray,
    *,
    title: str,
    out_file: Path,
) -> None:
    _pcolormesh(lats, lons, probs, cmap="YlOrRd", title=title, save_to=out_file)


def plot_binary_map(
    lats: np.ndarray,
    lons: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
    title: str,
    out_file: Path,
) -> None:
    binary = (probs >= threshold).astype(int)
    cmap = ListedColormap(["lightblue", "red"])
    _pcolormesh(lats, lons, binary, cmap=cmap, title=title, save_to=out_file)


def save_netcdf(
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
    *,
    valid_date: date,
    out_file: Path,
    kind: str = "raw",
) -> None:
    """Write a single‑time‑step prediction grid to NetCDF."""
    ds = xr.Dataset(
        {f"fire_probability_{kind}": (("latitude", "longitude"), data)},
        coords=dict(latitude=lats, longitude=lons, time=[np.datetime64(valid_date)]),
        attrs=dict(
            description=f"Fire prediction probabilities ({kind})",
            created=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        ),
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_file)
    ds.close()
    logger.debug("Saved NetCDF → %s", out_file)