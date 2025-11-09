"""Probability calibration helpers shared between prediction pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

try:  # optional dependency for GPU tensors
    import torch
except ImportError:  # pragma: no cover - torch optional at inference time
    torch = None  # type: ignore[assignment]

DEFAULT_TRAIN_PRIOR = 0.15
DEFAULT_DEPLOY_PRIOR = 1e-3


def adjust_probabilities_for_prior(
    scores: Any,
    *,
    train_prior: float = DEFAULT_TRAIN_PRIOR,
    deploy_prior: float = DEFAULT_DEPLOY_PRIOR,
    assume_logits: bool = False,
):
    """Correct wildfire risk scores for prior shift using log-odds adjustment."""

    if not (0.0 < train_prior < 1.0):
        raise ValueError(f"train_prior must be in (0, 1); received {train_prior}.")
    if not (0.0 < deploy_prior < 1.0):
        raise ValueError(f"deploy_prior must be in (0, 1); received {deploy_prior}.")

    shift = float(np.log(deploy_prior / (1.0 - deploy_prior)) - np.log(train_prior / (1.0 - train_prior)))

    if torch is not None and isinstance(scores, torch.Tensor):
        shift_tensor = torch.tensor(shift, dtype=scores.dtype, device=scores.device)
        if assume_logits:
            logits = scores + shift_tensor
        else:
            clipped = scores.clamp(min=1e-12, max=1 - 1e-12)
            logits = torch.log(clipped) - torch.log1p(-clipped) + shift_tensor
        return torch.sigmoid(logits)

    array = np.asarray(scores)
    if assume_logits:
        logits = array + shift
    else:
        clipped = np.clip(array, 1e-12, 1 - 1e-12)
        logits = np.log(clipped) - np.log1p(-clipped) + shift

    corrected = expit(logits)
    if isinstance(scores, pd.Series):
        return pd.Series(corrected, index=scores.index, name=scores.name)
    return corrected
