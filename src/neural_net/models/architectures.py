"""Model architecture registry for neural network training."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type, List

import torch
from torch import nn


def _build_embedding_layers(
    categorical_embeddings: Optional[List[Dict]]
) -> tuple[nn.ModuleList, int]:
    layers = nn.ModuleList()
    total_dim = 0
    if categorical_embeddings:
        for meta in categorical_embeddings:
            cardinality = int(meta.get("cardinality", 0))
            if cardinality <= 0:
                continue
            embedding_dim = int(meta.get("embedding_dim", max(8, cardinality // 2)))
            layer = nn.Embedding(cardinality, embedding_dim)
            layers.append(layer)
            total_dim += embedding_dim
    return layers, total_dim


def _concat_embeddings(
    embedding_layers: nn.ModuleList,
    stat: torch.Tensor,
    cat: Optional[torch.Tensor],
) -> torch.Tensor:
    if not embedding_layers:
        return stat
    if cat is None:
        raise ValueError("Categorical features tensor is required but was not provided.")
    if cat.ndim != 2 or cat.shape[1] != len(embedding_layers):
        raise ValueError(
            f"Expected categorical feature matrix with {len(embedding_layers)} columns, got {cat.shape if cat is not None else None}."
        )
    embeddings: List[torch.Tensor] = []
    for idx, layer in enumerate(embedding_layers):
        embeddings.append(layer(cat[:, idx]))
    if embeddings:
        embedded = torch.cat(embeddings, dim=1)
        stat = torch.cat([stat, embedded], dim=1)
    return stat


_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Decorator to register an architecture under a given name."""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def available_models() -> list[str]:
    """Return the list of registered architecture names."""

    return sorted(_MODEL_REGISTRY.keys())


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model architecture."""

    try:
        model_cls = _MODEL_REGISTRY[name]
    except KeyError as err:
        available = ", ".join(available_models()) or "<none>"
        raise ValueError(f"Unknown model architecture '{name}'. Available: {available}") from err
    return model_cls(**kwargs)


class AttentionPooling(nn.Module):
    """Additive attention pooling over sequence outputs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self._last_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, hidden)
        energy = torch.tanh(self.projection(x))
        scores = self.score(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        self._last_weights = weights
        return context


@register_model("mlp")
class MLPModel(nn.Module):
    """Feed-forward baseline that flattens sequence inputs before fusion."""

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_static: int,
        dynamic_units: int = 512,
        dropout_dynamic: float = 0.0,
        static_units: int = 512,
        dropout_static: float = 0.0,
        merge_units: int = 256,
        dropout_merged: float = 0.0,
        categorical_embeddings: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0 or n_channels <= 0:
            raise ValueError("'seq_len' and 'n_channels' must be positive")
        self.embedding_layers, embedding_dim = _build_embedding_layers(categorical_embeddings)
        static_input_dim = n_static + embedding_dim
        if static_input_dim <= 0:
            raise ValueError("Static feature dimension must be positive after adding embeddings")
        if dynamic_units <= 0 or static_units <= 0 or merge_units <= 0:
            raise ValueError("hidden dimensions must be positive")
        dyn_input_dim = seq_len * n_channels

        dyn_layers = [
            nn.Linear(dyn_input_dim, dynamic_units, bias=True),
            nn.BatchNorm1d(dynamic_units),
            nn.ReLU(),
        ]
        if dropout_dynamic > 0:
            dyn_layers.append(nn.Dropout(dropout_dynamic))
        self.dynamic_net = nn.Sequential(*dyn_layers)

        static_layers = [
            nn.Linear(static_input_dim, static_units, bias=True),
            nn.BatchNorm1d(static_units),
            nn.ReLU(),
        ]
        if dropout_static > 0:
            static_layers.append(nn.Dropout(dropout_static))
        self.static_net = nn.Sequential(*static_layers)

        half_units = max(merge_units // 2, 1)
        quarter_units = max(merge_units // 4, 1)
        merged_layers = [
            nn.Linear(dynamic_units + static_units, merge_units, bias=False),
            nn.BatchNorm1d(merge_units),
            nn.ReLU(),
        ]
        if dropout_merged > 0:
            merged_layers.append(nn.Dropout(dropout_merged))
        merged_layers.extend(
            [
                nn.Linear(merge_units, half_units, bias=False),
                nn.BatchNorm1d(half_units),
                nn.ReLU(),
                nn.Linear(half_units, quarter_units, bias=False),
                nn.BatchNorm1d(quarter_units),
                nn.ReLU(),
            ]
        )
        self.mlp = nn.Sequential(*merged_layers)
        self.out = nn.Linear(quarter_units, 1, bias=True)

    def forward(self, dyn: torch.Tensor, stat: torch.Tensor, cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = dyn.size(0)
        dyn_feat = dyn.reshape(batch_size, -1)
        dyn_feat = self.dynamic_net(dyn_feat)
        stat = _concat_embeddings(self.embedding_layers, stat, cat)
        stat_feat = self.static_net(stat)
        merged = torch.cat([dyn_feat, stat_feat], dim=1)
        logits = self.out(self.mlp(merged))
        return logits.squeeze(-1)


@register_model("lstm_mlp")
class LSTMMlpModel(nn.Module):
    """Baseline LSTM + MLP architecture used in training."""

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_static: int,
        lstm_units: int = 192,
        dropout_lstm: float = 0.1,
        static_units: int = 512,
        dropout_static: float = 0.0,
        merge_units: int = 256,
        dropout_merged: float = 0.0,
        categorical_embeddings: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.dropout_lstm = nn.Dropout(dropout_lstm) if dropout_lstm > 0 else nn.Identity()

        self.embedding_layers, embedding_dim = _build_embedding_layers(categorical_embeddings)
        static_input_dim = n_static + embedding_dim
        if static_input_dim <= 0:
            raise ValueError("Static feature dimension must be positive after adding embeddings")
        static_layers = [
            nn.Linear(static_input_dim, static_units, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(static_units),
        ]
        if dropout_static > 0:
            static_layers.append(nn.Dropout(dropout_static))
        self.static_net = nn.Sequential(*static_layers)

        merged_layers = [
            nn.Linear(lstm_units + static_units, merge_units, bias=False),
            nn.BatchNorm1d(merge_units),
            nn.ReLU(),
        ]
        if dropout_merged > 0:
            merged_layers.append(nn.Dropout(dropout_merged))
        merged_layers.extend(
            [
                nn.Linear(merge_units, merge_units // 2, bias=False),
                nn.BatchNorm1d(merge_units // 2),
                nn.ReLU(),
                nn.Linear(merge_units // 2, merge_units // 4, bias=False),
                nn.BatchNorm1d(merge_units // 4),
                nn.ReLU(),
            ]
        )
        self.mlp = nn.Sequential(*merged_layers)
        self.out = nn.Linear(merge_units // 4, 1, bias=True)

    def forward(self, dyn: torch.Tensor, stat: torch.Tensor, cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        lstm_out, _ = self.lstm(dyn)
        dyn_feat = lstm_out[:, -1, :]
        dyn_feat = self.dropout_lstm(dyn_feat)
        stat = _concat_embeddings(self.embedding_layers, stat, cat)
        stat_feat = self.static_net(stat)
        merged = torch.cat([dyn_feat, stat_feat], dim=1)
        logits = self.out(self.mlp(merged))
        return logits.squeeze(-1)


@register_model("lstm_early_fusion")
class LSTMEarlyFusionModel(nn.Module):
    """LSTM variant that fuses static features into the sequence encoder."""

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_static: int,
        lstm_units: int = 192,
        dropout_lstm: float = 0.1,
        static_units: int = 512,
        dropout_static: float = 0.0,
        merge_units: int = 256,
        dropout_merged: float = 0.0,
        categorical_embeddings: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0 or n_channels <= 0 or n_static <= 0:
            raise ValueError("'seq_len', 'n_channels', and 'n_static' must be positive")
        self.seq_len = seq_len
        self.embedding_layers, embedding_dim = _build_embedding_layers(categorical_embeddings)
        static_input_dim = n_static + embedding_dim
        if static_input_dim <= 0:
            raise ValueError("Static feature dimension must be positive after adding embeddings")
        fused_input_size = n_channels + static_input_dim
        self.lstm = nn.LSTM(
            input_size=fused_input_size,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.dropout_lstm = nn.Dropout(dropout_lstm) if dropout_lstm > 0 else nn.Identity()

        static_layers = [
            nn.Linear(static_input_dim, static_units, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(static_units),
        ]
        if dropout_static > 0:
            static_layers.append(nn.Dropout(dropout_static))
        self.static_net = nn.Sequential(*static_layers)

        merged_layers = [
            nn.Linear(lstm_units + static_units, merge_units, bias=False),
            nn.BatchNorm1d(merge_units),
            nn.ReLU(),
        ]
        if dropout_merged > 0:
            merged_layers.append(nn.Dropout(dropout_merged))
        half_units = max(merge_units // 2, 1)
        quarter_units = max(merge_units // 4, 1)
        merged_layers.extend(
            [
                nn.Linear(merge_units, half_units, bias=False),
                nn.BatchNorm1d(half_units),
                nn.ReLU(),
                nn.Linear(half_units, quarter_units, bias=False),
                nn.BatchNorm1d(quarter_units),
                nn.ReLU(),
            ]
        )
        self.mlp = nn.Sequential(*merged_layers)
        self.out = nn.Linear(quarter_units, 1, bias=True)

    def forward(self, dyn: torch.Tensor, stat: torch.Tensor, cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if dyn.size(1) != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, got {dyn.size(1)}",
            )
        stat = _concat_embeddings(self.embedding_layers, stat, cat)
        stat_expanded = stat.unsqueeze(1).expand(-1, self.seq_len, -1)
        fused_sequence = torch.cat([dyn, stat_expanded], dim=-1)
        lstm_out, _ = self.lstm(fused_sequence)
        dyn_feat = lstm_out[:, -1, :]
        dyn_feat = self.dropout_lstm(dyn_feat)
        stat_feat = self.static_net(stat)
        merged = torch.cat([dyn_feat, stat_feat], dim=1)
        logits = self.out(self.mlp(merged))
        return logits.squeeze(-1)


@register_model("lstm_attention")
class LSTMAttentionModel(nn.Module):
    """LSTM encoder with attention pooling followed by an MLP."""

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_static: int,
        lstm_units: int = 192,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout_lstm: float = 0.1,
        static_units: int = 512,
        dropout_static: float = 0.0,
        merge_units: int = 256,
        dropout_merged: float = 0.0,
        categorical_embeddings: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=lstm_units,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
        )
        direction_factor = 2 if bidirectional else 1
        self.attention = AttentionPooling(lstm_units * direction_factor)
        self.dropout_lstm = nn.Dropout(dropout_lstm) if dropout_lstm > 0 else nn.Identity()

        self.embedding_layers, embedding_dim = _build_embedding_layers(categorical_embeddings)
        static_input_dim = n_static + embedding_dim
        if static_input_dim <= 0:
            raise ValueError("Static feature dimension must be positive after adding embeddings")
        static_layers = [
            nn.Linear(static_input_dim, static_units, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(static_units),
        ]
        if dropout_static > 0:
            static_layers.append(nn.Dropout(dropout_static))
        self.static_net = nn.Sequential(*static_layers)

        merged_input_dim = lstm_units * direction_factor + static_units
        merged_layers = [
            nn.Linear(merged_input_dim, merge_units, bias=False),
            nn.BatchNorm1d(merge_units),
            nn.ReLU(),
        ]
        if dropout_merged > 0:
            merged_layers.append(nn.Dropout(dropout_merged))
        merged_layers.extend(
            [
                nn.Linear(merge_units, merge_units // 2, bias=False),
                nn.BatchNorm1d(merge_units // 2),
                nn.ReLU(),
                nn.Linear(merge_units // 2, merge_units // 4, bias=False),
                nn.BatchNorm1d(merge_units // 4),
                nn.ReLU(),
            ]
        )
        self.mlp = nn.Sequential(*merged_layers)
        self.out = nn.Linear(merge_units // 4, 1, bias=True)

    def forward(self, dyn: torch.Tensor, stat: torch.Tensor, cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        lstm_out, _ = self.lstm(dyn)
        dyn_feat = self.attention(lstm_out)
        dyn_feat = self.dropout_lstm(dyn_feat)
        stat = _concat_embeddings(self.embedding_layers, stat, cat)
        stat_feat = self.static_net(stat)
        merged = torch.cat([dyn_feat, stat_feat], dim=1)
        logits = self.out(self.mlp(merged))
        return logits.squeeze(-1)


__all__ = [
    "MLPModel",
    "available_models",
    "build_model",
    "register_model",
    "LSTMMlpModel",
    "LSTMEarlyFusionModel",
    "LSTMAttentionModel",
]
