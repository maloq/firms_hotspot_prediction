"""Model architecture registry for neural network training."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type, List

import torch
from torch import nn
from torch.nn import functional as F


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


def _normalize_categorical_mode(mode: str | None, has_embeddings: bool) -> str:
    value = str(mode or "auto").strip().lower()
    if value == "auto":
        return "embedding" if has_embeddings else "ignore"
    if value in {"none", "ignore", "disabled", "off"}:
        return "ignore"
    if value in {"raw", "flat", "passthrough", "ordinal"}:
        return "raw"
    if value in {"embedding", "embeddings", "learned_embedding", "learned_embeddings"}:
        return "embedding"
    raise ValueError(
        "categorical_mode must be one of 'auto', 'ignore', 'raw', or 'embedding', "
        f"got '{mode}'."
    )


def _build_categorical_layers(
    categorical_embeddings: Optional[List[Dict]],
    n_categorical: int = 0,
    categorical_mode: str | None = "auto",
) -> tuple[nn.ModuleList, int, str, int]:
    n_categorical = int(n_categorical or 0)
    embeddings_meta = list(categorical_embeddings or [])
    mode = _normalize_categorical_mode(categorical_mode, has_embeddings=bool(embeddings_meta))

    if mode == "embedding":
        layers, embedding_dim = _build_embedding_layers(embeddings_meta)
        if n_categorical == 0:
            n_categorical = len(layers)
        if len(layers) != n_categorical:
            raise ValueError(
                f"Expected {n_categorical} categorical embedding specs, got {len(layers)}."
            )
        return layers, embedding_dim, mode, n_categorical

    if mode == "raw":
        return nn.ModuleList(), n_categorical, mode, n_categorical

    return nn.ModuleList(), 0, mode, n_categorical


def _concat_categorical_features(
    embedding_layers: nn.ModuleList,
    stat: torch.Tensor,
    cat: Optional[torch.Tensor],
    categorical_mode: str = "auto",
    n_categorical: int = 0,
) -> torch.Tensor:
    mode = _normalize_categorical_mode(categorical_mode, has_embeddings=bool(embedding_layers))
    if mode == "ignore":
        return stat

    expected_cols = len(embedding_layers) if mode == "embedding" else int(n_categorical or 0)
    if expected_cols == 0:
        return stat

    if cat is None:
        raise ValueError("Categorical features tensor is required but was not provided.")

    if cat.ndim != 2 or cat.shape[1] != expected_cols:
        raise ValueError(
            f"Expected categorical feature matrix with {expected_cols} columns, "
            f"got {cat.shape if cat is not None else None}."
        )

    if mode == "raw":
        return torch.cat([stat, cat.float()], dim=1)

    embeddings: List[torch.Tensor] = [layer(cat[:, idx]) for idx, layer in enumerate(embedding_layers)]
    embedded = torch.cat(embeddings, dim=1)
    return torch.cat([stat, embedded], dim=1)


def _normalize_dynamic_mode(mode: str | None) -> str:
    value = str(mode or "flatten").strip().lower()
    if value in {"flat", "flatten", "flattened"}:
        return "flatten"
    if value in {"last", "last_step"}:
        return "last"
    if value in {"mean", "avg", "average"}:
        return "mean"
    if value in {"summary", "summaries", "stats", "statistics"}:
        return "summary"
    raise ValueError(
        "dynamic_mode must be one of 'flatten', 'last', 'mean', or 'summary', "
        f"got '{mode}'."
    )


def _dynamic_feature_count(seq_len: int, n_channels: int, mode: str | None) -> int:
    mode = _normalize_dynamic_mode(mode)
    if mode == "flatten":
        return seq_len * n_channels
    if mode in {"last", "mean"}:
        return n_channels
    if mode == "summary":
        return 5 * n_channels
    raise AssertionError(f"Unhandled dynamic mode: {mode}")


def _extract_dynamic_features(dyn: torch.Tensor, mode: str | None) -> torch.Tensor:
    mode = _normalize_dynamic_mode(mode)
    if mode == "flatten":
        return dyn.reshape(dyn.size(0), -1)
    if mode == "last":
        return dyn[:, -1, :]
    if mode == "mean":
        return dyn.mean(dim=1)
    if mode == "summary":
        return torch.cat(
            [
                dyn[:, -1, :],
                dyn.mean(dim=1),
                dyn.std(dim=1, unbiased=False),
                torch.amin(dyn, dim=1),
                torch.amax(dyn, dim=1),
            ],
            dim=1,
        )
    raise AssertionError(f"Unhandled dynamic mode: {mode}")


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


def _as_int_tuple(value, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        items = (value,)
    elif isinstance(value, (list, tuple)):
        items = tuple(int(item) for item in value)
    else:
        raise ValueError(f"{name} must be an int or a sequence of ints")
    if not items or any(item <= 0 for item in items):
        raise ValueError(f"{name} must contain positive integers")
    return items


def _normalize_pooling_mode(value: str | None) -> str:
    mode = str(value or "attention").strip().lower()
    if mode in {"attention", "attn"}:
        return "attention"
    if mode in {"last", "last_step"}:
        return "last"
    if mode in {"mean", "avg", "average"}:
        return "mean"
    if mode == "max":
        return "max"
    if mode in {"mean_max", "mean+max", "avg_max"}:
        return "mean_max"
    raise ValueError("pooling must be one of 'attention', 'last', 'mean', 'max', or 'mean_max'")


class DilatedTemporalConvBlock(nn.Module):
    """Residual temporal block with multiple kernel windows and dilations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...],
        dilations: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        branch_specs = [(kernel, dilation) for kernel in kernel_sizes for dilation in dilations]
        branch_channels = max(out_channels // len(branch_specs), 1)
        self.branch_specs = branch_specs
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    branch_channels,
                    kernel_size=kernel,
                    dilation=dilation,
                    padding=0,
                )
                for kernel, dilation in branch_specs
            ]
        )
        merged_channels = branch_channels * len(branch_specs)
        self.project = nn.Conv1d(merged_channels, out_channels, kernel_size=1)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        seq_len = x.shape[-1]
        for (kernel, dilation), conv in zip(self.branch_specs, self.branches):
            padded = F.pad(x, (dilation * (kernel - 1), 0))
            y = conv(padded)
            if y.shape[-1] != seq_len:
                y = y[..., -seq_len:]
            outputs.append(y)
        merged = torch.cat(outputs, dim=1)
        out = self.project(merged)
        out = self.norm(out + self.residual(x))
        return self.dropout(self.activation(out))


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
        n_categorical: int = 0,
        categorical_mode: str = "auto",
    ) -> None:
        super().__init__()
        if seq_len <= 0 or n_channels <= 0:
            raise ValueError("'seq_len' and 'n_channels' must be positive")
        (
            self.embedding_layers,
            categorical_dim,
            self.categorical_mode,
            self.n_categorical,
        ) = _build_categorical_layers(categorical_embeddings, n_categorical, categorical_mode)
        static_input_dim = n_static + categorical_dim
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
        stat = _concat_categorical_features(
            self.embedding_layers,
            stat,
            cat,
            self.categorical_mode,
            self.n_categorical,
        )
        stat_feat = self.static_net(stat)
        merged = torch.cat([dyn_feat, stat_feat], dim=1)
        logits = self.out(self.mlp(merged))
        return logits.squeeze(-1)


@register_model("minimal_mlp")
class MinimalMLPModel(nn.Module):
    """Small feed-forward neural baseline over flattened/summary features."""

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_static: int,
        hidden_units: int = 128,
        dropout: float = 0.1,
        dynamic_mode: str = "flatten",
        categorical_embeddings: Optional[List[Dict]] = None,
        n_categorical: int = 0,
        categorical_mode: str = "auto",
    ) -> None:
        super().__init__()
        if seq_len <= 0 or n_channels <= 0:
            raise ValueError("'seq_len' and 'n_channels' must be positive")
        if n_static < 0:
            raise ValueError("'n_static' cannot be negative")
        if hidden_units <= 0:
            raise ValueError("hidden_units must be positive")

        self.dynamic_mode = _normalize_dynamic_mode(dynamic_mode)
        (
            self.embedding_layers,
            categorical_dim,
            self.categorical_mode,
            self.n_categorical,
        ) = _build_categorical_layers(categorical_embeddings, n_categorical, categorical_mode)

        input_dim = _dynamic_feature_count(seq_len, n_channels, self.dynamic_mode) + n_static + categorical_dim
        if input_dim <= 0:
            raise ValueError("Input feature dimension must be positive")

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_units, 1)

    def forward(self, dyn: torch.Tensor, stat: torch.Tensor, cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        dyn_feat = _extract_dynamic_features(dyn, self.dynamic_mode)
        stat = _concat_categorical_features(
            self.embedding_layers,
            stat,
            cat,
            self.categorical_mode,
            self.n_categorical,
        )
        features = torch.cat([dyn_feat, stat], dim=1)
        logits = self.out(self.net(features))
        return logits.squeeze(-1)


@register_model("ft_transformer")
class FTTransformerModel(nn.Module):
    """FT-Transformer-style tabular baseline for dynamic summaries + static inputs."""

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_static: int,
        token_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        dynamic_mode: str = "summary",
        head_units: int = 0,
        head_dropout: float = 0.0,
        norm_first: bool = True,
        categorical_embeddings: Optional[List[Dict]] = None,
        n_categorical: int = 0,
        categorical_mode: str = "auto",
    ) -> None:
        super().__init__()
        if seq_len <= 0 or n_channels <= 0:
            raise ValueError("'seq_len' and 'n_channels' must be positive")
        if n_static < 0:
            raise ValueError("'n_static' cannot be negative")
        if token_dim <= 0 or num_layers <= 0 or num_heads <= 0 or ffn_dim <= 0:
            raise ValueError("token_dim, num_layers, num_heads, and ffn_dim must be positive")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if head_units < 0:
            raise ValueError("head_units cannot be negative")

        self.dynamic_mode = _normalize_dynamic_mode(dynamic_mode)
        self.categorical_mode = _normalize_categorical_mode(
            categorical_mode,
            has_embeddings=bool(categorical_embeddings),
        )
        self.n_categorical = int(n_categorical or 0)
        embeddings_meta = list(categorical_embeddings or [])
        self.categorical_token_embeddings = nn.ModuleList()

        raw_categorical_count = 0
        if self.categorical_mode == "embedding":
            if self.n_categorical == 0:
                self.n_categorical = len(embeddings_meta)
            if len(embeddings_meta) != self.n_categorical:
                raise ValueError(
                    f"Expected {self.n_categorical} categorical embedding specs, "
                    f"got {len(embeddings_meta)}."
                )
            for meta in embeddings_meta:
                cardinality = int(meta.get("cardinality", 0))
                if cardinality <= 0:
                    raise ValueError("Categorical embedding cardinality must be positive")
                self.categorical_token_embeddings.append(nn.Embedding(cardinality, token_dim))
        elif self.categorical_mode == "raw":
            raw_categorical_count = self.n_categorical

        self.numeric_feature_count = (
            _dynamic_feature_count(seq_len, n_channels, self.dynamic_mode)
            + n_static
            + raw_categorical_count
        )
        if self.numeric_feature_count <= 0 and not self.categorical_token_embeddings:
            raise ValueError("FTTransformerModel requires at least one numeric or categorical token")

        if self.numeric_feature_count > 0:
            self.numeric_weight = nn.Parameter(torch.empty(self.numeric_feature_count, token_dim))
            self.numeric_bias = nn.Parameter(torch.empty(self.numeric_feature_count, token_dim))
            nn.init.xavier_uniform_(self.numeric_weight)
            nn.init.zeros_(self.numeric_bias)
        else:
            self.register_parameter("numeric_weight", None)
            self.register_parameter("numeric_bias", None)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        head_layers: list[nn.Module] = [nn.LayerNorm(token_dim)]
        if head_units > 0:
            head_layers.extend(
                [
                    nn.Linear(token_dim, head_units),
                    nn.GELU(),
                ]
            )
            if head_dropout > 0:
                head_layers.append(nn.Dropout(head_dropout))
            head_layers.append(nn.Linear(head_units, 1))
        else:
            head_layers.append(nn.Linear(token_dim, 1))
        self.head = nn.Sequential(*head_layers)

    def _categorical_tokens(self, cat: Optional[torch.Tensor]) -> torch.Tensor | None:
        if self.categorical_mode != "embedding" or not self.categorical_token_embeddings:
            return None
        if cat is None:
            raise ValueError("Categorical features tensor is required but was not provided.")
        if cat.ndim != 2 or cat.shape[1] != len(self.categorical_token_embeddings):
            raise ValueError(
                f"Expected categorical feature matrix with {len(self.categorical_token_embeddings)} columns, "
                f"got {cat.shape if cat is not None else None}."
            )
        tokens = [layer(cat[:, idx]) for idx, layer in enumerate(self.categorical_token_embeddings)]
        return torch.stack(tokens, dim=1)

    def forward(self, dyn: torch.Tensor, stat: torch.Tensor, cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        numeric_parts = [_extract_dynamic_features(dyn, self.dynamic_mode)]
        if stat.shape[1] > 0:
            numeric_parts.append(stat)
        if self.categorical_mode == "raw" and self.n_categorical > 0:
            if cat is None:
                raise ValueError("Categorical features tensor is required but was not provided.")
            if cat.ndim != 2 or cat.shape[1] != self.n_categorical:
                raise ValueError(
                    f"Expected categorical feature matrix with {self.n_categorical} columns, "
                    f"got {cat.shape if cat is not None else None}."
                )
            numeric_parts.append(cat.float())

        tokens: list[torch.Tensor] = []
        if self.numeric_feature_count > 0:
            numeric = torch.cat(numeric_parts, dim=1)
            if numeric.shape[1] != self.numeric_feature_count:
                raise ValueError(
                    f"Expected {self.numeric_feature_count} numeric features, got {numeric.shape[1]}."
                )
            tokens.append(numeric.unsqueeze(-1) * self.numeric_weight.unsqueeze(0) + self.numeric_bias.unsqueeze(0))

        cat_tokens = self._categorical_tokens(cat)
        if cat_tokens is not None:
            tokens.append(cat_tokens)

        batch_size = dyn.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, *tokens], dim=1)
        encoded = self.transformer(x)
        logits = self.head(encoded[:, 0, :])
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
        n_categorical: int = 0,
        categorical_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.dropout_lstm = nn.Dropout(dropout_lstm) if dropout_lstm > 0 else nn.Identity()

        (
            self.embedding_layers,
            categorical_dim,
            self.categorical_mode,
            self.n_categorical,
        ) = _build_categorical_layers(categorical_embeddings, n_categorical, categorical_mode)
        static_input_dim = n_static + categorical_dim
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
        stat = _concat_categorical_features(
            self.embedding_layers,
            stat,
            cat,
            self.categorical_mode,
            self.n_categorical,
        )
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
        n_categorical: int = 0,
        categorical_mode: str = "auto",
    ) -> None:
        super().__init__()
        if seq_len <= 0 or n_channels <= 0 or n_static <= 0:
            raise ValueError("'seq_len', 'n_channels', and 'n_static' must be positive")
        self.seq_len = seq_len
        (
            self.embedding_layers,
            categorical_dim,
            self.categorical_mode,
            self.n_categorical,
        ) = _build_categorical_layers(categorical_embeddings, n_categorical, categorical_mode)
        static_input_dim = n_static + categorical_dim
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
        stat = _concat_categorical_features(
            self.embedding_layers,
            stat,
            cat,
            self.categorical_mode,
            self.n_categorical,
        )
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
        n_categorical: int = 0,
        categorical_mode: str = "auto",
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

        (
            self.embedding_layers,
            categorical_dim,
            self.categorical_mode,
            self.n_categorical,
        ) = _build_categorical_layers(categorical_embeddings, n_categorical, categorical_mode)
        static_input_dim = n_static + categorical_dim
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
        stat = _concat_categorical_features(
            self.embedding_layers,
            stat,
            cat,
            self.categorical_mode,
            self.n_categorical,
        )
        stat_feat = self.static_net(stat)
        merged = torch.cat([dyn_feat, stat_feat], dim=1)
        logits = self.out(self.mlp(merged))
        return logits.squeeze(-1)


@register_model("tsn_mlp")
@register_model("tcn_mlp")
class TemporalConvNetModel(nn.Module):
    """Dilated temporal-convolution encoder with LSTM-style static fusion."""

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_static: int,
        tcn_units: int = 192,
        num_blocks: int = 3,
        kernel_sizes: tuple[int, ...] | list[int] = (2, 3, 5),
        dilations: tuple[int, ...] | list[int] = (1, 2, 4),
        dropout_tcn: float = 0.1,
        pooling: str = "attention",
        static_units: int = 512,
        dropout_static: float = 0.0,
        merge_units: int = 256,
        dropout_merged: float = 0.0,
        categorical_embeddings: Optional[List[Dict]] = None,
        n_categorical: int = 0,
        categorical_mode: str = "auto",
    ) -> None:
        super().__init__()
        if seq_len <= 0 or n_channels <= 0:
            raise ValueError("'seq_len' and 'n_channels' must be positive")
        if tcn_units <= 0 or num_blocks <= 0 or static_units <= 0 or merge_units <= 0:
            raise ValueError("tcn_units, num_blocks, static_units, and merge_units must be positive")

        self.seq_len = int(seq_len)
        kernel_sizes = _as_int_tuple(kernel_sizes, "kernel_sizes")
        dilations = _as_int_tuple(dilations, "dilations")
        self.pooling = _normalize_pooling_mode(pooling)

        blocks: list[nn.Module] = []
        in_channels = n_channels
        for _ in range(num_blocks):
            blocks.append(
                DilatedTemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=tcn_units,
                    kernel_sizes=kernel_sizes,
                    dilations=dilations,
                    dropout=dropout_tcn,
                )
            )
            in_channels = tcn_units
        self.temporal_net = nn.Sequential(*blocks)

        self.attention = AttentionPooling(tcn_units) if self.pooling == "attention" else None
        dyn_dim = tcn_units * 2 if self.pooling == "mean_max" else tcn_units

        (
            self.embedding_layers,
            categorical_dim,
            self.categorical_mode,
            self.n_categorical,
        ) = _build_categorical_layers(categorical_embeddings, n_categorical, categorical_mode)
        static_input_dim = n_static + categorical_dim
        if static_input_dim <= 0:
            raise ValueError("Static feature dimension must be positive after adding categorical inputs")
        static_layers = [
            nn.Linear(static_input_dim, static_units, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(static_units),
        ]
        if dropout_static > 0:
            static_layers.append(nn.Dropout(dropout_static))
        self.static_net = nn.Sequential(*static_layers)

        half_units = max(merge_units // 2, 1)
        quarter_units = max(merge_units // 4, 1)
        merged_layers: list[nn.Module] = [
            nn.Linear(dyn_dim + static_units, merge_units, bias=False),
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

    def _pool_temporal_features(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.pooling == "attention":
            if self.attention is None:
                raise RuntimeError("Attention pooling was not initialized")
            return self.attention(encoded)
        if self.pooling == "last":
            return encoded[:, -1, :]
        if self.pooling == "mean":
            return encoded.mean(dim=1)
        if self.pooling == "max":
            return torch.amax(encoded, dim=1)
        if self.pooling == "mean_max":
            return torch.cat([encoded.mean(dim=1), torch.amax(encoded, dim=1)], dim=1)
        raise AssertionError(f"Unhandled pooling mode: {self.pooling}")

    def forward(self, dyn: torch.Tensor, stat: torch.Tensor, cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if dyn.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {dyn.size(1)}")
        temporal = dyn.transpose(1, 2)
        encoded = self.temporal_net(temporal).transpose(1, 2)
        dyn_feat = self._pool_temporal_features(encoded)

        stat = _concat_categorical_features(
            self.embedding_layers,
            stat,
            cat,
            self.categorical_mode,
            self.n_categorical,
        )
        stat_feat = self.static_net(stat)
        merged = torch.cat([dyn_feat, stat_feat], dim=1)
        logits = self.out(self.mlp(merged))
        return logits.squeeze(-1)


@register_model("spatial_tsn_mlp")
@register_model("tsn_spatial_mlp")
class SpatialTemporalConvNetModel(nn.Module):
    """Temporal-convolution encoder with a dedicated local spatial-context branch."""

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_static: int,
        tcn_units: int = 192,
        num_blocks: int = 3,
        kernel_sizes: tuple[int, ...] | list[int] = (2, 3, 5),
        dilations: tuple[int, ...] | list[int] = (1, 2, 4),
        dropout_tcn: float = 0.1,
        pooling: str = "attention",
        static_units: int = 512,
        dropout_static: float = 0.0,
        spatial_units: int = 192,
        dropout_spatial: float = 0.1,
        merge_units: int = 256,
        dropout_merged: float = 0.0,
        spatial_static_indices: list[int] | tuple[int, ...] | None = None,
        categorical_embeddings: Optional[List[Dict]] = None,
        n_categorical: int = 0,
        categorical_mode: str = "auto",
    ) -> None:
        super().__init__()
        if seq_len <= 0 or n_channels <= 0:
            raise ValueError("'seq_len' and 'n_channels' must be positive")
        if tcn_units <= 0 or num_blocks <= 0 or static_units <= 0 or merge_units <= 0:
            raise ValueError("tcn_units, num_blocks, static_units, and merge_units must be positive")
        if spatial_units <= 0:
            raise ValueError("spatial_units must be positive")

        self.seq_len = int(seq_len)
        kernel_sizes = _as_int_tuple(kernel_sizes, "kernel_sizes")
        dilations = _as_int_tuple(dilations, "dilations")
        self.pooling = _normalize_pooling_mode(pooling)

        spatial_indices = sorted({int(idx) for idx in (spatial_static_indices or [])})
        if not spatial_indices:
            raise ValueError("SpatialTemporalConvNetModel requires non-empty spatial_static_indices")
        if spatial_indices[0] < 0 or spatial_indices[-1] >= n_static:
            raise ValueError(
                f"spatial_static_indices must be within [0, {n_static - 1}], got {spatial_indices}"
            )
        spatial_set = set(spatial_indices)
        regular_indices = [idx for idx in range(n_static) if idx not in spatial_set]
        self.register_buffer(
            "spatial_static_indices",
            torch.as_tensor(spatial_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "regular_static_indices",
            torch.as_tensor(regular_indices, dtype=torch.long),
            persistent=False,
        )

        blocks: list[nn.Module] = []
        in_channels = n_channels
        for _ in range(num_blocks):
            blocks.append(
                DilatedTemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=tcn_units,
                    kernel_sizes=kernel_sizes,
                    dilations=dilations,
                    dropout=dropout_tcn,
                )
            )
            in_channels = tcn_units
        self.temporal_net = nn.Sequential(*blocks)
        self.attention = AttentionPooling(tcn_units) if self.pooling == "attention" else None
        dyn_dim = tcn_units * 2 if self.pooling == "mean_max" else tcn_units

        (
            self.embedding_layers,
            categorical_dim,
            self.categorical_mode,
            self.n_categorical,
        ) = _build_categorical_layers(categorical_embeddings, n_categorical, categorical_mode)

        regular_static_input_dim = len(regular_indices) + categorical_dim
        if regular_static_input_dim > 0:
            static_layers = [
                nn.Linear(regular_static_input_dim, static_units, bias=True),
                nn.ReLU(),
                nn.BatchNorm1d(static_units),
            ]
            if dropout_static > 0:
                static_layers.append(nn.Dropout(dropout_static))
            self.static_net: nn.Module | None = nn.Sequential(*static_layers)
            static_out_dim = static_units
        else:
            self.static_net = None
            static_out_dim = 0

        spatial_layers = [
            nn.Linear(len(spatial_indices), spatial_units, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(spatial_units),
        ]
        if dropout_spatial > 0:
            spatial_layers.append(nn.Dropout(dropout_spatial))
        spatial_layers.extend(
            [
                nn.Linear(spatial_units, spatial_units, bias=False),
                nn.BatchNorm1d(spatial_units),
                nn.ReLU(),
            ]
        )
        self.spatial_net = nn.Sequential(*spatial_layers)

        half_units = max(merge_units // 2, 1)
        quarter_units = max(merge_units // 4, 1)
        merged_layers: list[nn.Module] = [
            nn.Linear(dyn_dim + static_out_dim + spatial_units, merge_units, bias=False),
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

    def _pool_temporal_features(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.pooling == "attention":
            if self.attention is None:
                raise RuntimeError("Attention pooling was not initialized")
            return self.attention(encoded)
        if self.pooling == "last":
            return encoded[:, -1, :]
        if self.pooling == "mean":
            return encoded.mean(dim=1)
        if self.pooling == "max":
            return torch.amax(encoded, dim=1)
        if self.pooling == "mean_max":
            return torch.cat([encoded.mean(dim=1), torch.amax(encoded, dim=1)], dim=1)
        raise AssertionError(f"Unhandled pooling mode: {self.pooling}")

    def forward(self, dyn: torch.Tensor, stat: torch.Tensor, cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if dyn.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {dyn.size(1)}")
        temporal = dyn.transpose(1, 2)
        encoded = self.temporal_net(temporal).transpose(1, 2)
        dyn_feat = self._pool_temporal_features(encoded)

        spatial_input = stat.index_select(1, self.spatial_static_indices)
        spatial_feat = self.spatial_net(spatial_input)

        features = [dyn_feat]
        if self.static_net is not None:
            regular_stat = stat.index_select(1, self.regular_static_indices)
            regular_stat = _concat_categorical_features(
                self.embedding_layers,
                regular_stat,
                cat,
                self.categorical_mode,
                self.n_categorical,
            )
            features.append(self.static_net(regular_stat))
        features.append(spatial_feat)

        logits = self.out(self.mlp(torch.cat(features, dim=1)))
        return logits.squeeze(-1)


@register_model("spatial_climate_tsn_mlp")
@register_model("tsn_spatial_climate_mlp")
class SpatialClimateTemporalConvNetModel(nn.Module):
    """Encode daily spatial climate patches, then model the 128-day sequence."""

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_static: int,
        spatial_height: int = 3,
        spatial_width: int = 3,
        spatial_conv_units: int = 48,
        spatial_patch_units: int = 96,
        tcn_units: int = 256,
        num_blocks: int = 3,
        kernel_sizes: tuple[int, ...] | list[int] = (3, 5, 9),
        dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8),
        dropout_spatial: float = 0.05,
        dropout_tcn: float = 0.12,
        pooling: str = "attention",
        static_units: int = 512,
        dropout_static: float = 0.05,
        merge_units: int = 448,
        dropout_merged: float = 0.15,
        categorical_embeddings: Optional[List[Dict]] = None,
        n_categorical: int = 0,
        categorical_mode: str = "auto",
    ) -> None:
        super().__init__()
        if seq_len <= 0 or n_channels <= 0:
            raise ValueError("'seq_len' and 'n_channels' must be positive")
        if spatial_height <= 0 or spatial_width <= 0:
            raise ValueError("'spatial_height' and 'spatial_width' must be positive")
        if spatial_conv_units <= 0 or spatial_patch_units <= 0:
            raise ValueError("spatial_conv_units and spatial_patch_units must be positive")
        if tcn_units <= 0 or num_blocks <= 0 or static_units <= 0 or merge_units <= 0:
            raise ValueError("tcn_units, num_blocks, static_units, and merge_units must be positive")

        self.seq_len = int(seq_len)
        self.n_channels = int(n_channels)
        self.spatial_height = int(spatial_height)
        self.spatial_width = int(spatial_width)
        kernel_sizes = _as_int_tuple(kernel_sizes, "kernel_sizes")
        dilations = _as_int_tuple(dilations, "dilations")
        self.pooling = _normalize_pooling_mode(pooling)

        patch_layers: list[nn.Module] = [
            nn.Conv2d(n_channels, spatial_conv_units, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(spatial_conv_units),
            nn.GELU(),
            nn.Conv2d(spatial_conv_units, spatial_conv_units, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(spatial_conv_units),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(spatial_conv_units, spatial_patch_units, bias=True),
            nn.GELU(),
        ]
        if dropout_spatial > 0:
            patch_layers.append(nn.Dropout(dropout_spatial))
        self.patch_encoder = nn.Sequential(*patch_layers)

        blocks: list[nn.Module] = []
        in_channels = spatial_patch_units
        for _ in range(num_blocks):
            blocks.append(
                DilatedTemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=tcn_units,
                    kernel_sizes=kernel_sizes,
                    dilations=dilations,
                    dropout=dropout_tcn,
                )
            )
            in_channels = tcn_units
        self.temporal_net = nn.Sequential(*blocks)
        self.attention = AttentionPooling(tcn_units) if self.pooling == "attention" else None
        dyn_dim = tcn_units * 2 if self.pooling == "mean_max" else tcn_units

        (
            self.embedding_layers,
            categorical_dim,
            self.categorical_mode,
            self.n_categorical,
        ) = _build_categorical_layers(categorical_embeddings, n_categorical, categorical_mode)
        static_input_dim = n_static + categorical_dim
        if static_input_dim <= 0:
            raise ValueError("Static feature dimension must be positive after adding categorical inputs")
        static_layers = [
            nn.Linear(static_input_dim, static_units, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(static_units),
        ]
        if dropout_static > 0:
            static_layers.append(nn.Dropout(dropout_static))
        self.static_net = nn.Sequential(*static_layers)

        half_units = max(merge_units // 2, 1)
        quarter_units = max(merge_units // 4, 1)
        merged_layers: list[nn.Module] = [
            nn.Linear(dyn_dim + static_units, merge_units, bias=False),
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

    def _pool_temporal_features(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.pooling == "attention":
            if self.attention is None:
                raise RuntimeError("Attention pooling was not initialized")
            return self.attention(encoded)
        if self.pooling == "last":
            return encoded[:, -1, :]
        if self.pooling == "mean":
            return encoded.mean(dim=1)
        if self.pooling == "max":
            return torch.amax(encoded, dim=1)
        if self.pooling == "mean_max":
            return torch.cat([encoded.mean(dim=1), torch.amax(encoded, dim=1)], dim=1)
        raise AssertionError(f"Unhandled pooling mode: {self.pooling}")

    def forward(self, dyn: torch.Tensor, stat: torch.Tensor, cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if dyn.ndim != 5:
            raise ValueError(
                "SpatialClimateTemporalConvNetModel expects dynamic input shaped "
                "(batch, time, height, width, channels)."
            )
        if dyn.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {dyn.size(1)}")
        if dyn.size(2) != self.spatial_height or dyn.size(3) != self.spatial_width:
            raise ValueError(
                f"Expected spatial patch {(self.spatial_height, self.spatial_width)}, "
                f"got {(dyn.size(2), dyn.size(3))}"
            )
        if dyn.size(4) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} climate channels, got {dyn.size(4)}")

        batch_size, seq_len, height, width, channels = dyn.shape
        patches = dyn.reshape(batch_size * seq_len, height, width, channels)
        patches = patches.permute(0, 3, 1, 2).contiguous()
        patch_features = self.patch_encoder(patches).reshape(batch_size, seq_len, -1)

        temporal = patch_features.transpose(1, 2)
        encoded = self.temporal_net(temporal).transpose(1, 2)
        dyn_feat = self._pool_temporal_features(encoded)

        stat = _concat_categorical_features(
            self.embedding_layers,
            stat,
            cat,
            self.categorical_mode,
            self.n_categorical,
        )
        stat_feat = self.static_net(stat)
        logits = self.out(self.mlp(torch.cat([dyn_feat, stat_feat], dim=1)))
        return logits.squeeze(-1)


@register_model("lstm_gated_moe")
class LSTMGatedMoEModel(nn.Module):
    """Global LSTM model with learned mixture-of-experts fusion.

    The gate is conditioned on the shared dynamic/static representation, so the
    model can learn different fire-risk regimes without requiring per-region
    thresholds or separate regional models.
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_static: int,
        lstm_units: int = 192,
        num_layers: int = 1,
        bidirectional: bool = True,
        use_attention: bool = True,
        dropout_lstm: float = 0.1,
        static_units: int = 384,
        dropout_static: float = 0.1,
        fusion_units: int = 384,
        dropout_merged: float = 0.15,
        num_experts: int = 4,
        expert_units: int = 192,
        gate_hidden_units: int = 192,
        categorical_embeddings: Optional[List[Dict]] = None,
        n_categorical: int = 0,
        categorical_mode: str = "auto",
    ) -> None:
        super().__init__()
        if seq_len <= 0 or n_channels <= 0:
            raise ValueError("'seq_len' and 'n_channels' must be positive")
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        if fusion_units <= 0 or expert_units <= 0 or gate_hidden_units <= 0:
            raise ValueError("fusion, expert, and gate hidden dimensions must be positive")

        self.use_attention = bool(use_attention)
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=lstm_units,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
        )
        direction_factor = 2 if bidirectional else 1
        dyn_dim = lstm_units * direction_factor
        self.attention = AttentionPooling(dyn_dim) if self.use_attention else None
        self.dropout_lstm = nn.Dropout(dropout_lstm) if dropout_lstm > 0 else nn.Identity()

        (
            self.embedding_layers,
            categorical_dim,
            self.categorical_mode,
            self.n_categorical,
        ) = _build_categorical_layers(categorical_embeddings, n_categorical, categorical_mode)
        static_input_dim = n_static + categorical_dim
        if static_input_dim <= 0:
            raise ValueError("Static feature dimension must be positive after adding categorical inputs")

        static_layers = [
            nn.Linear(static_input_dim, static_units, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(static_units),
        ]
        if dropout_static > 0:
            static_layers.append(nn.Dropout(dropout_static))
        self.static_net = nn.Sequential(*static_layers)

        fusion_layers = [
            nn.Linear(dyn_dim + static_units, fusion_units, bias=False),
            nn.BatchNorm1d(fusion_units),
            nn.ReLU(),
        ]
        if dropout_merged > 0:
            fusion_layers.append(nn.Dropout(dropout_merged))
        self.fusion_net = nn.Sequential(*fusion_layers)

        def make_expert() -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Linear(fusion_units, expert_units),
                nn.ReLU(),
            ]
            if dropout_merged > 0:
                layers.append(nn.Dropout(dropout_merged))
            layers.append(nn.Linear(expert_units, 1))
            return nn.Sequential(*layers)

        self.experts = nn.ModuleList([make_expert() for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(fusion_units, gate_hidden_units),
            nn.ReLU(),
            nn.Linear(gate_hidden_units, num_experts),
        )

    def forward(self, dyn: torch.Tensor, stat: torch.Tensor, cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        lstm_out, (hidden, _) = self.lstm(dyn)
        if self.attention is not None:
            dyn_feat = self.attention(lstm_out)
        else:
            dyn_feat = hidden[-1]
        dyn_feat = self.dropout_lstm(dyn_feat)

        stat = _concat_categorical_features(
            self.embedding_layers,
            stat,
            cat,
            self.categorical_mode,
            self.n_categorical,
        )
        stat_feat = self.static_net(stat)
        fused = self.fusion_net(torch.cat([dyn_feat, stat_feat], dim=1))

        gate_weights = torch.softmax(self.gate(fused), dim=1)
        expert_logits = torch.cat([expert(fused) for expert in self.experts], dim=1)
        logits = (gate_weights * expert_logits).sum(dim=1)
        return logits


__all__ = [
    "MLPModel",
    "MinimalMLPModel",
    "FTTransformerModel",
    "TemporalConvNetModel",
    "SpatialTemporalConvNetModel",
    "SpatialClimateTemporalConvNetModel",
    "available_models",
    "build_model",
    "register_model",
    "LSTMMlpModel",
    "LSTMEarlyFusionModel",
    "LSTMAttentionModel",
    "LSTMGatedMoEModel",
]
