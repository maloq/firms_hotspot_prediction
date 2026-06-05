from __future__ import annotations

import torch

from src.neural_net.models.architectures import FeatureTokenEncoder, available_models, build_model


CAT_META = [
    {"name": "ecoregion_name", "cardinality": 5, "embedding_dim": 3},
    {"name": "ecoregion_realm", "cardinality": 4, "embedding_dim": 2},
]


def test_feature_token_encoder_projects_continuous_and_categorical_tokens() -> None:
    encoder = FeatureTokenEncoder(
        n_continuous=3,
        token_dim=8,
        categorical_embeddings=CAT_META,
        n_categorical=2,
        categorical_mode="embedding",
    )

    stat = torch.randn(4, 3)
    cat = torch.tensor([[0, 1], [2, 3], [4, 0], [1, 2]], dtype=torch.long)

    tokens = encoder(stat, cat)

    assert tokens.shape == (4, 5, 8)
    assert torch.isfinite(tokens).all()


def test_embedding_fusion_architectures_are_registered() -> None:
    names = set(available_models())

    assert {"lstm_embedding_fusion", "tsn_embedding_fusion", "spatial_climate_tsn_embedding_fusion"} <= names


def test_lstm_embedding_fusion_forward_shape() -> None:
    model = build_model(
        "lstm_embedding_fusion",
        seq_len=4,
        n_channels=3,
        n_static=5,
        lstm_units=8,
        bidirectional=True,
        token_dim=8,
        token_layers=1,
        token_heads=2,
        token_ffn_dim=16,
        static_units=10,
        merge_units=16,
        categorical_embeddings=CAT_META,
        n_categorical=2,
        categorical_mode="embedding",
    ).eval()

    logits = model(torch.randn(3, 4, 3), torch.randn(3, 5), torch.randint(0, 4, (3, 2)))

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_tsn_embedding_fusion_forward_shape() -> None:
    model = build_model(
        "tsn_embedding_fusion",
        seq_len=5,
        n_channels=3,
        n_static=5,
        token_dim=8,
        tcn_units=12,
        num_blocks=1,
        kernel_sizes=[2],
        dilations=[1],
        pooling="mean",
        token_layers=1,
        token_heads=2,
        token_ffn_dim=16,
        static_units=10,
        merge_units=16,
        categorical_embeddings=CAT_META,
        n_categorical=2,
        categorical_mode="embedding",
    ).eval()

    logits = model(torch.randn(3, 5, 3), torch.randn(3, 5), torch.randint(0, 4, (3, 2)))

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_spatial_climate_tsn_embedding_fusion_forward_shape() -> None:
    model = build_model(
        "spatial_climate_tsn_embedding_fusion",
        seq_len=3,
        n_channels=2,
        n_static=5,
        spatial_height=3,
        spatial_width=3,
        spatial_conv_units=4,
        spatial_patch_units=8,
        token_dim=8,
        tcn_units=8,
        num_blocks=1,
        kernel_sizes=[2],
        dilations=[1],
        pooling="mean",
        token_layers=1,
        token_heads=2,
        token_ffn_dim=16,
        static_units=10,
        merge_units=16,
        categorical_embeddings=CAT_META,
        n_categorical=2,
        categorical_mode="embedding",
    ).eval()

    logits = model(torch.randn(3, 3, 3, 3, 2), torch.randn(3, 5), torch.randint(0, 4, (3, 2)))

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
