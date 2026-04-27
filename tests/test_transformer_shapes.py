import torch
import pytest

from ar_image_generation.models.transformer import TransformerBackbone


def test_transformer_backbone_shape() -> None:
    model = TransformerBackbone(
        dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
    )

    x = torch.randn(2, 10, 32)
    attention_mask = torch.ones(10, 10, dtype=torch.bool)

    y = model(x, attention_mask=attention_mask)

    assert y.shape == x.shape


def test_transformer_rejects_invalid_head_count() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        TransformerBackbone(
            dim=30,
            depth=2,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
        )
