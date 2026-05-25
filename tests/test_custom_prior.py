import torch

from ar_image_generation.approaches.custom.model import (
    GaussianPrimitiveAR,
    GaussianPrimitiveARConfig,
)
from ar_image_generation.approaches.custom.normalization import PrimitiveNormalizer


def test_primitive_normalizer_roundtrip() -> None:
    primitives = torch.randn(2, 4, 6)
    mean = primitives.reshape(-1, 6).mean(dim=0)
    std = primitives.reshape(-1, 6).std(dim=0).clamp_min(1e-6)

    normalizer = PrimitiveNormalizer(mean=mean, std=std)

    normalized = normalizer.normalize(primitives)
    restored = normalizer.denormalize(normalized)

    assert torch.allclose(primitives, restored, atol=1e-5)


def test_gaussian_primitive_ar_forward_shapes() -> None:
    cfg = GaussianPrimitiveARConfig(
        num_primitives=16,
        primitive_dim=14,
        dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        class_conditional=True,
        num_classes=9,
    )

    model = GaussianPrimitiveAR(cfg)

    primitives = torch.randn(2, 16, 14)
    labels = torch.tensor([0, 1])

    output = model(primitives, labels)

    assert output.mean.shape == primitives.shape
    assert output.log_std.shape == primitives.shape
    assert output.loss.ndim == 0
    assert output.nll_loss.ndim == 0
    assert output.mean_loss.ndim == 0


def test_gaussian_primitive_ar_generate_shapes() -> None:
    cfg = GaussianPrimitiveARConfig(
        num_primitives=16,
        primitive_dim=14,
        dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        class_conditional=True,
        num_classes=9,
    )

    model = GaussianPrimitiveAR(cfg)

    generated = model.generate(
        batch_size=2,
        labels=None,
        device=torch.device("cpu"),
        temperature=0.85,
    )

    assert generated.shape == (2, 16, 14)
    assert torch.isfinite(generated).all()
