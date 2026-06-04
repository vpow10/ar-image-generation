import torch

from ar_image_generation.tokenizers.graft_gs import (
    GaussianSplatTokenizer,
    GaussianTokenizerConfig,
)


def test_gaussian_tokenizer_forward_shapes() -> None:
    cfg = GaussianTokenizerConfig(
        image_size=32,
        image_channels=3,
        num_primitives=16,
        primitive_feature_dim=8,
        hidden_channels=32,
        renderer_chunk_size=8,
    )

    model = GaussianSplatTokenizer(cfg)

    images = torch.randn(2, 3, 32, 32)
    output = model(images)

    assert output.reconstructions.shape == images.shape
    assert output.raw_primitives.shape == (2, 16, 14)
    assert output.primitives.position.shape == (2, 16, 2)
    assert output.primitives.scale.shape == (2, 16, 2)
    assert output.primitives.rotation.shape == (2, 16, 1)
    assert output.primitives.opacity.shape == (2, 16, 1)
    assert output.primitives.feature.shape == (2, 16, 8)
    assert output.feature_map.shape == (2, 9, 32, 32)
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def test_gaussian_tokenizer_encode_decode_shapes() -> None:
    cfg = GaussianTokenizerConfig(
        image_size=32,
        image_channels=3,
        num_primitives=16,
        primitive_feature_dim=8,
        hidden_channels=32,
        renderer_chunk_size=8,
    )

    model = GaussianSplatTokenizer(cfg)

    images = torch.randn(2, 3, 32, 32)

    primitives = model.encode(images)
    reconstructions = model.decode(primitives)

    assert primitives.position.shape == (2, 16, 2)
    assert reconstructions.shape == images.shape


def test_gaussian_tokenizer_backward() -> None:
    cfg = GaussianTokenizerConfig(
        image_size=32,
        image_channels=3,
        num_primitives=16,
        primitive_feature_dim=8,
        hidden_channels=32,
        renderer_chunk_size=8,
    )

    model = GaussianSplatTokenizer(cfg)

    images = torch.randn(2, 3, 32, 32)
    output = model(images)

    output.loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    assert len(gradients) > 0
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
