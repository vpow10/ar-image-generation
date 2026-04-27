import torch

from ar_image_generation.tokenizers.vqvae import VQVAE


def test_vqvae_forward_shapes() -> None:
    model = VQVAE(
        image_channels=3,
        image_size=64,
        vocab_size=64,
        embedding_dim=32,
        hidden_channels=32,
        downsample_factor=8,
        commitment_cost=0.25,
    )

    images = torch.randn(2, 3, 64, 64)
    output = model(images)

    assert output.reconstructions.shape == images.shape
    assert output.indices.shape == (2, 8, 8)
    assert output.loss.ndim == 0
    assert output.reconstruction_loss.ndim == 0
    assert output.quantization_loss.ndim == 0
    assert output.commitment_loss.ndim == 0
    assert output.perplexity.ndim == 0


def test_vqvae_encode_decode_shapes() -> None:
    model = VQVAE(
        image_channels=3,
        image_size=64,
        vocab_size=64,
        embedding_dim=32,
        hidden_channels=32,
        downsample_factor=8,
        commitment_cost=0.25,
    )

    images = torch.randn(2, 3, 64, 64)

    tokens = model.encode(images)
    reconstructions = model.decode(tokens)

    assert tokens.shape == (2, 8, 8)
    assert tokens.dtype == torch.long
    assert reconstructions.shape == images.shape
