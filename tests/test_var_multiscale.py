import torch
import pytest

from ar_image_generation.approaches.var.multiscale import (
    build_image_pyramid_token_grids,
    build_multiscale_sequence_from_images,
)
from ar_image_generation.approaches.var.schedule import VARSchedule
from ar_image_generation.tokenizers.base import ImageTokenizer


class DummyTokenizer(ImageTokenizer):
    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_shape = (8, 8)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.LongTensor:
        latent_size = images.shape[-1] // 8

        return torch.zeros(
            images.shape[0],
            latent_size,
            latent_size,
            device=images.device,
            dtype=torch.long,
        )

    @torch.no_grad()
    def decode(self, tokens: torch.LongTensor) -> torch.Tensor:
        return torch.zeros(tokens.shape[0], 3, 64, 64)


def test_var_schedule_properties() -> None:
    schedule = VARSchedule(scales=(1, 2, 4, 8))

    assert schedule.final_scale == 8
    assert schedule.num_scales == 4
    assert schedule.num_tokens == 1 + 4 + 16 + 64


def test_var_schedule_rejects_non_increasing_scales() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        VARSchedule(scales=(1, 4, 4, 8))


def test_build_image_pyramid_token_grids_shapes() -> None:
    schedule = VARSchedule(scales=(1, 2, 4, 8))
    tokenizer = DummyTokenizer()

    images = torch.randn(2, 3, 64, 64)

    grids = build_image_pyramid_token_grids(images, tokenizer, schedule)

    assert len(grids) == 4
    assert grids[0].shape == (2, 1, 1)
    assert grids[1].shape == (2, 2, 2)
    assert grids[2].shape == (2, 4, 4)
    assert grids[3].shape == (2, 8, 8)


def test_build_multiscale_sequence_from_images_shape() -> None:
    schedule = VARSchedule(scales=(1, 2, 4, 8))
    tokenizer = DummyTokenizer()

    images = torch.randn(2, 3, 64, 64)

    sequence = build_multiscale_sequence_from_images(images, tokenizer, schedule)

    assert sequence.shape == (2, 85)
