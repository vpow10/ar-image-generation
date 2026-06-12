import pytest
import torch

from ar_image_generation.approaches.base import SamplingConfig
from ar_image_generation.approaches.raster.model import RasterApproach
from ar_image_generation.approaches.registry import build_approach
from ar_image_generation.tokenizers.base import ImageTokenizer

VOCAB_SIZE = 64
LATENT_SHAPE = (4, 4)
DIM = 32
DEPTH = 2
NUM_HEADS = 4
BATCH_SIZE = 2


class DummyTokenizer(ImageTokenizer):
    def __init__(self, vocab_size: int = VOCAB_SIZE, latent_shape: tuple[int, int] = LATENT_SHAPE) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_shape = latent_shape

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.LongTensor:
        return torch.zeros(
            images.shape[0],
            self.latent_shape[0],
            self.latent_shape[1],
            device=images.device,
            dtype=torch.long,
        )

    @torch.no_grad()
    def decode(self, tokens: torch.LongTensor) -> torch.Tensor:
        return torch.zeros(tokens.shape[0], 3, 64, 64, device=tokens.device, dtype=torch.float32)


def build_small_raster() -> RasterApproach:
    return RasterApproach(
        vocab_size=VOCAB_SIZE,
        latent_shape=LATENT_SHAPE,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=2.0,
        dropout=0.0,
    )


def test_raster_forward_shapes() -> None:
    model = build_small_raster()
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, *LATENT_SHAPE))

    output = model(tokens)

    num_tokens = LATENT_SHAPE[0] * LATENT_SHAPE[1]
    assert output.logits.shape == (BATCH_SIZE, num_tokens, VOCAB_SIZE)
    assert output.targets.shape == (BATCH_SIZE, num_tokens)
    assert output.loss.ndim == 0


def test_raster_forward_loss_finite() -> None:
    model = build_small_raster()
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, *LATENT_SHAPE))
    output = model(tokens)
    assert output.loss.isfinite()


def test_raster_generate_shapes() -> None:
    model = build_small_raster()
    tokenizer = DummyTokenizer()

    images = model.generate(
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        labels=None,
        device=torch.device("cpu"),
        sampling_cfg=SamplingConfig(temperature=1.0, top_k=10, top_p=None, num_samples=BATCH_SIZE),
    )

    assert images.shape == (BATCH_SIZE, 3, 64, 64)


def test_raster_class_conditional_raises() -> None:
    with pytest.raises(NotImplementedError):
        RasterApproach(
            vocab_size=VOCAB_SIZE,
            latent_shape=LATENT_SHAPE,
            class_conditional=True,
        )


def test_build_raster_from_registry() -> None:
    approach = build_approach(
        "raster",
        vocab_size=VOCAB_SIZE,
        latent_shape=LATENT_SHAPE,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=2.0,
        dropout=0.0,
        class_conditional=False,
    )

    assert isinstance(approach, RasterApproach)
