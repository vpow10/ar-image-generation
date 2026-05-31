import math

import torch

from ar_image_generation.approaches.base import SamplingConfig
from ar_image_generation.approaches.maskgit.model import MaskGITApproach, mask_ratio
from ar_image_generation.approaches.registry import build_approach
from ar_image_generation.data.batch import ImageBatch
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


def build_small_maskgit() -> MaskGITApproach:
    return MaskGITApproach(
        vocab_size=VOCAB_SIZE,
        latent_shape=LATENT_SHAPE,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=2.0,
        dropout=0.0,
        num_iterations=4,
    )


def test_maskgit_forward_shapes() -> None:
    model = build_small_maskgit()
    num_tokens = LATENT_SHAPE[0] * LATENT_SHAPE[1]
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, num_tokens))

    logits = model(tokens)

    assert logits.shape == (BATCH_SIZE, num_tokens, VOCAB_SIZE)


def test_maskgit_training_step_loss_finite() -> None:
    model = build_small_maskgit()
    tokenizer = DummyTokenizer()
    batch = ImageBatch(images=torch.randn(BATCH_SIZE, 3, 64, 64))

    output = model.training_step(batch, tokenizer)

    assert output["loss"].ndim == 0
    assert output["loss"].isfinite()


def test_maskgit_generate_shapes() -> None:
    model = build_small_maskgit()
    tokenizer = DummyTokenizer()

    images = model.generate(
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        labels=None,
        device=torch.device("cpu"),
        sampling_cfg=SamplingConfig(temperature=1.0, top_k=10, top_p=None, num_samples=BATCH_SIZE),
    )

    assert images.shape == (BATCH_SIZE, 3, 64, 64)


def test_maskgit_class_conditional_generate_shapes() -> None:
    num_classes = 9
    model = MaskGITApproach(
        vocab_size=VOCAB_SIZE,
        latent_shape=LATENT_SHAPE,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=2.0,
        dropout=0.0,
        num_iterations=4,
        class_conditional=True,
        num_classes=num_classes,
    )
    tokenizer = DummyTokenizer()
    batch = ImageBatch(
        images=torch.randn(BATCH_SIZE, 3, 64, 64),
        labels=torch.randint(0, num_classes, (BATCH_SIZE,)),
    )

    output = model.training_step(batch, tokenizer)
    assert output["loss"].isfinite()

    images = model.generate(
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        labels=None,
        device=torch.device("cpu"),
        sampling_cfg=SamplingConfig(temperature=1.0, top_k=10, top_p=None, num_samples=BATCH_SIZE),
    )
    assert images.shape == (BATCH_SIZE, 3, 64, 64)


def test_build_maskgit_from_registry() -> None:
    approach = build_approach(
        "maskgit",
        vocab_size=VOCAB_SIZE,
        latent_shape=LATENT_SHAPE,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=2.0,
        dropout=0.0,
        class_conditional=False,
    )

    assert isinstance(approach, MaskGITApproach)


def test_mask_ratio_cosine_endpoints() -> None:
    assert math.isclose(mask_ratio(0.0, "cosine"), 1.0)
    assert math.isclose(mask_ratio(1.0, "cosine"), 0.0, abs_tol=1e-6)
