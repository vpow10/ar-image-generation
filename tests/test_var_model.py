import torch

from ar_image_generation.approaches.base import SamplingConfig
from ar_image_generation.approaches.registry import build_approach
from ar_image_generation.approaches.var.model import VARApproach
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.tokenizers.base import ImageTokenizer


class DummyTokenizer(ImageTokenizer):
    def __init__(self, vocab_size: int = 64, latent_shape: tuple[int, int] = (8, 8)) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_shape = latent_shape

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.LongTensor:
        height = images.shape[-2]

        if height % 8 != 0:
            raise ValueError("Dummy tokenizer expects image height divisible by 8.")

        latent_size = height // 8

        return torch.zeros(
            images.shape[0],
            latent_size,
            latent_size,
            device=images.device,
            dtype=torch.long,
        )

    @torch.no_grad()
    def decode(self, tokens: torch.LongTensor) -> torch.Tensor:
        return torch.zeros(
            tokens.shape[0],
            3,
            64,
            64,
            device=tokens.device,
            dtype=torch.float32,
        )


def build_small_var() -> VARApproach:
    return VARApproach(
        vocab_size=64,
        latent_shape=(8, 8),
        dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        scales=(1, 2, 4, 8),
        class_conditional=False,
    )


def test_var_forward_sequence_shapes() -> None:
    model = build_small_var()

    targets = torch.randint(low=0, high=64, size=(2, 85), dtype=torch.long)

    output = model.forward_sequence(targets)

    assert output.logits.shape == (2, 85, 64)
    assert output.targets.shape == (2, 85)
    assert output.loss.ndim == 0


def test_var_training_step_returns_scalar_loss() -> None:
    model = build_small_var()
    tokenizer = DummyTokenizer(vocab_size=64, latent_shape=(8, 8))

    batch = ImageBatch(
        images=torch.randn(2, 3, 64, 64),
        labels=None,
        metadata=None,
    )

    output = model.training_step(batch, tokenizer)

    assert "loss" in output
    assert output["loss"].ndim == 0


def test_var_generate_shapes() -> None:
    model = build_small_var()
    tokenizer = DummyTokenizer(vocab_size=64, latent_shape=(8, 8))

    images = model.generate(
        tokenizer=tokenizer,
        batch_size=2,
        labels=None,
        device=torch.device("cpu"),
        sampling_cfg=SamplingConfig(
            temperature=1.0,
            top_k=10,
            top_p=None,
            num_samples=2,
        ),
    )

    assert images.shape == (2, 3, 64, 64)


def test_build_var_from_registry() -> None:
    approach = build_approach(
        "var",
        vocab_size=64,
        latent_shape=(8, 8),
        dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        scales=(1, 2, 4, 8),
        class_conditional=False,
    )

    assert isinstance(approach, VARApproach)
