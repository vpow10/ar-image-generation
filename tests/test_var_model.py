import torch

from ar_image_generation.approaches.base import SamplingConfig
from ar_image_generation.approaches.registry import build_approach
from ar_image_generation.approaches.var.model import VARApproach
from ar_image_generation.tokenizers.base import ImageTokenizer


class DummyTokenizer(ImageTokenizer):
    def __init__(self, vocab_size: int = 64, latent_shape: tuple[int, int] = (8, 8)) -> None:
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


def test_var_forward_shapes() -> None:
    model = build_small_var()
    tokens = torch.randint(low=0, high=64, size=(2, 8, 8), dtype=torch.long)

    output = model(tokens)

    assert output.logits.shape == (2, 85, 64)
    assert output.targets.shape == (2, 85)
    assert output.loss.ndim == 0


def test_var_predict_single_scale_shape() -> None:
    model = build_small_var()

    logits = model.predict_scale_logits(
        context_tokens=None,
        scale_index=0,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert logits.shape == (2, 1, 64)


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
