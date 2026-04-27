import torch

from ar_image_generation.approaches.var.sampler import (
    apply_top_k_top_p_filtering,
    sample_tokens_from_logits,
)


def test_top_k_filtering_keeps_expected_number_of_tokens() -> None:
    logits = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10)

    filtered = apply_top_k_top_p_filtering(
        logits,
        top_k=3,
        top_p=None,
    )

    kept = torch.isfinite(filtered).sum().item()

    assert kept == 3


def test_sample_tokens_from_logits_shape() -> None:
    logits = torch.randn(2, 5, 16)

    sampled = sample_tokens_from_logits(
        logits,
        temperature=1.0,
        top_k=8,
        top_p=None,
    )

    assert sampled.shape == (2, 5)
    assert sampled.dtype == torch.long
