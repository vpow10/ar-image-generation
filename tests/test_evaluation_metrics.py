import torch
from torch import nn

from ar_image_generation.evaluation.metrics import count_parameters, measure_sampling_speed


def test_count_parameters() -> None:
    model = nn.Linear(4, 2)

    expected = 4 * 2 + 2

    assert count_parameters(model, trainable_only=True) == expected
    assert count_parameters(model, trainable_only=False) == expected


def test_count_parameters_trainable_only() -> None:
    model = nn.Linear(4, 2)
    model.weight.requires_grad_(False)

    assert count_parameters(model, trainable_only=True) == 2
    assert count_parameters(model, trainable_only=False) == 10


def test_measure_sampling_speed() -> None:
    device = torch.device("cpu")

    speed = measure_sampling_speed(
        sample_fn=lambda: torch.zeros(4, 3, 64, 64),
        num_samples=4,
        device=device,
        warmup_steps=0,
        measured_steps=2,
    )

    assert speed.num_samples == 8
    assert speed.total_seconds > 0.0
    assert speed.samples_per_second > 0.0
