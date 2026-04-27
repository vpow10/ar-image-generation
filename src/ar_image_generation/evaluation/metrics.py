import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class SamplingSpeed:
    num_samples: int
    total_seconds: float
    samples_per_second: float


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    parameters = model.parameters()

    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)

    return sum(parameter.numel() for parameter in parameters)


@torch.no_grad()
def measure_sampling_speed(
    *,
    sample_fn: Callable[[], torch.Tensor],
    num_samples: int,
    device: torch.device,
    warmup_steps: int = 1,
    measured_steps: int = 3,
) -> SamplingSpeed:
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

    if measured_steps <= 0:
        raise ValueError(f"measured_steps must be positive, got {measured_steps}")

    for _ in range(warmup_steps):
        _ = sample_fn()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()

    for _ in range(measured_steps):
        _ = sample_fn()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    total_samples = num_samples * measured_steps

    return SamplingSpeed(
        num_samples=total_samples,
        total_seconds=elapsed,
        samples_per_second=total_samples / elapsed,
    )
