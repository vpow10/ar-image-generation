from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.tokenizers.base import ImageTokenizer


@dataclass(slots=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    num_samples: int = 64


class AutoregressiveApproach(nn.Module, ABC):
    name: str

    @abstractmethod
    def training_step(
        self,
        batch: ImageBatch,
        tokenizer: ImageTokenizer,
    ) -> dict[str, torch.Tensor]:
        """Return a dict containing at least a scalar `loss` tensor."""

    @torch.no_grad()
    @abstractmethod
    def generate(
        self,
        tokenizer: ImageTokenizer,
        batch_size: int,
        labels: torch.Tensor | None,
        device: torch.device,
        sampling_cfg: SamplingConfig,
    ) -> torch.Tensor:
        """Generate images [B, C, H, W]."""
