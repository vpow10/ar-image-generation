from abc import ABC, abstractmethod

import torch
from torch import nn


class ImageTokenizer(nn.Module, ABC):
    vocab_size: int
    latent_shape: tuple[int, int]

    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.LongTensor:
        """Encode images [B, C, H, W] into token ids [B, H_latent, W_latent]."""

    @abstractmethod
    def decode(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Decode token ids [B, H_latent, W_latent] into images [B, C, H, W]."""
