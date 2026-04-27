from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ar_image_generation.tokenizers.base import ImageTokenizer


@dataclass(slots=True)
class VQVAEOutput:
    reconstructions: torch.Tensor
    indices: torch.LongTensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    quantization_loss: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        # 64 -> 32 -> 16 -> 8
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        # 8 -> 16 -> 32 -> 64
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.SiLU(),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        commitment_cost: float,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

    def forward(
        self,
        z_e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: continuous encoder latents [B, D, H, W]

        Returns:
            z_q_st: straight-through quantized latents [B, D, H, W]
            indices: codebook indices [B, H, W]
            quantization_loss: codebook loss
            commitment_loss: encoder commitment loss
            perplexity: codebook usage perplexity
        """

        b, d, h, w = z_e.shape

        # [B, D, H, W] -> [B, H, W, D] -> [B*H*W, D]
        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z = z.view(-1, d)

        embedding_weight = self.embedding.weight

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + embedding_weight.pow(2).sum(dim=1)
            - 2.0 * flat_z @ embedding_weight.t()
        )

        flat_indices = torch.argmin(distances, dim=1)
        indices = flat_indices.view(b, h, w)

        z_q = self.embedding(flat_indices)
        z_q = z_q.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        quantization_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach()) * self.commitment_cost

        # Straight-through estimator.
        z_q_st = z_e + (z_q - z_e).detach()

        one_hot = F.one_hot(flat_indices, num_classes=self.vocab_size).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, indices, quantization_loss, commitment_loss, perplexity

    def quantize_from_indices(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            indices: [B, H, W]

        Returns:
            quantized embeddings [B, D, H, W]
        """

        z_q = self.embedding(indices)
        return z_q.permute(0, 3, 1, 2).contiguous()


class VQVAE(ImageTokenizer):
    def __init__(
        self,
        *,
        image_channels: int = 3,
        image_size: int = 64,
        vocab_size: int = 512,
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        downsample_factor: int = 8,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()

        if image_size % downsample_factor != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by downsample_factor={downsample_factor}"
            )

        self.vocab_size = vocab_size
        self.latent_shape = (image_size // downsample_factor, image_size // downsample_factor)

        self.encoder = Encoder(
            in_channels=image_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
        )
        self.quantizer = VectorQuantizer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = Decoder(
            out_channels=image_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
        )

    def forward(self, images: torch.Tensor) -> VQVAEOutput:
        z_e = self.encoder(images)
        z_q, indices, quantization_loss, commitment_loss, perplexity = self.quantizer(z_e)
        reconstructions = self.decoder(z_q)

        reconstruction_loss = F.mse_loss(reconstructions, images)
        loss = reconstruction_loss + quantization_loss + commitment_loss

        return VQVAEOutput(
            reconstructions=reconstructions,
            indices=indices,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            quantization_loss=quantization_loss,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.LongTensor:
        z_e = self.encoder(images)
        _, indices, _, _, _ = self.quantizer(z_e)
        return indices

    @torch.no_grad()
    def decode(self, tokens: torch.LongTensor) -> torch.Tensor:
        z_q = self.quantizer.quantize_from_indices(tokens)
        return self.decoder(z_q)
