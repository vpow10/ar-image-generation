import math
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
    l1_reconstruction_loss: torch.Tensor
    mse_reconstruction_loss: torch.Tensor
    quantization_loss: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor


def _num_downsample_layers(downsample_factor: int) -> int:
    if downsample_factor < 2:
        raise ValueError(f"downsample_factor must be >= 2, got {downsample_factor}")

    if downsample_factor & (downsample_factor - 1) != 0:
        raise ValueError(f"downsample_factor must be a power of two, got {downsample_factor}")

    return int(math.log2(downsample_factor))


def _make_group_norm(channels: int) -> nn.GroupNorm:
    for num_groups in (8, 4, 2, 1):
        if channels % num_groups == 0:
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    raise ValueError(f"Could not create GroupNorm for channels={channels}")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            _make_group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _make_group_norm(channels),
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
        num_downsample_layers: int,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_channels = in_channels

        for _ in range(num_downsample_layers):
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        hidden_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.SiLU(),
                ]
            )
            current_channels = hidden_channels

        layers.extend(
            [
                ResidualBlock(hidden_channels),
                ResidualBlock(hidden_channels),
                _make_group_norm(hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        num_upsample_layers: int,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            _make_group_norm(hidden_channels),
            nn.SiLU(),
        ]

        for layer_index in range(num_upsample_layers):
            is_last_layer = layer_index == num_upsample_layers - 1
            next_channels = out_channels if is_last_layer else hidden_channels

            layers.append(
                nn.ConvTranspose2d(
                    hidden_channels,
                    next_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )

            if not is_last_layer:
                layers.append(nn.SiLU())

        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

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

        batch_size, embedding_dim, height, width = z_e.shape

        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z = z.view(-1, embedding_dim)

        embedding_weight = self.embedding.weight

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + embedding_weight.pow(2).sum(dim=1)
            - 2.0 * flat_z @ embedding_weight.t()
        )

        flat_indices = torch.argmin(distances, dim=1)
        indices = flat_indices.view(batch_size, height, width)

        z_q = self.embedding(flat_indices)
        z_q = z_q.view(batch_size, height, width, embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        quantization_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach()) * self.commitment_cost

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
        vocab_size: int = 1024,
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        downsample_factor: int = 4,
        commitment_cost: float = 0.25,
        reconstruction_l1_weight: float = 1.0,
        reconstruction_mse_weight: float = 0.25,
    ) -> None:
        super().__init__()

        if image_size % downsample_factor != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by "
                f"downsample_factor={downsample_factor}"
            )

        num_downsample_layers = _num_downsample_layers(downsample_factor)

        self.vocab_size = vocab_size
        self.latent_shape = (
            image_size // downsample_factor,
            image_size // downsample_factor,
        )

        self.reconstruction_l1_weight = reconstruction_l1_weight
        self.reconstruction_mse_weight = reconstruction_mse_weight

        self.encoder = Encoder(
            in_channels=image_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            num_downsample_layers=num_downsample_layers,
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
            num_upsample_layers=num_downsample_layers,
        )

    def forward(self, images: torch.Tensor) -> VQVAEOutput:
        z_e = self.encoder(images)
        z_q, indices, quantization_loss, commitment_loss, perplexity = self.quantizer(z_e)
        reconstructions = self.decoder(z_q)

        l1_reconstruction_loss = F.l1_loss(reconstructions, images)
        mse_reconstruction_loss = F.mse_loss(reconstructions, images)

        reconstruction_loss = (
            self.reconstruction_l1_weight * l1_reconstruction_loss
            + self.reconstruction_mse_weight * mse_reconstruction_loss
        )

        loss = reconstruction_loss + quantization_loss + commitment_loss

        return VQVAEOutput(
            reconstructions=reconstructions,
            indices=indices,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            l1_reconstruction_loss=l1_reconstruction_loss,
            mse_reconstruction_loss=mse_reconstruction_loss,
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
