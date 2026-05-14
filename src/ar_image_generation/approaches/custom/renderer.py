from dataclasses import dataclass

import torch
from torch import nn

from ar_image_generation.approaches.custom.primitives import GaussianPrimitives


@dataclass(slots=True)
class GaussianRendererConfig:
    image_size: int = 64
    feature_dim: int = 32
    chunk_size: int = 64
    eps: float = 1e-6


class GaussianRenderer2D(nn.Module):
    def __init__(self, cfg: GaussianRendererConfig) -> None:
        super().__init__()

        self.cfg = cfg

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, cfg.image_size),
            torch.linspace(0.0, 1.0, cfg.image_size),
            indexing="ij",
        )

        grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, primitives: GaussianPrimitives) -> torch.Tensor:
        position = primitives.position
        scale = primitives.scale
        rotation = primitives.rotation
        opacity = primitives.opacity
        feature = primitives.feature

        if position.ndim != 3:
            raise ValueError(f"Expected position [B, N, 2], got {tuple(position.shape)}")

        batch_size, num_primitives, _ = position.shape
        feature_dim = feature.shape[-1]

        if feature_dim != self.cfg.feature_dim:
            raise ValueError(f"Expected feature dim {self.cfg.feature_dim}, got {feature_dim}")

        image_size = self.cfg.image_size
        device = position.device
        dtype = position.dtype

        grid = self.grid.to(device=device, dtype=dtype)
        grid = grid.view(1, 1, image_size, image_size, 2)

        feature_map = torch.zeros(
            batch_size,
            feature_dim,
            image_size,
            image_size,
            device=device,
            dtype=dtype,
        )
        opacity_map = torch.zeros(
            batch_size,
            1,
            image_size,
            image_size,
            device=device,
            dtype=dtype,
        )

        for start in range(0, num_primitives, self.cfg.chunk_size):
            end = min(start + self.cfg.chunk_size, num_primitives)

            chunk_position = position[:, start:end]
            chunk_scale = scale[:, start:end].clamp_min(self.cfg.eps)
            chunk_rotation = rotation[:, start:end]
            chunk_opacity = opacity[:, start:end]
            chunk_feature = feature[:, start:end]

            chunk_weight = self._render_weight(
                grid=grid,
                position=chunk_position,
                scale=chunk_scale,
                rotation=chunk_rotation,
                opacity=chunk_opacity,
            )

            feature_map = feature_map + torch.einsum("bnhw,bnc->bchw", chunk_weight, chunk_feature)
            opacity_map = opacity_map + chunk_weight.sum(dim=1, keepdim=True)

        normalized_feature_map = feature_map / opacity_map.clamp_min(self.cfg.eps)
        opacity_channel = opacity_map.clamp(0.0, 1.0)

        return torch.cat([normalized_feature_map, opacity_channel], dim=1)

    def _render_weight(
        self,
        *,
        grid: torch.Tensor,
        position: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        center = position[:, :, None, None, :]
        delta = grid - center

        dx = delta[..., 0]
        dy = delta[..., 1]

        angle = rotation[..., 0]
        cos_angle = torch.cos(angle)[:, :, None, None]
        sin_angle = torch.sin(angle)[:, :, None, None]

        rotated_x = cos_angle * dx + sin_angle * dy
        rotated_y = -sin_angle * dx + cos_angle * dy

        scale_x = scale[..., 0][:, :, None, None]
        scale_y = scale[..., 1][:, :, None, None]

        exponent = -0.5 * ((rotated_x / scale_x) ** 2 + (rotated_y / scale_y) ** 2)
        gaussian = torch.exp(exponent)

        return opacity[:, :, None, None, 0] * gaussian
