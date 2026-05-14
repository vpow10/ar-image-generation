from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from ar_image_generation.approaches.custom.primitives import (
    GaussianPrimitiveConfig,
    GaussianPrimitives,
    constrain_primitives,
    primitive_parameter_dim,
)
from ar_image_generation.approaches.custom.renderer import (
    GaussianRenderer2D,
    GaussianRendererConfig,
)


@dataclass(slots=True)
class GaussianTokenizerConfig:
    image_size: int = 64
    image_channels: int = 3
    num_primitives: int = 256
    primitive_feature_dim: int = 64
    hidden_channels: int = 192
    min_scale: float = 0.006
    max_scale: float = 0.12
    max_position_offset: float = 0.025
    renderer_chunk_size: int = 64


@dataclass(slots=True)
class GaussianTokenizerOutput:
    reconstructions: torch.Tensor
    raw_primitives: torch.Tensor
    primitives: GaussianPrimitives
    feature_map: torch.Tensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    l1_reconstruction_loss: torch.Tensor
    mse_reconstruction_loss: torch.Tensor
    opacity_loss: torch.Tensor
    scale_loss: torch.Tensor
    coverage_loss: torch.Tensor


def make_anchor_grid(num_primitives: int) -> torch.Tensor:
    if num_primitives <= 0:
        raise ValueError(f"num_primitives must be positive, got {num_primitives}")

    rows = int(math.sqrt(num_primitives))
    cols = math.ceil(num_primitives / rows)

    while rows * cols < num_primitives:
        rows += 1

    y = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
    x = (torch.arange(cols, dtype=torch.float32) + 0.5) / cols

    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    anchors = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

    return anchors[:num_primitives]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


class GaussianTokenizerEncoder(nn.Module):
    def __init__(self, cfg: GaussianTokenizerConfig) -> None:
        super().__init__()

        self.cfg = cfg
        primitive_dim = primitive_parameter_dim(cfg.primitive_feature_dim)

        self.backbone = nn.Sequential(
            ConvBlock(cfg.image_channels, cfg.hidden_channels, stride=2),
            ConvBlock(cfg.hidden_channels, cfg.hidden_channels, stride=2),
            ConvBlock(cfg.hidden_channels, cfg.hidden_channels, stride=1),
            ResidualBlock(cfg.hidden_channels),
            ResidualBlock(cfg.hidden_channels),
        )

        anchors = make_anchor_grid(cfg.num_primitives)
        self.register_buffer("anchors", anchors, persistent=False)

        self.output_layer = nn.Linear(cfg.hidden_channels, primitive_dim)

        self.proj = nn.Sequential(
            nn.Linear(cfg.hidden_channels * 2 + 2, cfg.hidden_channels),
            nn.SiLU(),
            nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
            nn.SiLU(),
            self.output_layer,
        )

        self._init_output_layer()

    def _init_output_layer(self) -> None:
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

        target_scale = 0.035
        scale_unit = (target_scale - self.cfg.min_scale) / (self.cfg.max_scale - self.cfg.min_scale)
        scale_unit = min(max(scale_unit, 1e-4), 1.0 - 1e-4)
        scale_bias = math.log(scale_unit / (1.0 - scale_unit))

        target_opacity = 0.75
        opacity_bias = math.log(target_opacity / (1.0 - target_opacity))

        with torch.no_grad():
            self.output_layer.bias[2:4].fill_(scale_bias)
            self.output_layer.bias[5:6].fill_(opacity_bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)

        batch_size = images.shape[0]
        anchors = self.anchors.to(device=images.device, dtype=images.dtype)

        sample_grid = anchors * 2.0 - 1.0
        sample_grid = sample_grid[None, :, None, :].expand(batch_size, -1, -1, -1)

        sampled_features = F.grid_sample(
            features,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        sampled_features = sampled_features.squeeze(-1).transpose(1, 2)

        global_features = features.mean(dim=(2, 3))
        global_features = global_features[:, None, :].expand(-1, self.cfg.num_primitives, -1)

        anchor_features = anchors[None, :, :].expand(batch_size, -1, -1)

        primitive_features = torch.cat(
            [
                sampled_features,
                global_features,
                anchor_features,
            ],
            dim=-1,
        )

        return self.proj(primitive_features)


class GaussianTokenizerDecoder(nn.Module):
    def __init__(self, cfg: GaussianTokenizerConfig) -> None:
        super().__init__()

        input_channels = cfg.primitive_feature_dim + 1

        self.net = nn.Sequential(
            ConvBlock(input_channels, cfg.hidden_channels),
            ResidualBlock(cfg.hidden_channels),
            ResidualBlock(cfg.hidden_channels),
            ConvBlock(cfg.hidden_channels, cfg.hidden_channels),
            nn.Conv2d(cfg.hidden_channels, cfg.image_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.net(feature_map)


class GaussianSplatTokenizer(nn.Module):
    def __init__(
        self,
        cfg: GaussianTokenizerConfig,
        *,
        reconstruction_l1_weight: float = 1.0,
        reconstruction_mse_weight: float = 0.25,
        opacity_weight: float = 0.01,
        scale_weight: float = 0.0001,
        coverage_weight: float = 0.02,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.reconstruction_l1_weight = reconstruction_l1_weight
        self.reconstruction_mse_weight = reconstruction_mse_weight
        self.opacity_weight = opacity_weight
        self.scale_weight = scale_weight
        self.coverage_weight = coverage_weight

        self.primitive_cfg = GaussianPrimitiveConfig(
            feature_dim=cfg.primitive_feature_dim,
            min_scale=cfg.min_scale,
            max_scale=cfg.max_scale,
            max_position_offset=cfg.max_position_offset,
        )

        self.encoder = GaussianTokenizerEncoder(cfg)

        self.renderer = GaussianRenderer2D(
            GaussianRendererConfig(
                image_size=cfg.image_size,
                feature_dim=cfg.primitive_feature_dim,
                chunk_size=cfg.renderer_chunk_size,
            )
        )

        self.decoder = GaussianTokenizerDecoder(cfg)

    def encode(self, images: torch.Tensor) -> GaussianPrimitives:
        raw_primitives = self.encoder(images)
        return constrain_primitives(
            raw_primitives,
            self.primitive_cfg,
            anchors=self.encoder.anchors,
        )

    def decode(self, primitives: GaussianPrimitives) -> torch.Tensor:
        feature_map = self.renderer(primitives)
        return self.decoder(feature_map)

    def forward(self, images: torch.Tensor) -> GaussianTokenizerOutput:
        raw_primitives = self.encoder(images)
        primitives = constrain_primitives(
            raw_primitives,
            self.primitive_cfg,
            anchors=self.encoder.anchors,
        )

        feature_map = self.renderer(primitives)
        reconstructions = self.decoder(feature_map)

        l1_reconstruction_loss = F.l1_loss(reconstructions, images)
        mse_reconstruction_loss = F.mse_loss(reconstructions, images)

        reconstruction_loss = (
            self.reconstruction_l1_weight * l1_reconstruction_loss
            + self.reconstruction_mse_weight * mse_reconstruction_loss
        )

        mean_opacity = primitives.opacity.mean()
        opacity_loss = F.relu(0.55 - mean_opacity).pow(2)

        scale_loss = primitives.scale.square().mean()

        coverage = feature_map[:, -1:].mean()
        coverage_loss = F.relu(0.65 - coverage).pow(2)

        loss = (
            reconstruction_loss
            + self.opacity_weight * opacity_loss
            + self.scale_weight * scale_loss
            + self.coverage_weight * coverage_loss
        )

        return GaussianTokenizerOutput(
            reconstructions=reconstructions,
            raw_primitives=raw_primitives,
            primitives=primitives,
            feature_map=feature_map,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            l1_reconstruction_loss=l1_reconstruction_loss,
            mse_reconstruction_loss=mse_reconstruction_loss,
            opacity_loss=opacity_loss,
            scale_loss=scale_loss,
            coverage_loss=coverage_loss,
        )
