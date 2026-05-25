from dataclasses import dataclass

import torch


@dataclass(slots=True)
class GaussianPrimitiveConfig:
    feature_dim: int = 32
    min_scale: float = 0.015
    max_scale: float = 0.35
    max_position_offset: float | None = None


@dataclass(slots=True)
class GaussianPrimitives:
    position: torch.Tensor
    scale: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    feature: torch.Tensor


def primitive_parameter_dim(feature_dim: int) -> int:
    return 6 + feature_dim


def constrain_primitives(
    raw_primitives: torch.Tensor,
    cfg: GaussianPrimitiveConfig,
    anchors: torch.Tensor | None = None,
) -> GaussianPrimitives:
    if raw_primitives.ndim != 3:
        raise ValueError(
            f"Expected raw primitives [B, N, D], got shape {tuple(raw_primitives.shape)}"
        )

    expected_dim = primitive_parameter_dim(cfg.feature_dim)
    if raw_primitives.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected primitive dim {expected_dim}, got {raw_primitives.shape[-1]}"
        )

    raw_position = raw_primitives[..., 0:2]
    raw_scale = raw_primitives[..., 2:4]
    raw_rotation = raw_primitives[..., 4:5]
    raw_opacity = raw_primitives[..., 5:6]
    raw_feature = raw_primitives[..., 6:]

    if anchors is None:
        position = torch.sigmoid(raw_position)
    else:
        if cfg.max_position_offset is None:
            raise ValueError("max_position_offset must be set when anchors are used.")

        if anchors.ndim == 2:
            anchors = anchors[None, :, :]

        if anchors.shape[-1] != 2:
            raise ValueError(f"Expected anchors [..., 2], got {tuple(anchors.shape)}")

        offset = cfg.max_position_offset * torch.tanh(raw_position)
        position = (
            anchors.to(device=raw_primitives.device, dtype=raw_primitives.dtype)
            + offset
        )
        position = position.clamp(0.0, 1.0)

    scale_unit = torch.sigmoid(raw_scale)
    scale = cfg.min_scale + (cfg.max_scale - cfg.min_scale) * scale_unit

    rotation = torch.pi * torch.tanh(raw_rotation)
    opacity = torch.sigmoid(raw_opacity)
    feature = torch.tanh(raw_feature)

    return GaussianPrimitives(
        position=position,
        scale=scale,
        rotation=rotation,
        opacity=opacity,
        feature=feature,
    )


def flatten_primitives(primitives: GaussianPrimitives) -> torch.Tensor:
    return torch.cat(
        [
            primitives.position,
            primitives.scale,
            primitives.rotation,
            primitives.opacity,
            primitives.feature,
        ],
        dim=-1,
    )


def primitives_from_flattened(
    flattened: torch.Tensor,
    cfg: GaussianPrimitiveConfig,
    anchors: torch.Tensor | None = None,
) -> GaussianPrimitives:
    if flattened.ndim != 3:
        raise ValueError(
            f"Expected flattened primitives [B, N, D], got {tuple(flattened.shape)}"
        )

    expected_dim = primitive_parameter_dim(cfg.feature_dim)
    if flattened.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected primitive dim {expected_dim}, got {flattened.shape[-1]}"
        )

    position = flattened[..., 0:2]
    scale = flattened[..., 2:4]
    rotation = flattened[..., 4:5]
    opacity = flattened[..., 5:6]
    feature = flattened[..., 6:]

    if anchors is not None:
        if cfg.max_position_offset is None:
            raise ValueError("max_position_offset must be set when anchors are used.")

        if anchors.ndim == 2:
            anchors = anchors[None, :, :]

        anchors = anchors.to(device=flattened.device, dtype=flattened.dtype)
        min_position = anchors - cfg.max_position_offset
        max_position = anchors + cfg.max_position_offset
        position = torch.maximum(torch.minimum(position, max_position), min_position)

    position = position.clamp(0.0, 1.0)
    scale = scale.clamp(cfg.min_scale, cfg.max_scale)
    rotation = rotation.clamp(-torch.pi, torch.pi)
    opacity = opacity.clamp(0.0, 1.0)
    feature = feature.clamp(-1.0, 1.0)

    return GaussianPrimitives(
        position=position,
        scale=scale,
        rotation=rotation,
        opacity=opacity,
        feature=feature,
    )
