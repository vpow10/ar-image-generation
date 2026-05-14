import torch

from ar_image_generation.approaches.custom.primitives import (
    GaussianPrimitiveConfig,
    constrain_primitives,
    primitive_parameter_dim,
)
from ar_image_generation.approaches.custom.renderer import (
    GaussianRenderer2D,
    GaussianRendererConfig,
)


def test_constrain_primitives_shapes() -> None:
    cfg = GaussianPrimitiveConfig(
        feature_dim=16,
        min_scale=0.01,
        max_scale=0.25,
    )

    raw = torch.randn(2, 8, primitive_parameter_dim(cfg.feature_dim))
    primitives = constrain_primitives(raw, cfg)

    assert primitives.position.shape == (2, 8, 2)
    assert primitives.scale.shape == (2, 8, 2)
    assert primitives.rotation.shape == (2, 8, 1)
    assert primitives.opacity.shape == (2, 8, 1)
    assert primitives.feature.shape == (2, 8, 16)

    assert primitives.position.min().item() >= 0.0
    assert primitives.position.max().item() <= 1.0

    assert primitives.scale.min().item() >= cfg.min_scale
    assert primitives.scale.max().item() <= cfg.max_scale

    assert primitives.opacity.min().item() >= 0.0
    assert primitives.opacity.max().item() <= 1.0


def test_gaussian_renderer_shapes() -> None:
    primitive_cfg = GaussianPrimitiveConfig(
        feature_dim=16,
        min_scale=0.01,
        max_scale=0.25,
    )
    renderer_cfg = GaussianRendererConfig(
        image_size=32,
        feature_dim=16,
        chunk_size=4,
    )

    raw = torch.randn(2, 8, primitive_parameter_dim(primitive_cfg.feature_dim))
    primitives = constrain_primitives(raw, primitive_cfg)

    renderer = GaussianRenderer2D(renderer_cfg)
    feature_map = renderer(primitives)

    assert feature_map.shape == (2, 17, 32, 32)
    assert torch.isfinite(feature_map).all()


def test_gaussian_renderer_is_differentiable() -> None:
    primitive_cfg = GaussianPrimitiveConfig(
        feature_dim=8,
        min_scale=0.02,
        max_scale=0.25,
    )
    renderer_cfg = GaussianRendererConfig(
        image_size=16,
        feature_dim=8,
        chunk_size=4,
    )

    raw = torch.randn(
        2,
        6,
        primitive_parameter_dim(primitive_cfg.feature_dim),
        requires_grad=True,
    )

    primitives = constrain_primitives(raw, primitive_cfg)

    renderer = GaussianRenderer2D(renderer_cfg)
    feature_map = renderer(primitives)

    loss = feature_map.mean()
    loss.backward()

    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
