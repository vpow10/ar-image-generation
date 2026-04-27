from pathlib import Path

import torch
from torchvision.utils import save_image


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to [0, 1]."""

    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    *,
    nrow: int = 8,
    normalized: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images_to_save = denormalize_images(images) if normalized else images.clamp(0.0, 1.0)

    save_image(
        images_to_save,
        fp=str(path),
        nrow=nrow,
    )
