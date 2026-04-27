from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_model_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, path)


def load_model_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
