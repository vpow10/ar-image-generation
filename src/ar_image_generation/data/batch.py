from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class ImageBatch:
    images: torch.Tensor
    labels: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None
