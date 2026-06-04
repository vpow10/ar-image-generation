from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PrimitiveNormalizer:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    def normalize(self, primitives: torch.Tensor) -> torch.Tensor:
        return (primitives - self.mean.to(primitives.device)) / self.std.to(primitives.device)

    def denormalize(self, primitives: torch.Tensor) -> torch.Tensor:
        return primitives * self.std.to(primitives.device) + self.mean.to(primitives.device)

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "PrimitiveNormalizer":
        return cls(
            mean=state["mean"],
            std=state["std"],
            eps=float(state.get("eps", 1e-6)),
        )
