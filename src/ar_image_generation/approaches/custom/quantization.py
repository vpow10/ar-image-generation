from dataclasses import dataclass

import torch


@dataclass(slots=True)
class FeatureCodebook:
    centers: torch.Tensor

    @property
    def num_codes(self) -> int:
        return int(self.centers.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.centers.shape[1])

    def assign(
        self,
        features: torch.Tensor,
        *,
        chunk_size: int = 65536,
    ) -> torch.LongTensor:
        original_shape = features.shape[:-1]
        flat_features = features.reshape(-1, features.shape[-1])

        centers = self.centers.to(
            device=flat_features.device, dtype=flat_features.dtype
        )
        center_norm = centers.square().sum(dim=1)

        codes: list[torch.Tensor] = []

        for start in range(0, flat_features.shape[0], chunk_size):
            end = min(start + chunk_size, flat_features.shape[0])
            chunk = flat_features[start:end]

            distances = (
                chunk.square().sum(dim=1, keepdim=True)
                - 2.0 * chunk @ centers.t()
                + center_norm[None, :]
            )
            codes.append(distances.argmin(dim=1))

        return torch.cat(codes, dim=0).reshape(original_shape)

    def lookup(self, codes: torch.LongTensor) -> torch.Tensor:
        return self.centers.to(device=codes.device)[codes]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "centers": self.centers.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "FeatureCodebook":
        return cls(centers=state["centers"])


def fit_kmeans(
    samples: torch.Tensor,
    *,
    num_codes: int,
    num_iters: int = 40,
    chunk_size: int = 65536,
) -> FeatureCodebook:
    if samples.ndim != 2:
        raise ValueError(f"Expected samples [N, D], got {tuple(samples.shape)}")

    num_samples = samples.shape[0]
    if num_samples < num_codes:
        raise ValueError(
            f"Need at least {num_codes} samples to fit k-means, got {num_samples}"
        )

    device = samples.device
    dtype = samples.dtype

    permutation = torch.randperm(num_samples, device=device)
    centers = samples[permutation[:num_codes]].clone()

    for _ in range(num_iters):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(num_codes, device=device, dtype=dtype)

        center_norm = centers.square().sum(dim=1)

        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            chunk = samples[start:end]

            distances = (
                chunk.square().sum(dim=1, keepdim=True)
                - 2.0 * chunk @ centers.t()
                + center_norm[None, :]
            )
            assignments = distances.argmin(dim=1)

            sums.index_add_(0, assignments, chunk)
            counts.index_add_(
                0,
                assignments,
                torch.ones(assignments.shape[0], device=device, dtype=dtype),
            )

        empty = counts == 0
        if empty.any():
            refill_indices = torch.randperm(num_samples, device=device)[
                : int(empty.sum().item())
            ]
            centers[empty] = samples[refill_indices]
            counts[empty] = 1.0
            sums[empty] = centers[empty]

        centers = sums / counts[:, None]

    return FeatureCodebook(centers=centers.detach().cpu())
