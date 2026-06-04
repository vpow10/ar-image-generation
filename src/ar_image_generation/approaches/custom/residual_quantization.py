from dataclasses import dataclass

import torch

from ar_image_generation.approaches.custom.quantization import (
    FeatureCodebook,
    fit_kmeans,
)


@dataclass(slots=True)
class ResidualFeatureCodebook:
    codebooks: list[FeatureCodebook]

    @property
    def num_quantizers(self) -> int:
        return len(self.codebooks)

    @property
    def num_codes(self) -> int:
        if not self.codebooks:
            raise ValueError("Residual codebook has no codebooks.")
        return self.codebooks[0].num_codes

    @property
    def feature_dim(self) -> int:
        if not self.codebooks:
            raise ValueError("Residual codebook has no codebooks.")
        return self.codebooks[0].feature_dim

    def assign(
        self,
        features: torch.Tensor,
        *,
        chunk_size: int = 65536,
    ) -> torch.LongTensor:
        residual = features
        codes: list[torch.Tensor] = []

        for codebook in self.codebooks:
            code = codebook.assign(residual, chunk_size=chunk_size)
            quantized = codebook.lookup(code).to(
                device=features.device, dtype=features.dtype
            )

            codes.append(code)
            residual = residual - quantized

        return torch.stack(codes, dim=-1)

    def lookup(self, codes: torch.LongTensor) -> torch.Tensor:
        if codes.shape[-1] != self.num_quantizers:
            raise ValueError(
                f"Expected {self.num_quantizers} residual code dimensions, got {codes.shape[-1]}"
            )

        reconstructed = None

        for index, codebook in enumerate(self.codebooks):
            partial = codebook.lookup(codes[..., index])

            if reconstructed is None:
                reconstructed = partial
            else:
                reconstructed = (
                    reconstructed.to(device=partial.device, dtype=partial.dtype)
                    + partial
                )

        if reconstructed is None:
            raise ValueError("Residual codebook has no codebooks.")

        return reconstructed

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        centers = torch.stack(
            [codebook.centers.detach().cpu() for codebook in self.codebooks],
            dim=0,
        )

        return {
            "centers": centers,
            "num_quantizers": self.num_quantizers,
            "num_codes": self.num_codes,
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ResidualFeatureCodebook":
        centers = state["centers"]

        if centers.ndim != 3:
            raise ValueError(
                f"Expected centers [Q, K, D], got shape {tuple(centers.shape)}"
            )

        codebooks = [
            FeatureCodebook(centers=centers[index]) for index in range(centers.shape[0])
        ]

        return cls(codebooks=codebooks)


def fit_residual_kmeans(
    samples: torch.Tensor,
    *,
    num_quantizers: int,
    num_codes: int,
    num_iters: int = 40,
    chunk_size: int = 65536,
) -> ResidualFeatureCodebook:
    if samples.ndim != 2:
        raise ValueError(f"Expected samples [N, D], got {tuple(samples.shape)}")

    if num_quantizers <= 0:
        raise ValueError(f"num_quantizers must be positive, got {num_quantizers}")

    residual = samples
    codebooks: list[FeatureCodebook] = []

    for quantizer_index in range(num_quantizers):
        print(f"Fitting residual codebook {quantizer_index + 1}/{num_quantizers}")

        codebook = fit_kmeans(
            residual,
            num_codes=num_codes,
            num_iters=num_iters,
            chunk_size=chunk_size,
        )

        codebooks.append(codebook)

        codes = codebook.assign(residual, chunk_size=chunk_size)
        quantized = codebook.lookup(codes).to(
            device=samples.device, dtype=samples.dtype
        )
        residual = residual - quantized

        residual_mse = residual.square().mean().item()
        print(f"Residual MSE after codebook {quantizer_index + 1}: {residual_mse:.6f}")

    return ResidualFeatureCodebook(codebooks=codebooks)
