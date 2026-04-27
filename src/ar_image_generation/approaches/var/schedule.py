from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VARSchedule:
    scales: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.scales) == 0:
            raise ValueError("VAR schedule must contain at least one scale.")

        previous = 0
        for scale in self.scales:
            if scale <= 0:
                raise ValueError(f"Scale must be positive, got {scale}")

            if scale <= previous:
                raise ValueError(f"Scales must be strictly increasing, got {self.scales}")

            previous = scale

    @property
    def final_scale(self) -> int:
        return self.scales[-1]

    @property
    def num_tokens(self) -> int:
        return sum(scale * scale for scale in self.scales)

    @property
    def num_scales(self) -> int:
        return len(self.scales)

    def scale_slices(self, *, include_bos: bool = False) -> list[slice]:
        offset = 1 if include_bos else 0
        slices: list[slice] = []

        for scale in self.scales:
            length = scale * scale
            slices.append(slice(offset, offset + length))
            offset += length

        return slices
