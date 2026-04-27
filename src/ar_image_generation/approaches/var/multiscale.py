import torch
import torch.nn.functional as F

from ar_image_generation.approaches.var.schedule import VARSchedule


def validate_token_grid(tokens: torch.Tensor) -> None:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens [B, H, W], got shape {tuple(tokens.shape)}")

    height, width = tokens.shape[-2:]

    if height != width:
        raise ValueError(f"Expected square token grid, got {height}x{width}")


def downsample_token_grid(tokens: torch.LongTensor, size: int) -> torch.LongTensor:
    """
    Downsample a discrete token grid with nearest-neighbor sampling.

    This is a simple project-friendly approximation for VAR-style multiscale targets.
    Full VAR-style implementations may use richer scale-specific quantization.
    """

    validate_token_grid(tokens)

    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    tokens_float = tokens.float().unsqueeze(1)
    resized = F.interpolate(tokens_float, size=(size, size), mode="nearest")
    return resized.squeeze(1).long()


def build_multiscale_token_grids(
    tokens: torch.LongTensor,
    schedule: VARSchedule,
) -> list[torch.LongTensor]:
    """
    Args:
        tokens: full-resolution latent tokens [B, H, W]

    Returns:
        list of token grids:
            [B, 1, 1], [B, 2, 2], ..., [B, H, W]
    """

    validate_token_grid(tokens)

    final_height = tokens.shape[-2]
    if final_height != schedule.final_scale:
        raise ValueError(
            f"Token grid has final scale {final_height}, "
            f"but schedule expects {schedule.final_scale}"
        )

    return [downsample_token_grid(tokens, scale) for scale in schedule.scales]


def flatten_multiscale_token_grids(
    multiscale_tokens: list[torch.LongTensor],
) -> torch.LongTensor:
    """
    Args:
        multiscale_tokens: list of [B, S, S] token grids

    Returns:
        flattened tokens [B, sum(S*S)]
    """

    if len(multiscale_tokens) == 0:
        raise ValueError("Expected at least one scale.")

    batch_size = multiscale_tokens[0].shape[0]
    flattened: list[torch.Tensor] = []

    for tokens in multiscale_tokens:
        validate_token_grid(tokens)

        if tokens.shape[0] != batch_size:
            raise ValueError("All scales must have the same batch size.")

        flattened.append(tokens.reshape(batch_size, -1))

    return torch.cat(flattened, dim=1)


def build_multiscale_sequence(
    tokens: torch.LongTensor,
    schedule: VARSchedule,
) -> torch.LongTensor:
    multiscale_tokens = build_multiscale_token_grids(tokens, schedule)
    return flatten_multiscale_token_grids(multiscale_tokens)


def build_scale_ids(
    schedule: VARSchedule,
    *,
    include_bos: bool,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """
    Returns a 1D tensor assigning each sequence position to a scale index.

    BOS position receives -1 when include_bos=True.
    """

    ids: list[int] = []

    if include_bos:
        ids.append(-1)

    for scale_index, scale in enumerate(schedule.scales):
        ids.extend([scale_index] * (scale * scale))

    return torch.tensor(ids, dtype=torch.long, device=device)


def build_next_scale_attention_mask(
    schedule: VARSchedule,
    *,
    include_bos: bool = True,
    device: torch.device | None = None,
) -> torch.BoolTensor:
    """
    Build a VAR-style attention mask.

    Rule:
        - BOS attends only to BOS.
        - Tokens at scale k may attend to BOS and all tokens from previous scales.
        - Tokens may not attend to tokens from the same scale.
        - Tokens may not attend to future scales.

    This prevents direct leakage from target tokens at the same scale.
    """

    scale_ids = build_scale_ids(schedule, include_bos=include_bos, device=device)
    query_scale_ids = scale_ids[:, None]
    key_scale_ids = scale_ids[None, :]

    if include_bos:
        bos_id = -1
        is_query_bos = query_scale_ids == bos_id
        is_key_bos = key_scale_ids == bos_id

        can_attend = is_key_bos | (key_scale_ids < query_scale_ids)
        can_attend = can_attend & ~is_query_bos
        can_attend[0, 0] = True

        return can_attend

    return key_scale_ids < query_scale_ids
