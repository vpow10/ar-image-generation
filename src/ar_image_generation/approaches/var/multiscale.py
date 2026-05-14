import torch
import torch.nn.functional as F

from ar_image_generation.approaches.var.schedule import VARSchedule
from ar_image_generation.tokenizers.base import ImageTokenizer


def validate_token_grid(tokens: torch.Tensor) -> None:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens [B, H, W], got shape {tuple(tokens.shape)}")

    height, width = tokens.shape[-2:]

    if height != width:
        raise ValueError(f"Expected square token grid, got {height}x{width}")


def build_image_pyramid_token_grids(
    images: torch.Tensor,
    tokenizer: ImageTokenizer,
    schedule: VARSchedule,
) -> list[torch.LongTensor]:
    """
    Build meaningful VAR multiscale targets from an image pyramid.

    For a 64x64 image and final latent scale 16:

        scale 1  -> resize image to 4x4   -> tokenizer.encode -> 1x1 tokens
        scale 2  -> resize image to 8x8   -> tokenizer.encode -> 2x2 tokens
        scale 4  -> resize image to 16x16 -> tokenizer.encode -> 4x4 tokens
        scale 8  -> resize image to 32x32 -> tokenizer.encode -> 8x8 tokens
        scale 16 -> original 64x64        -> tokenizer.encode -> 16x16 tokens

    This is much more meaningful than downsampling categorical token IDs.
    """

    if images.ndim != 4:
        raise ValueError(f"Expected images [B, C, H, W], got shape {tuple(images.shape)}")

    image_height, image_width = images.shape[-2:]

    if image_height != image_width:
        raise ValueError(f"Expected square images, got {image_height}x{image_width}")

    final_scale = schedule.final_scale

    if image_height % final_scale != 0:
        raise ValueError(
            f"Image size {image_height} must be divisible by final latent scale {final_scale}"
        )

    pixels_per_token = image_height // final_scale

    multiscale_tokens: list[torch.LongTensor] = []

    for scale in schedule.scales:
        target_image_size = scale * pixels_per_token

        if target_image_size == image_height:
            resized_images = images
        else:
            resized_images = F.interpolate(
                images,
                size=(target_image_size, target_image_size),
                mode="bilinear",
                align_corners=False,
            )

        tokens = tokenizer.encode(resized_images)

        expected_shape = (images.shape[0], scale, scale)
        if tuple(tokens.shape) != expected_shape:
            raise ValueError(
                f"Tokenizer returned shape {tuple(tokens.shape)} for scale {scale}, "
                f"expected {expected_shape}. "
                "Check tokenizer downsample factor and VAR scale schedule."
            )

        multiscale_tokens.append(tokens)

    return multiscale_tokens


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


def build_multiscale_sequence_from_images(
    images: torch.Tensor,
    tokenizer: ImageTokenizer,
    schedule: VARSchedule,
) -> torch.LongTensor:
    multiscale_tokens = build_image_pyramid_token_grids(images, tokenizer, schedule)
    return flatten_multiscale_token_grids(multiscale_tokens)


def build_multiscale_sequence_from_final_tokens(
    tokens: torch.LongTensor,
    schedule: VARSchedule,
) -> torch.LongTensor:
    """
    Test/debug fallback.
    """

    validate_token_grid(tokens)

    final_height = tokens.shape[-2]
    if final_height != schedule.final_scale:
        raise ValueError(
            f"Token grid has final scale {final_height}, "
            f"but schedule expects {schedule.final_scale}"
        )

    return tokens.reshape(tokens.shape[0], -1)


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
