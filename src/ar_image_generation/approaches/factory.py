from ar_image_generation.approaches.base import AutoregressiveApproach
from ar_image_generation.approaches.registry import build_approach
from ar_image_generation.config import ApproachConfig


def build_approach_from_config(
    cfg: ApproachConfig,
    *,
    vocab_size: int,
    latent_shape: tuple[int, int],
) -> AutoregressiveApproach:
    cfg_dict = cfg.model_dump()
    name = cfg_dict.pop("name")

    return build_approach(
        name,
        vocab_size=vocab_size,
        latent_shape=latent_shape,
        **cfg_dict,
    )
