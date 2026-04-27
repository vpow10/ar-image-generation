from ar_image_generation.config import TokenizerConfig
from ar_image_generation.tokenizers.vqvae import VQVAE


def build_tokenizer(
    cfg: TokenizerConfig,
    *,
    image_size: int,
    image_channels: int = 3,
) -> VQVAE:
    if cfg.name != "vqvae_small":
        raise ValueError(f"Unknown tokenizer: {cfg.name}")

    return VQVAE(
        image_channels=image_channels,
        image_size=image_size,
        vocab_size=cfg.vocab_size,
        embedding_dim=cfg.embedding_dim,
        hidden_channels=cfg.hidden_channels,
        downsample_factor=cfg.downsample_factor,
        commitment_cost=cfg.train.commitment_cost,
        reconstruction_l1_weight=cfg.train.reconstruction_l1_weight,
        reconstruction_mse_weight=cfg.train.reconstruction_mse_weight,
    )
