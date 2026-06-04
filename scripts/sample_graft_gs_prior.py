import argparse
from pathlib import Path
from typing import Any

import torch

from ar_image_generation.approaches.custom.model import (
    GaussianPrimitiveAR,
    GaussianPrimitiveARConfig,
)
from ar_image_generation.approaches.custom.normalization import PrimitiveNormalizer
from ar_image_generation.approaches.custom.primitives import primitives_from_flattened
from ar_image_generation.config import load_yaml
from ar_image_generation.engine.checkpointing import load_model_checkpoint
from ar_image_generation.tokenizers.graft_gs import (
    GaussianSplatTokenizer,
    GaussianTokenizerConfig,
)
from ar_image_generation.utils.device import get_device
from ar_image_generation.utils.image_grid import save_image_grid
from ar_image_generation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from GRAFT-GS autoregressive prior."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to config YAML."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Prior checkpoint path."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output image grid path."
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Override number of samples."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Override sampling temperature."
    )
    parser.add_argument(
        "--label", type=int, default=None, help="Optional class label for all samples."
    )
    parser.add_argument(
        "--geometry-noise-scale",
        type=float,
        default=None,
        help="Noise scale for geometry primitive dimensions.",
    )
    parser.add_argument(
        "--feature-noise-scale",
        type=float,
        default=None,
        help="Noise scale for feature primitive dimensions.",
    )
    return parser.parse_args()


def build_graft_tokenizer(config: dict[str, Any]) -> GaussianSplatTokenizer:
    tokenizer_cfg = config["tokenizer"]

    model_cfg = GaussianTokenizerConfig(
        image_size=tokenizer_cfg["image_size"],
        image_channels=tokenizer_cfg["image_channels"],
        num_primitives=tokenizer_cfg["num_primitives"],
        primitive_feature_dim=tokenizer_cfg["primitive_feature_dim"],
        hidden_channels=tokenizer_cfg["hidden_channels"],
        min_scale=tokenizer_cfg["min_scale"],
        max_scale=tokenizer_cfg["max_scale"],
        max_position_offset=tokenizer_cfg["max_position_offset"],
        renderer_chunk_size=tokenizer_cfg["renderer_chunk_size"],
    )

    return GaussianSplatTokenizer(model_cfg)


def build_prior(config: dict[str, Any]) -> GaussianPrimitiveAR:
    prior_cfg = config["prior"]

    return GaussianPrimitiveAR(
        GaussianPrimitiveARConfig(
            num_primitives=prior_cfg["num_primitives"],
            primitive_dim=prior_cfg["primitive_dim"],
            dim=prior_cfg["dim"],
            depth=prior_cfg["depth"],
            num_heads=prior_cfg["num_heads"],
            mlp_ratio=prior_cfg["mlp_ratio"],
            dropout=prior_cfg["dropout"],
            class_conditional=prior_cfg["class_conditional"],
            num_classes=prior_cfg["num_classes"],
            log_std_min=prior_cfg["log_std_min"],
            log_std_max=prior_cfg["log_std_max"],
            mean_loss_weight=prior_cfg["mean_loss_weight"],
        )
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed_everything(config["seed"])

    device = get_device(config["device"])

    tokenizer = build_graft_tokenizer(config).to(device)
    load_model_checkpoint(
        path=config["tokenizer"]["checkpoint_path"],
        model=tokenizer,
        map_location=device,
    )
    tokenizer.eval()

    prior = build_prior(config).to(device)
    checkpoint = load_model_checkpoint(
        path=args.checkpoint,
        model=prior,
        map_location=device,
    )
    prior.eval()

    normalizer = PrimitiveNormalizer.from_state_dict(checkpoint["extra"]["normalizer"])

    num_samples = args.num_samples or config["sampling"]["num_samples"]
    temperature = args.temperature or config["sampling"]["temperature"]

    labels = None
    if args.label is not None:
        labels = torch.full(
            (num_samples,),
            args.label,
            dtype=torch.long,
            device=device,
        )

    geometry_noise_scale = (
        args.geometry_noise_scale
        if args.geometry_noise_scale is not None
        else config["sampling"].get("geometry_noise_scale", 0.7)
    )
    feature_noise_scale = (
        args.feature_noise_scale
        if args.feature_noise_scale is not None
        else config["sampling"].get("feature_noise_scale", 0.05)
    )

    normalized_primitives = prior.generate(
        batch_size=num_samples,
        labels=labels,
        device=device,
        temperature=temperature,
        geometry_noise_scale=geometry_noise_scale,
        feature_noise_scale=feature_noise_scale,
    )

    normalized_primitives = normalized_primitives.clamp(-3.0, 3.0)
    flattened_primitives = normalizer.denormalize(normalized_primitives)

    primitives = primitives_from_flattened(
        flattened_primitives,
        tokenizer.primitive_cfg,
        anchors=tokenizer.encoder.anchors,
    )

    images = tokenizer.decode(primitives)

    save_image_grid(
        images.detach().cpu(),
        args.output,
        nrow=8,
        normalized=True,
    )

    print("GRAFT-GS sampling complete")
    print("--------------------------")
    print(f"checkpoint:  {args.checkpoint}")
    print(f"output:      {args.output}")
    print(f"samples:     {num_samples}")
    print(f"temperature: {temperature}")
    print(f"label:       {args.label}")
    print(f"geometry noise: {geometry_noise_scale}")
    print(f"feature noise:  {feature_noise_scale}")


if __name__ == "__main__":
    main()
