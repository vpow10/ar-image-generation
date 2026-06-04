import argparse
from pathlib import Path

import torch
from torch.amp import autocast
from tqdm import tqdm

from ar_image_generation.approaches.custom.residual_quantization import (
    fit_residual_kmeans,
)
from ar_image_generation.config import DatasetConfig, load_yaml
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.data.medmnist import build_dataloaders
from ar_image_generation.engine.checkpointing import load_model_checkpoint
from ar_image_generation.tokenizers.graft_gs import (
    GaussianSplatTokenizer,
    GaussianTokenizerConfig,
)
from ar_image_generation.utils.device import get_device
from ar_image_generation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GRAFT-GS residual feature codebook."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to config YAML."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output codebook path."
    )
    parser.add_argument("--num-quantizers", type=int, default=2)
    parser.add_argument("--num-codes", type=int, default=1024)
    parser.add_argument("--max-features", type=int, default=500000)
    parser.add_argument("--num-iters", type=int, default=50)
    return parser.parse_args()


def move_batch_to_device(batch: ImageBatch, device: torch.device) -> ImageBatch:
    return ImageBatch(
        images=batch.images.to(device, non_blocking=True),
        labels=(
            None if batch.labels is None else batch.labels.to(device, non_blocking=True)
        ),
        metadata=batch.metadata,
    )


def build_tokenizer(config: dict) -> GaussianSplatTokenizer:
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


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed_everything(config["seed"])

    device = get_device(config["device"])
    mixed_precision = device.type == "cuda"

    dataset_cfg = DatasetConfig.model_validate(config["dataset"])
    train_loader, _, _ = build_dataloaders(dataset_cfg)

    tokenizer = build_tokenizer(config).to(device)
    load_model_checkpoint(
        path=config["tokenizer"]["checkpoint_path"],
        model=tokenizer,
        map_location=device,
    )
    tokenizer.eval()

    collected_features: list[torch.Tensor] = []
    total_features = 0

    progress = tqdm(train_loader, desc="collecting primitive features", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        with autocast(device_type=device.type, enabled=mixed_precision):
            primitives = tokenizer.encode(batch.images)

        features = (
            primitives.feature.reshape(-1, primitives.feature.shape[-1]).detach().cpu()
        )

        remaining = args.max_features - total_features
        if remaining <= 0:
            break

        if features.shape[0] > remaining:
            indices = torch.randperm(features.shape[0])[:remaining]
            features = features[indices]

        collected_features.append(features)
        total_features += features.shape[0]

        progress.set_postfix(collected=total_features)

        if total_features >= args.max_features:
            break

    all_features = torch.cat(collected_features, dim=0).float().to(device)

    print("Building GRAFT-GS residual feature codebook")
    print("-------------------------------------------")
    print(f"collected features: {all_features.shape[0]}")
    print(f"feature dim:        {all_features.shape[1]}")
    print(f"num quantizers:     {args.num_quantizers}")
    print(f"num codes:          {args.num_codes}")
    print(f"num iters:          {args.num_iters}")

    codebook = fit_residual_kmeans(
        all_features,
        num_quantizers=args.num_quantizers,
        num_codes=args.num_codes,
        num_iters=args.num_iters,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(codebook.state_dict(), args.output)

    print(f"saved residual codebook: {args.output}")


if __name__ == "__main__":
    main()
