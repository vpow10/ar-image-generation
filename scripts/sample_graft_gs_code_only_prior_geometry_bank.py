import argparse
from pathlib import Path

import torch
from torch.amp import autocast
from tqdm import tqdm

from ar_image_generation.approaches.custom.code_only_model import (
    ResidualCodeAR,
    ResidualCodeARConfig,
)
from ar_image_generation.approaches.custom.primitives import (
    GaussianPrimitives,
    flatten_primitives,
)
from ar_image_generation.approaches.custom.residual_quantization import (
    ResidualFeatureCodebook,
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
from ar_image_generation.utils.image_grid import save_image_grid
from ar_image_generation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from GRAFT-GS code-only prior with a real geometry bank."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--code-temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--max-bank-per-class", type=int, default=512)
    return parser.parse_args()


def move_batch_to_device(batch: ImageBatch, device: torch.device) -> ImageBatch:
    return ImageBatch(
        images=batch.images.to(device, non_blocking=True),
        labels=(
            None if batch.labels is None else batch.labels.to(device, non_blocking=True)
        ),
        metadata=batch.metadata,
    )


def prepare_labels(labels: torch.Tensor, num_classes: int) -> torch.LongTensor:
    labels = labels.long()

    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)

    if labels.ndim != 1:
        raise ValueError(f"Expected labels [B], got shape {tuple(labels.shape)}")

    if labels.min().item() < 0 or labels.max().item() >= num_classes:
        raise ValueError("Labels are outside the configured class range.")

    return labels


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


def build_prior(config: dict) -> ResidualCodeAR:
    prior_cfg = config["prior"]

    return ResidualCodeAR(
        ResidualCodeARConfig(
            num_primitives=prior_cfg["num_primitives"],
            num_quantizers=prior_cfg["num_quantizers"],
            num_feature_codes=prior_cfg["num_feature_codes"],
            dim=prior_cfg["dim"],
            depth=prior_cfg["depth"],
            num_heads=prior_cfg["num_heads"],
            mlp_ratio=prior_cfg["mlp_ratio"],
            dropout=prior_cfg["dropout"],
            class_conditional=prior_cfg["class_conditional"],
            num_classes=prior_cfg["num_classes"],
        )
    )


@torch.no_grad()
def build_geometry_bank(
    *,
    tokenizer: GaussianSplatTokenizer,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
    num_classes: int,
    max_bank_per_class: int,
) -> list[torch.Tensor]:
    geometry_bank: list[list[torch.Tensor]] = [[] for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]

    for batch in tqdm(loader, desc="building geometry bank", leave=False):
        batch = move_batch_to_device(batch, device)

        if batch.labels is None:
            raise ValueError("Labels are required to build class geometry bank.")

        labels = prepare_labels(batch.labels, num_classes)

        with autocast(device_type=device.type, enabled=mixed_precision):
            primitives = tokenizer.encode(batch.images)
            flattened = flatten_primitives(primitives)

        geometry = flattened[..., :6].detach().float().cpu()
        labels_cpu = labels.detach().cpu()

        for item_index in range(geometry.shape[0]):
            class_index = int(labels_cpu[item_index].item())

            if counts[class_index] < max_bank_per_class:
                geometry_bank[class_index].append(geometry[item_index])
                counts[class_index] += 1

        if all(count >= max_bank_per_class for count in counts):
            break

    output_bank: list[torch.Tensor] = []

    all_geometries = [item for class_items in geometry_bank for item in class_items]

    if not all_geometries:
        raise ValueError("Geometry bank is empty.")

    global_fallback = torch.stack(all_geometries, dim=0)

    for class_index, class_items in enumerate(geometry_bank):
        if class_items:
            output_bank.append(torch.stack(class_items, dim=0))
        else:
            output_bank.append(global_fallback)

    return output_bank


def sample_labels(
    *,
    num_samples: int,
    label: int | None,
    num_classes: int,
    device: torch.device,
) -> torch.LongTensor:
    if label is not None:
        if label < 0 or label >= num_classes:
            raise ValueError(f"label must be in [0, {num_classes - 1}], got {label}")

        return torch.full(
            (num_samples,),
            label,
            dtype=torch.long,
            device=device,
        )

    return torch.randint(
        low=0,
        high=num_classes,
        size=(num_samples,),
        dtype=torch.long,
        device=device,
    )


def sample_geometry_from_bank(
    *,
    geometry_bank: list[torch.Tensor],
    labels: torch.LongTensor,
    device: torch.device,
) -> torch.Tensor:
    sampled: list[torch.Tensor] = []

    labels_cpu = labels.detach().cpu()

    for label in labels_cpu.tolist():
        class_bank = geometry_bank[int(label)]
        index = torch.randint(low=0, high=class_bank.shape[0], size=(1,)).item()
        sampled.append(class_bank[index])

    return torch.stack(sampled, dim=0).to(device=device, dtype=torch.float32)


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

    prior = build_prior(config).to(device)
    load_model_checkpoint(
        path=args.checkpoint,
        model=prior,
        map_location=device,
    )
    prior.eval()

    codebook_state = torch.load(config["residual_codebook"]["path"], map_location="cpu")
    codebook = ResidualFeatureCodebook.from_state_dict(codebook_state)

    num_classes = config["prior"]["num_classes"]
    num_samples = args.num_samples or config["sampling"]["num_samples"]
    code_temperature = (
        args.code_temperature
        if args.code_temperature is not None
        else config["sampling"]["code_temperature"]
    )
    top_k = args.top_k if args.top_k is not None else config["sampling"]["top_k"]

    geometry_bank = build_geometry_bank(
        tokenizer=tokenizer,
        loader=train_loader,
        device=device,
        mixed_precision=mixed_precision,
        num_classes=num_classes,
        max_bank_per_class=args.max_bank_per_class,
    )

    labels = sample_labels(
        num_samples=num_samples,
        label=args.label,
        num_classes=num_classes,
        device=device,
    )

    codes, _ = prior.generate(
        batch_size=num_samples,
        labels=labels,
        device=device,
        code_temperature=code_temperature,
        top_k=top_k,
    )

    geometry = sample_geometry_from_bank(
        geometry_bank=geometry_bank,
        labels=labels,
        device=device,
    )

    features = codebook.lookup(codes).to(device=device, dtype=geometry.dtype)
    flattened = torch.cat([geometry, features], dim=-1)

    primitives = GaussianPrimitives(
        position=flattened[..., 0:2].clamp(0.0, 1.0),
        scale=flattened[..., 2:4].clamp(
            tokenizer.primitive_cfg.min_scale, tokenizer.primitive_cfg.max_scale
        ),
        rotation=flattened[..., 4:5].clamp(-torch.pi, torch.pi),
        opacity=flattened[..., 5:6].clamp(0.0, 1.0),
        feature=flattened[..., 6:].clamp(-1.0, 1.0),
    )

    images = tokenizer.decode(primitives)

    save_image_grid(images.detach().cpu(), args.output, nrow=8, normalized=True)

    print("GRAFT-GS geometry-bank sampling complete")
    print("----------------------------------------")
    print(f"checkpoint:          {args.checkpoint}")
    print(f"output:              {args.output}")
    print(f"samples:             {num_samples}")
    print(f"code temperature:    {code_temperature}")
    print(f"top_k:               {top_k}")
    print(f"label:               {args.label}")
    print(f"max bank per class:  {args.max_bank_per_class}")


if __name__ == "__main__":
    main()
