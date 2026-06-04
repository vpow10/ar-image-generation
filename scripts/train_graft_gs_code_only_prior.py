import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
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
from ar_image_generation.engine.checkpointing import (
    load_model_checkpoint,
    save_model_checkpoint,
)
from ar_image_generation.tokenizers.graft_gs import (
    GaussianSplatTokenizer,
    GaussianTokenizerConfig,
)
from ar_image_generation.utils.device import get_device
from ar_image_generation.utils.image_grid import save_image_grid
from ar_image_generation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRAFT-GS code-only prior.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to config YAML."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs."
    )
    parser.add_argument("--run-name", type=str, default=None, help="Override run name.")
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


def build_tokenizer(config: dict[str, Any]) -> GaussianSplatTokenizer:
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


def build_prior(config: dict[str, Any]) -> ResidualCodeAR:
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


def freeze_model(model: nn.Module) -> None:
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def compute_geometry_templates(
    *,
    tokenizer: GaussianSplatTokenizer,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
    num_classes: int,
) -> torch.Tensor:
    sum_geometry = torch.zeros(
        num_classes,
        tokenizer.cfg.num_primitives,
        6,
        device=device,
        dtype=torch.float32,
    )
    counts = torch.zeros(num_classes, device=device, dtype=torch.float32)

    global_sum = torch.zeros(
        tokenizer.cfg.num_primitives,
        6,
        device=device,
        dtype=torch.float32,
    )
    global_count = 0.0

    for batch in tqdm(loader, desc="computing geometry templates", leave=False):
        batch = move_batch_to_device(batch, device)

        if batch.labels is None:
            raise ValueError("Class labels are required for class geometry templates.")

        labels = prepare_labels(batch.labels, num_classes)

        with autocast(device_type=device.type, enabled=mixed_precision):
            primitives = tokenizer.encode(batch.images)
            flattened = flatten_primitives(primitives)

        geometry = flattened[..., :6].float()

        global_sum = global_sum + geometry.sum(dim=0)
        global_count += float(geometry.shape[0])

        for class_index in range(num_classes):
            mask = labels == class_index
            if mask.any():
                sum_geometry[class_index] += geometry[mask].sum(dim=0)
                counts[class_index] += float(mask.sum().item())

    global_template = global_sum / max(global_count, 1.0)

    templates = torch.empty_like(sum_geometry)
    for class_index in range(num_classes):
        if counts[class_index].item() > 0:
            templates[class_index] = sum_geometry[class_index] / counts[class_index]
        else:
            templates[class_index] = global_template

    return templates.detach().cpu()


@torch.no_grad()
def encode_codes(
    *,
    tokenizer: GaussianSplatTokenizer,
    codebook: ResidualFeatureCodebook,
    images: torch.Tensor,
    mixed_precision: bool,
    device: torch.device,
) -> torch.LongTensor:
    with autocast(device_type=device.type, enabled=mixed_precision):
        primitives = tokenizer.encode(images)
        flattened = flatten_primitives(primitives)

    features = flattened[..., 6:]
    return codebook.assign(features)


def train_one_epoch(
    *,
    model: ResidualCodeAR,
    tokenizer: GaussianSplatTokenizer,
    codebook: ResidualFeatureCodebook,
    loader: torch.utils.data.DataLoader[ImageBatch],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    mixed_precision: bool,
    grad_clip_norm: float,
    epoch: int,
) -> dict[str, float]:
    model.train()
    tokenizer.eval()

    total_loss = 0.0
    total_code_loss = 0.0
    num_batches = 0

    progress = tqdm(
        loader, desc=f"train GRAFT-GS code-only prior epoch {epoch}", leave=False
    )

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            codes = encode_codes(
                tokenizer=tokenizer,
                codebook=codebook,
                images=batch.images,
                mixed_precision=mixed_precision,
                device=device,
            )

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(codes, batch.labels)
            loss = output.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        total_loss += output.loss.item()
        total_code_loss += output.code_loss.item()
        num_batches += 1

        progress.set_postfix(
            loss=f"{output.loss.item():.4f}",
            code=f"{output.code_loss.item():.4f}",
        )

    return {
        "loss": total_loss / num_batches,
        "code_loss": total_code_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    *,
    model: ResidualCodeAR,
    tokenizer: GaussianSplatTokenizer,
    codebook: ResidualFeatureCodebook,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
) -> dict[str, float]:
    model.eval()
    tokenizer.eval()

    total_loss = 0.0
    total_code_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        codes = encode_codes(
            tokenizer=tokenizer,
            codebook=codebook,
            images=batch.images,
            mixed_precision=mixed_precision,
            device=device,
        )

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(codes, batch.labels)

        total_loss += output.loss.item()
        total_code_loss += output.code_loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "code_loss": total_code_loss / num_batches,
    }


@torch.no_grad()
def generate_images(
    *,
    prior: ResidualCodeAR,
    tokenizer: GaussianSplatTokenizer,
    codebook: ResidualFeatureCodebook,
    geometry_templates: torch.Tensor,
    batch_size: int,
    labels: torch.Tensor | None,
    device: torch.device,
    code_temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    codes, generated_labels = prior.generate(
        batch_size=batch_size,
        labels=labels,
        device=device,
        code_temperature=code_temperature,
        top_k=top_k,
    )

    if generated_labels is None:
        geometry = geometry_templates.mean(dim=0, keepdim=True).expand(
            batch_size, -1, -1
        )
    else:
        geometry = geometry_templates.to(device=device, dtype=torch.float32)[
            generated_labels
        ]

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

    return tokenizer.decode(primitives)


@torch.no_grad()
def save_generated_samples(
    *,
    prior: ResidualCodeAR,
    tokenizer: GaussianSplatTokenizer,
    codebook: ResidualFeatureCodebook,
    geometry_templates: torch.Tensor,
    path: Path,
    device: torch.device,
    num_samples: int,
    code_temperature: float,
    top_k: int | None,
) -> None:
    images = generate_images(
        prior=prior,
        tokenizer=tokenizer,
        codebook=codebook,
        geometry_templates=geometry_templates,
        batch_size=num_samples,
        labels=None,
        device=device,
        code_temperature=code_temperature,
        top_k=top_k,
    )

    save_image_grid(images.detach().cpu(), path, nrow=8, normalized=True)


def make_run_dir(config: dict[str, Any], run_name_override: str | None) -> Path:
    run_name = run_name_override or config["logging"]["run_name"]
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed_everything(config["seed"])

    device = get_device(config["device"])
    mixed_precision = config["train"]["mixed_precision"] and device.type == "cuda"

    dataset_cfg = DatasetConfig.model_validate(config["dataset"])
    train_loader, val_loader, _ = build_dataloaders(dataset_cfg)

    tokenizer = build_tokenizer(config).to(device)
    load_model_checkpoint(
        path=config["tokenizer"]["checkpoint_path"],
        model=tokenizer,
        map_location=device,
    )
    freeze_model(tokenizer)

    codebook_state = torch.load(config["residual_codebook"]["path"], map_location="cpu")
    codebook = ResidualFeatureCodebook.from_state_dict(codebook_state)

    prior = build_prior(config).to(device)

    optimizer = AdamW(
        prior.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scaler = GradScaler(device=device.type, enabled=mixed_precision)

    epochs = args.epochs if args.epochs is not None else config["train"]["epochs"]

    run_dir = make_run_dir(config, args.run_name)
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config.json", config)

    geometry_templates = compute_geometry_templates(
        tokenizer=tokenizer,
        loader=train_loader,
        device=device,
        mixed_precision=mixed_precision,
        num_classes=config["prior"]["num_classes"],
    )
    torch.save(
        {"geometry_templates": geometry_templates},
        run_dir / "geometry_templates.pt",
    )

    print("GRAFT-GS code-only prior training")
    print("---------------------------------")
    print(f"dataset:         {dataset_cfg.name}")
    print(f"image size:      {dataset_cfg.size}")
    print(f"batch size:      {dataset_cfg.batch_size}")
    print(f"primitives:      {config['prior']['num_primitives']}")
    print(f"quantizers:      {config['prior']['num_quantizers']}")
    print(f"feature codes:   {config['prior']['num_feature_codes']}")
    print(f"model dim:       {config['prior']['dim']}")
    print(f"depth:           {config['prior']['depth']}")
    print(f"device:          {device}")
    print(f"mixed precision: {mixed_precision}")
    print(f"epochs:          {epochs}")
    print(f"run dir:         {run_dir}")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=prior,
            tokenizer=tokenizer,
            codebook=codebook,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixed_precision=mixed_precision,
            grad_clip_norm=config["train"]["grad_clip_norm"],
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=prior,
            tokenizer=tokenizer,
            codebook=codebook,
            loader=val_loader,
            device=device,
            mixed_precision=mixed_precision,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val code ce {val_metrics['code_loss']:.4f}"
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }

        append_jsonl(run_dir / "metrics.jsonl", record)
        save_json(run_dir / "latest_metrics.json", record)

        metrics = {
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }

        extra = {
            "geometry_templates": {"geometry_templates": geometry_templates},
            "residual_codebook_path": config["residual_codebook"]["path"],
            "tokenizer_checkpoint_path": config["tokenizer"]["checkpoint_path"],
        }

        save_model_checkpoint(
            path=checkpoints_dir / "last.pt",
            model=prior,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
            extra=extra,
        )

        if (
            epoch == 1
            or epoch % config["train"]["sample_every_epochs"] == 0
            or epoch == epochs
        ):
            save_generated_samples(
                prior=prior,
                tokenizer=tokenizer,
                codebook=codebook,
                geometry_templates=geometry_templates.to(device),
                path=samples_dir / f"epoch_{epoch:03d}.png",
                device=device,
                num_samples=config["sampling"]["num_samples"],
                code_temperature=config["sampling"]["code_temperature"],
                top_k=config["sampling"]["top_k"],
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            save_model_checkpoint(
                path=checkpoints_dir / "best.pt",
                model=prior,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
                extra=extra,
            )

            save_generated_samples(
                prior=prior,
                tokenizer=tokenizer,
                codebook=codebook,
                geometry_templates=geometry_templates.to(device),
                path=samples_dir / "best.png",
                device=device,
                num_samples=config["sampling"]["num_samples"],
                code_temperature=config["sampling"]["code_temperature"],
                top_k=config["sampling"]["top_k"],
            )

    summary = {
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(checkpoints_dir / "best.pt"),
        "tokenizer_checkpoint": config["tokenizer"]["checkpoint_path"],
        "residual_codebook_path": config["residual_codebook"]["path"],
    }
    save_json(run_dir / "summary.json", summary)

    print(f"best val loss: {best_val_loss:.4f}")
    print(f"best checkpoint: {checkpoints_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
