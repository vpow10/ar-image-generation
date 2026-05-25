import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from ar_image_generation.approaches.custom.model import (
    GaussianPrimitiveAR,
    GaussianPrimitiveARConfig,
)
from ar_image_generation.approaches.custom.normalization import PrimitiveNormalizer
from ar_image_generation.approaches.custom.primitives import (
    flatten_primitives,
    primitives_from_flattened,
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
    parser = argparse.ArgumentParser(description="Train GRAFT-GS autoregressive prior.")
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


def freeze_model(model: nn.Module) -> None:
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def compute_primitive_normalizer(
    *,
    tokenizer: GaussianSplatTokenizer,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
) -> PrimitiveNormalizer:
    tokenizer.eval()

    total_sum: torch.Tensor | None = None
    total_square_sum: torch.Tensor | None = None
    total_count = 0

    for batch in tqdm(loader, desc="computing primitive normalizer", leave=False):
        batch = move_batch_to_device(batch, device)

        with autocast(device_type=device.type, enabled=mixed_precision):
            primitives = tokenizer.encode(batch.images)
            flattened_primitives = flatten_primitives(primitives)

        flat = flattened_primitives.reshape(-1, flattened_primitives.shape[-1]).float()

        batch_sum = flat.sum(dim=0)
        batch_square_sum = flat.square().sum(dim=0)

        if total_sum is None:
            total_sum = batch_sum
            total_square_sum = batch_square_sum
        else:
            total_sum = total_sum + batch_sum
            assert total_square_sum is not None
            total_square_sum = total_square_sum + batch_square_sum

        total_count += flat.shape[0]

    assert total_sum is not None
    assert total_square_sum is not None

    mean = total_sum / total_count
    variance = total_square_sum / total_count - mean.square()
    std = variance.clamp_min(1e-6).sqrt()

    return PrimitiveNormalizer(
        mean=mean.detach().cpu(),
        std=std.detach().cpu(),
    )


@torch.no_grad()
def encode_batch_primitives(
    *,
    tokenizer: GaussianSplatTokenizer,
    batch: ImageBatch,
    normalizer: PrimitiveNormalizer,
    device: torch.device,
    mixed_precision: bool,
) -> torch.Tensor:
    batch = move_batch_to_device(batch, device)

    with autocast(device_type=device.type, enabled=mixed_precision):
        raw_primitives = tokenizer.encoder(batch.images)

    return normalizer.normalize(raw_primitives)


def train_one_epoch(
    *,
    model: GaussianPrimitiveAR,
    tokenizer: GaussianSplatTokenizer,
    normalizer: PrimitiveNormalizer,
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
    total_nll_loss = 0.0
    total_mean_loss = 0.0
    num_batches = 0

    progress = tqdm(loader, desc=f"train GRAFT-GS prior epoch {epoch}", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            with autocast(device_type=device.type, enabled=mixed_precision):
                encoded_primitives = tokenizer.encode(batch.images)
                flattened_primitives = flatten_primitives(encoded_primitives)

            primitives = normalizer.normalize(flattened_primitives)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(primitives, batch.labels)
            loss = output.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        total_loss += output.loss.item()
        total_nll_loss += output.nll_loss.item()
        total_mean_loss += output.mean_loss.item()
        num_batches += 1

        progress.set_postfix(
            loss=f"{output.loss.item():.4f}",
            nll=f"{output.nll_loss.item():.4f}",
            mse=f"{output.mean_loss.item():.4f}",
        )

    return {
        "loss": total_loss / num_batches,
        "nll_loss": total_nll_loss / num_batches,
        "mean_loss": total_mean_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    *,
    model: GaussianPrimitiveAR,
    tokenizer: GaussianSplatTokenizer,
    normalizer: PrimitiveNormalizer,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
) -> dict[str, float]:
    model.eval()
    tokenizer.eval()

    total_loss = 0.0
    total_nll_loss = 0.0
    total_mean_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with autocast(device_type=device.type, enabled=mixed_precision):
            encoded_primitives = tokenizer.encode(batch.images)
            flattened_primitives = flatten_primitives(encoded_primitives)

        primitives = normalizer.normalize(flattened_primitives)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(primitives, batch.labels)

        total_loss += output.loss.item()
        total_nll_loss += output.nll_loss.item()
        total_mean_loss += output.mean_loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "nll_loss": total_nll_loss / num_batches,
        "mean_loss": total_mean_loss / num_batches,
    }


@torch.no_grad()
def generate_images(
    *,
    prior: GaussianPrimitiveAR,
    tokenizer: GaussianSplatTokenizer,
    normalizer: PrimitiveNormalizer,
    batch_size: int,
    labels: torch.Tensor | None,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    normalized_primitives = prior.generate(
        batch_size=batch_size,
        labels=labels,
        device=device,
        temperature=temperature,
        geometry_noise_scale=0.7,
        feature_noise_scale=0.05,
    )

    normalized_primitives = normalized_primitives.clamp(-3.0, 3.0)
    flattened_primitives = normalizer.denormalize(normalized_primitives)

    primitives = primitives_from_flattened(
        flattened_primitives,
        tokenizer.primitive_cfg,
        anchors=tokenizer.encoder.anchors,
    )

    return tokenizer.decode(primitives)


@torch.no_grad()
def save_generated_samples(
    *,
    prior: GaussianPrimitiveAR,
    tokenizer: GaussianSplatTokenizer,
    normalizer: PrimitiveNormalizer,
    path: Path,
    device: torch.device,
    num_samples: int,
    temperature: float,
) -> None:
    images = generate_images(
        prior=prior,
        tokenizer=tokenizer,
        normalizer=normalizer,
        batch_size=num_samples,
        labels=None,
        device=device,
        temperature=temperature,
    )

    save_image_grid(
        images.detach().cpu(),
        path,
        nrow=8,
        normalized=True,
    )


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

    tokenizer = build_graft_tokenizer(config).to(device)

    tokenizer_checkpoint_path = Path(config["tokenizer"]["checkpoint_path"])
    if not tokenizer_checkpoint_path.exists():
        raise FileNotFoundError(
            f"GRAFT-GS tokenizer checkpoint not found: {tokenizer_checkpoint_path}"
        )

    load_model_checkpoint(
        path=tokenizer_checkpoint_path,
        model=tokenizer,
        map_location=device,
    )
    freeze_model(tokenizer)

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

    normalizer = compute_primitive_normalizer(
        tokenizer=tokenizer,
        loader=train_loader,
        device=device,
        mixed_precision=mixed_precision,
    )
    torch.save(normalizer.state_dict(), run_dir / "primitive_normalizer.pt")

    print("GRAFT-GS prior training")
    print("-----------------------")
    print(f"dataset:        {dataset_cfg.name}")
    print(f"image size:     {dataset_cfg.size}")
    print(f"batch size:     {dataset_cfg.batch_size}")
    print(f"primitives:     {config['prior']['num_primitives']}")
    print(f"primitive dim:  {config['prior']['primitive_dim']}")
    print(f"model dim:      {config['prior']['dim']}")
    print(f"depth:          {config['prior']['depth']}")
    print(f"heads:          {config['prior']['num_heads']}")
    print(f"device:         {device}")
    print(f"mixed precision:{mixed_precision}")
    print(f"epochs:         {epochs}")
    print(f"run dir:        {run_dir}")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=prior,
            tokenizer=tokenizer,
            normalizer=normalizer,
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
            normalizer=normalizer,
            loader=val_loader,
            device=device,
            mixed_precision=mixed_precision,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val nll {val_metrics['nll_loss']:.4f} | "
            f"val mse {val_metrics['mean_loss']:.4f}"
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
            "normalizer": normalizer.state_dict(),
            "tokenizer_checkpoint_path": str(tokenizer_checkpoint_path),
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
                normalizer=normalizer,
                path=samples_dir / f"epoch_{epoch:03d}.png",
                device=device,
                num_samples=config["sampling"]["num_samples"],
                temperature=config["sampling"]["temperature"],
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
                normalizer=normalizer,
                path=samples_dir / "best.png",
                device=device,
                num_samples=config["sampling"]["num_samples"],
                temperature=config["sampling"]["temperature"],
            )

    summary = {
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(checkpoints_dir / "best.pt"),
        "tokenizer_checkpoint": str(tokenizer_checkpoint_path),
    }
    save_json(run_dir / "summary.json", summary)

    print(f"best val loss: {best_val_loss:.4f}")
    print(f"best checkpoint: {checkpoints_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
