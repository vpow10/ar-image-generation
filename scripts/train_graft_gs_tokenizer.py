import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from ar_image_generation.config import DatasetConfig, load_yaml
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.data.medmnist import build_dataloaders
from ar_image_generation.engine.checkpointing import save_model_checkpoint
from ar_image_generation.tokenizers.graft_gs import (
    GaussianSplatTokenizer,
    GaussianTokenizerConfig,
)
from ar_image_generation.utils.device import get_device
from ar_image_generation.utils.image_grid import save_image_grid
from ar_image_generation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRAFT-GS Gaussian tokenizer.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name.")
    return parser.parse_args()


def move_batch_to_device(batch: ImageBatch, device: torch.device) -> ImageBatch:
    return ImageBatch(
        images=batch.images.to(device, non_blocking=True),
        labels=(None if batch.labels is None else batch.labels.to(device, non_blocking=True)),
        metadata=batch.metadata,
    )


def build_model(config: dict[str, Any]) -> GaussianSplatTokenizer:
    tokenizer_cfg = config["tokenizer"]
    train_cfg = config["train"]

    model_cfg = GaussianTokenizerConfig(
        image_size=tokenizer_cfg["image_size"],
        image_channels=tokenizer_cfg["image_channels"],
        num_primitives=tokenizer_cfg["num_primitives"],
        primitive_feature_dim=tokenizer_cfg["primitive_feature_dim"],
        hidden_channels=tokenizer_cfg["hidden_channels"],
        min_scale=tokenizer_cfg["min_scale"],
        max_scale=tokenizer_cfg["max_scale"],
        max_position_offset=tokenizer_cfg.get("max_position_offset", 0.035),
        renderer_chunk_size=tokenizer_cfg["renderer_chunk_size"],
    )

    return GaussianSplatTokenizer(
        model_cfg,
        reconstruction_l1_weight=train_cfg["reconstruction_l1_weight"],
        reconstruction_mse_weight=train_cfg["reconstruction_mse_weight"],
        opacity_weight=train_cfg["opacity_weight"],
        scale_weight=train_cfg["scale_weight"],
        coverage_weight=train_cfg.get("coverage_weight", 0.02),
    )


def make_run_dir(config: dict[str, Any], run_name_override: str | None) -> Path:
    run_name = run_name_override or config["logging"]["run_name"]
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(config: dict[str, Any], run_dir: Path) -> None:
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        raise ValueError("Cannot average empty metrics list.")

    keys = metrics[0].keys()

    return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}


def output_to_metrics(output) -> dict[str, float]:
    return {
        "loss": output.loss.item(),
        "reconstruction_loss": output.reconstruction_loss.item(),
        "l1_reconstruction_loss": output.l1_reconstruction_loss.item(),
        "mse_reconstruction_loss": output.mse_reconstruction_loss.item(),
        "opacity_loss": output.opacity_loss.item(),
        "scale_loss": output.scale_loss.item(),
        "coverage_loss": output.coverage_loss.item(),
        "mean_opacity": output.primitives.opacity.mean().item(),
        "mean_scale": output.primitives.scale.mean().item(),
        "mean_coverage": output.feature_map[:, -1:].mean().item(),
    }


def train_one_epoch(
    *,
    model: GaussianSplatTokenizer,
    loader: torch.utils.data.DataLoader[ImageBatch],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    mixed_precision: bool,
    grad_clip_norm: float,
    epoch: int,
) -> dict[str, float]:
    model.train()

    collected_metrics: list[dict[str, float]] = []
    progress = tqdm(loader, desc=f"train GRAFT-GS tokenizer epoch {epoch}", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(batch.images)
            loss = output.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        metrics = output_to_metrics(output)
        collected_metrics.append(metrics)

        progress.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            recon=f"{metrics['reconstruction_loss']:.4f}",
            opacity=f"{metrics['mean_opacity']:.3f}",
            coverage=f"{metrics['mean_coverage']:.3f}",
            scale=f"{metrics['mean_scale']:.3f}",
        )

    return average_metrics(collected_metrics)


@torch.no_grad()
def evaluate(
    *,
    model: GaussianSplatTokenizer,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
) -> dict[str, float]:
    model.eval()

    collected_metrics: list[dict[str, float]] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(batch.images)

        collected_metrics.append(output_to_metrics(output))

    return average_metrics(collected_metrics)


@torch.no_grad()
def save_reconstruction_grid(
    *,
    model: GaussianSplatTokenizer,
    batch: ImageBatch,
    path: Path,
    device: torch.device,
    mixed_precision: bool,
) -> None:
    model.eval()

    images = batch.images.to(device)

    with autocast(device_type=device.type, enabled=mixed_precision):
        output = model(images)

    originals = images[:16].detach().cpu()
    reconstructions = output.reconstructions[:16].detach().cpu()

    comparison = torch.cat([originals, reconstructions], dim=0)
    save_image_grid(comparison, path, nrow=8, normalized=True)


def save_metrics_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed_everything(config["seed"])

    device = get_device(config["device"])
    mixed_precision = config["train"]["mixed_precision"] and device.type == "cuda"

    dataset_cfg = DatasetConfig.model_validate(config["dataset"])
    train_loader, val_loader, _ = build_dataloaders(dataset_cfg)
    fixed_val_batch = next(iter(val_loader))

    model = build_model(config).to(device)

    optimizer = AdamW(
        model.parameters(),
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
    save_config_snapshot(config, run_dir)

    checkpoint_path = Path(config["tokenizer"]["checkpoint_path"])

    print("GRAFT-GS tokenizer training")
    print("---------------------------")
    print(f"dataset:        {dataset_cfg.name}")
    print(f"image size:     {dataset_cfg.size}")
    print(f"batch size:     {dataset_cfg.batch_size}")
    print(f"primitives:     {config['tokenizer']['num_primitives']}")
    print(f"feature dim:    {config['tokenizer']['primitive_feature_dim']}")
    print(f"position offset:{config['tokenizer'].get('max_position_offset', 0.035)}")
    print(f"hidden ch:      {config['tokenizer']['hidden_channels']}")
    print(
        f"scale range:    {config['tokenizer']['min_scale']} - {config['tokenizer']['max_scale']}"
    )
    print(f"device:         {device}")
    print(f"mixed precision:{mixed_precision}")
    print(f"epochs:         {epochs}")
    print(f"run dir:        {run_dir}")
    print(f"checkpoint:     {checkpoint_path}")

    best_val_reconstruction = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixed_precision=mixed_precision,
            grad_clip_norm=config["train"]["grad_clip_norm"],
            epoch=epoch,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            mixed_precision=mixed_precision,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val recon {val_metrics['reconstruction_loss']:.4f} | "
            f"opacity {val_metrics['mean_opacity']:.3f} | "
            f"coverage {val_metrics['mean_coverage']:.3f} | "
            f"scale {val_metrics['mean_scale']:.3f}"
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        save_metrics_jsonl(run_dir / "metrics.jsonl", record)

        save_model_checkpoint(
            path=checkpoints_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            },
            config=config,
        )

        should_sample = (
            epoch == 1 or epoch % config["train"]["sample_every_epochs"] == 0 or epoch == epochs
        )

        if should_sample:
            save_reconstruction_grid(
                model=model,
                batch=fixed_val_batch,
                path=samples_dir / f"epoch_{epoch:03d}.png",
                device=device,
                mixed_precision=mixed_precision,
            )

        if val_metrics["reconstruction_loss"] < best_val_reconstruction:
            best_val_reconstruction = val_metrics["reconstruction_loss"]

            best_path = checkpoints_dir / "best.pt"

            save_model_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                },
                config=config,
            )

            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(best_path, checkpoint_path)

            save_reconstruction_grid(
                model=model,
                batch=fixed_val_batch,
                path=samples_dir / "best.png",
                device=device,
                mixed_precision=mixed_precision,
            )

    summary = {
        "best_val_reconstruction_loss": best_val_reconstruction,
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"best val reconstruction loss: {best_val_reconstruction:.4f}")
    print(f"saved tokenizer: {checkpoint_path}")


if __name__ == "__main__":
    main()
