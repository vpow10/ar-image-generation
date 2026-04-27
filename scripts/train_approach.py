import argparse
import json
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from ar_image_generation.approaches.base import SamplingConfig
from ar_image_generation.approaches.factory import build_approach_from_config
from ar_image_generation.config import ExperimentConfig, load_experiment_config
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.data.medmnist import build_dataloaders
from ar_image_generation.engine.checkpointing import save_model_checkpoint
from ar_image_generation.tokenizers.checkpoint import load_tokenizer_checkpoint
from ar_image_generation.tokenizers.factory import build_tokenizer
from ar_image_generation.tokenizers.base import ImageTokenizer
from ar_image_generation.approaches.base import AutoregressiveApproach
from ar_image_generation.utils.device import get_device
from ar_image_generation.utils.image_grid import save_image_grid
from ar_image_generation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an autoregressive image generation approach."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment config YAML."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name.")
    return parser.parse_args()


def move_batch_to_device(batch: ImageBatch, device: torch.device) -> ImageBatch:
    return ImageBatch(
        images=batch.images.to(device, non_blocking=True),
        labels=None if batch.labels is None else batch.labels.to(device, non_blocking=True),
        metadata=batch.metadata,
    )


def freeze_tokenizer(tokenizer: ImageTokenizer) -> None:
    tokenizer.eval()

    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)


def make_sampling_config(cfg: ExperimentConfig) -> SamplingConfig:
    return SamplingConfig(
        temperature=cfg.sampling.temperature,
        top_k=cfg.sampling.top_k,
        top_p=cfg.sampling.top_p,
        num_samples=cfg.sampling.num_samples,
    )


def make_run_dir(cfg: ExperimentConfig, run_name_override: str | None) -> Path:
    run_name = run_name_override or cfg.logging.run_name
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(cfg: ExperimentConfig, run_dir: Path) -> None:
    path = run_dir / "config.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(cfg.model_dump(mode="json"), file, indent=2)


def load_trained_tokenizer(
    *,
    cfg: ExperimentConfig,
    device: torch.device,
) -> ImageTokenizer:
    tokenizer = build_tokenizer(
        cfg.tokenizer,
        image_size=cfg.dataset.size,
        image_channels=3,
    ).to(device)

    checkpoint_path = cfg.tokenizer.checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Tokenizer checkpoint does not exist. Train the tokenizer first:\n"
            "uv run python scripts/train_tokenizer.py "
            f"--config <your_config>\n\nMissing checkpoint: {checkpoint_path}"
        )

    load_tokenizer_checkpoint(
        path=checkpoint_path,
        model=tokenizer,
        map_location=device,
    )

    freeze_tokenizer(tokenizer)

    return tokenizer


def train_one_epoch(
    *,
    model: AutoregressiveApproach,
    tokenizer: ImageTokenizer,
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
    num_batches = 0

    progress = tqdm(loader, desc=f"train approach epoch {epoch}", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model.training_step(batch, tokenizer)
            loss = output["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    *,
    model: AutoregressiveApproach,
    tokenizer: ImageTokenizer,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
) -> dict[str, float]:
    model.eval()
    tokenizer.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model.training_step(batch, tokenizer)
            loss = output["loss"]

        total_loss += loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
    }


@torch.no_grad()
def save_generated_samples(
    *,
    model: AutoregressiveApproach,
    tokenizer: ImageTokenizer,
    path: Path,
    device: torch.device,
    sampling_cfg: SamplingConfig,
) -> None:
    model.eval()
    tokenizer.eval()

    images = model.generate(
        tokenizer=tokenizer,
        batch_size=sampling_cfg.num_samples,
        labels=None,
        device=device,
        sampling_cfg=sampling_cfg,
    )

    save_image_grid(
        images.detach().cpu(),
        path,
        nrow=8,
        normalized=True,
    )


def main() -> None:
    args = parse_args()

    cfg = load_experiment_config(args.config)
    seed_everything(cfg.seed)

    device = get_device(cfg.device)
    mixed_precision = cfg.train.mixed_precision and device.type == "cuda"

    run_dir = make_run_dir(cfg, args.run_name)
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(cfg, run_dir)

    train_loader, val_loader, _ = build_dataloaders(cfg.dataset)

    tokenizer = load_trained_tokenizer(cfg=cfg, device=device)

    model = build_approach_from_config(
        cfg.approach,
        vocab_size=tokenizer.vocab_size,
        latent_shape=tokenizer.latent_shape,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scaler = GradScaler(device=device.type, enabled=mixed_precision)
    sampling_cfg = make_sampling_config(cfg)

    epochs = args.epochs if args.epochs is not None else cfg.train.epochs

    print("Approach training")
    print("-----------------")
    print(f"dataset:          {cfg.dataset.name}")
    print(f"image size:       {cfg.dataset.size}")
    print(f"batch size:       {cfg.dataset.batch_size}")
    print(f"approach:         {cfg.approach.name}")
    print(f"tokenizer ckpt:   {cfg.tokenizer.checkpoint_path}")
    print(f"tokenizer vocab:  {tokenizer.vocab_size}")
    print(f"latent shape:     {tokenizer.latent_shape}")
    print(f"device:           {device}")
    print(f"mixed precision:  {mixed_precision}")
    print(f"epochs:           {epochs}")
    print(f"run dir:          {run_dir}")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixed_precision=mixed_precision,
            grad_clip_norm=cfg.train.grad_clip_norm,
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=model,
            tokenizer=tokenizer,
            loader=val_loader,
            device=device,
            mixed_precision=mixed_precision,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}"
        )

        metrics = {
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }

        save_model_checkpoint(
            path=checkpoints_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=cfg.model_dump(mode="json"),
            extra={
                "tokenizer_checkpoint_path": str(cfg.tokenizer.checkpoint_path),
                "approach_name": cfg.approach.name,
            },
        )

        if epoch % cfg.train.sample_every_epochs == 0 or epoch == 1:
            save_generated_samples(
                model=model,
                tokenizer=tokenizer,
                path=samples_dir / f"epoch_{epoch:03d}.png",
                device=device,
                sampling_cfg=sampling_cfg,
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            best_path = checkpoints_dir / "best.pt"

            save_model_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=cfg.model_dump(mode="json"),
                extra={
                    "tokenizer_checkpoint_path": str(cfg.tokenizer.checkpoint_path),
                    "approach_name": cfg.approach.name,
                },
            )

            stable_checkpoint_dir = Path("checkpoints") / "approaches" / cfg.approach.name
            stable_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(best_path, stable_checkpoint_dir / "best.pt")

    save_generated_samples(
        model=model,
        tokenizer=tokenizer,
        path=samples_dir / "final.png",
        device=device,
        sampling_cfg=sampling_cfg,
    )

    print(f"best val loss: {best_val_loss:.4f}")
    print(f"best checkpoint: {checkpoints_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
