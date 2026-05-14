import argparse
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from ar_image_generation.config import ExperimentConfig, load_experiment_config
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.data.medmnist import build_dataloaders
from ar_image_generation.tokenizers.checkpoint import save_tokenizer_checkpoint
from ar_image_generation.tokenizers.factory import build_tokenizer
from ar_image_generation.tokenizers.vqvae import VQVAE
from ar_image_generation.utils.device import get_device
from ar_image_generation.utils.image_grid import save_image_grid
from ar_image_generation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VQ-VAE image tokenizer.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment config YAML."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override tokenizer training epochs from config.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run name.",
    )
    return parser.parse_args()


def move_batch_to_device(batch: ImageBatch, device: torch.device) -> ImageBatch:
    return ImageBatch(
        images=batch.images.to(device, non_blocking=True),
        labels=(None if batch.labels is None else batch.labels.to(device, non_blocking=True)),
        metadata=batch.metadata,
    )


def train_one_epoch(
    *,
    model: VQVAE,
    loader: torch.utils.data.DataLoader[ImageBatch],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    mixed_precision: bool,
    epoch: int,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_quantization_loss = 0.0
    total_commitment_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    progress = tqdm(loader, desc=f"train tokenizer epoch {epoch}", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(batch.images)
            loss = output.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += output.loss.item()
        total_reconstruction_loss += output.reconstruction_loss.item()
        total_quantization_loss += output.quantization_loss.item()
        total_commitment_loss += output.commitment_loss.item()
        total_perplexity += output.perplexity.item()
        num_batches += 1

        progress.set_postfix(
            loss=f"{output.loss.item():.4f}",
            recon=f"{output.reconstruction_loss.item():.4f}",
            ppl=f"{output.perplexity.item():.1f}",
        )

    return {
        "loss": total_loss / num_batches,
        "reconstruction_loss": total_reconstruction_loss / num_batches,
        "quantization_loss": total_quantization_loss / num_batches,
        "commitment_loss": total_commitment_loss / num_batches,
        "perplexity": total_perplexity / num_batches,
    }


@torch.no_grad()
def evaluate(
    *,
    model: VQVAE,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_quantization_loss = 0.0
    total_commitment_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(batch.images)

        total_loss += output.loss.item()
        total_reconstruction_loss += output.reconstruction_loss.item()
        total_quantization_loss += output.quantization_loss.item()
        total_commitment_loss += output.commitment_loss.item()
        total_perplexity += output.perplexity.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "reconstruction_loss": total_reconstruction_loss / num_batches,
        "quantization_loss": total_quantization_loss / num_batches,
        "commitment_loss": total_commitment_loss / num_batches,
        "perplexity": total_perplexity / num_batches,
    }


@torch.no_grad()
def save_reconstruction_grid(
    *,
    model: VQVAE,
    batch: ImageBatch,
    path: Path,
    device: torch.device,
) -> None:
    model.eval()

    images = batch.images.to(device)
    output = model(images)

    originals = images[:16].detach().cpu()
    reconstructions = output.reconstructions[:16].detach().cpu()

    comparison = torch.cat([originals, reconstructions], dim=0)
    save_image_grid(comparison, path, nrow=8, normalized=True)


def make_run_dir(cfg: ExperimentConfig, run_name_override: str | None) -> Path:
    run_name = run_name_override or f"tokenizer_{cfg.logging.run_name}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()

    cfg = load_experiment_config(args.config)
    seed_everything(cfg.seed)

    device = get_device(cfg.device)
    mixed_precision = cfg.train.mixed_precision and device.type == "cuda"

    train_loader, val_loader, _ = build_dataloaders(cfg.dataset)
    fixed_val_batch = next(iter(val_loader))

    model = build_tokenizer(
        cfg.tokenizer,
        image_size=cfg.dataset.size,
        image_channels=3,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.tokenizer.train.lr,
        weight_decay=cfg.tokenizer.train.weight_decay,
    )
    scaler = GradScaler(device=device.type, enabled=mixed_precision)

    epochs = args.epochs if args.epochs is not None else cfg.tokenizer.train.epochs
    run_dir = make_run_dir(cfg, args.run_name)

    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    print("Tokenizer training")
    print("------------------")
    print(f"dataset:       {cfg.dataset.name}")
    print(f"image size:    {cfg.dataset.size}")
    print(f"batch size:    {cfg.dataset.batch_size}")
    print(f"latent shape:  {model.latent_shape}")
    print(f"vocab size:    {model.vocab_size}")
    print(f"embedding dim: {cfg.tokenizer.embedding_dim}")
    print(f"hidden ch:     {cfg.tokenizer.hidden_channels}")
    print(f"downsample:    {cfg.tokenizer.downsample_factor}")
    print(f"ckpt path:     {cfg.tokenizer.checkpoint_path}")
    print(f"device:        {device}")
    print(f"mixed precision: {mixed_precision}")
    print(f"run dir:       {run_dir}")

    best_val_reconstruction_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixed_precision=mixed_precision,
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
            f"val perplexity {val_metrics['perplexity']:.1f}"
        )

        metrics = {
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }

        last_checkpoint_path = checkpoints_dir / "last.pt"
        save_tokenizer_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=cfg.model_dump(mode="json"),
        )

        save_reconstruction_grid(
            model=model,
            batch=fixed_val_batch,
            path=samples_dir / f"epoch_{epoch:03d}.png",
            device=device,
        )

        if val_metrics["reconstruction_loss"] < best_val_reconstruction_loss:
            best_val_reconstruction_loss = val_metrics["reconstruction_loss"]
            best_checkpoint_path = checkpoints_dir / "best.pt"

            save_tokenizer_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=cfg.model_dump(mode="json"),
            )

            cfg.tokenizer.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(best_checkpoint_path, cfg.tokenizer.checkpoint_path)

    print(f"best val reconstruction loss: {best_val_reconstruction_loss:.4f}")
    print(f"saved tokenizer: {cfg.tokenizer.checkpoint_path}")


if __name__ == "__main__":
    main()
