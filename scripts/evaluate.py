import argparse
from pathlib import Path

import torch
from torch.amp import autocast
from tqdm import tqdm

from ar_image_generation.approaches.base import AutoregressiveApproach, SamplingConfig
from ar_image_generation.approaches.factory import build_approach_from_config
from ar_image_generation.config import ExperimentConfig, load_experiment_config
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.data.medmnist import build_dataloaders
from ar_image_generation.engine.checkpointing import load_model_checkpoint
from ar_image_generation.evaluation.io import save_json
from ar_image_generation.evaluation.metrics import count_parameters, measure_sampling_speed
from ar_image_generation.tokenizers.base import ImageTokenizer
from ar_image_generation.tokenizers.checkpoint import load_tokenizer_checkpoint
from ar_image_generation.tokenizers.factory import build_tokenizer
from ar_image_generation.utils.device import get_device
from ar_image_generation.utils.image_grid import save_image_grid
from ar_image_generation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained autoregressive approach.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment config YAML."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Approach checkpoint. Defaults to checkpoints/approaches/<approach>/best.pt.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used for loss evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Evaluation output directory.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Number of samples to generate."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Override sampling temperature."
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k sampling.")
    parser.add_argument("--top-p", type=float, default=None, help="Override top-p sampling.")
    parser.add_argument(
        "--max-loss-batches",
        type=int,
        default=None,
        help="Limit number of batches for quick evaluation.",
    )
    return parser.parse_args()


def move_batch_to_device(batch: ImageBatch, device: torch.device) -> ImageBatch:
    return ImageBatch(
        images=batch.images.to(device, non_blocking=True),
        labels=None if batch.labels is None else batch.labels.to(device, non_blocking=True),
        metadata=batch.metadata,
    )


def select_loader(
    *,
    cfg: ExperimentConfig,
    split: str,
) -> torch.utils.data.DataLoader[ImageBatch]:
    train_loader, val_loader, test_loader = build_dataloaders(cfg.dataset)

    if split == "train":
        return train_loader

    if split == "val":
        return val_loader

    if split == "test":
        return test_loader

    raise ValueError(f"Unknown split: {split}")


def make_sampling_config(cfg: ExperimentConfig, args: argparse.Namespace) -> SamplingConfig:
    return SamplingConfig(
        temperature=args.temperature if args.temperature is not None else cfg.sampling.temperature,
        top_k=args.top_k if args.top_k is not None else cfg.sampling.top_k,
        top_p=args.top_p if args.top_p is not None else cfg.sampling.top_p,
        num_samples=args.num_samples if args.num_samples is not None else cfg.sampling.num_samples,
    )


def load_models(
    *,
    cfg: ExperimentConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ImageTokenizer, AutoregressiveApproach, dict]:
    tokenizer = build_tokenizer(
        cfg.tokenizer,
        image_size=cfg.dataset.size,
        image_channels=3,
    ).to(device)

    load_tokenizer_checkpoint(
        path=cfg.tokenizer.checkpoint_path,
        model=tokenizer,
        map_location=device,
    )
    tokenizer.eval()

    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)

    model = build_approach_from_config(
        cfg.approach,
        vocab_size=tokenizer.vocab_size,
        latent_shape=tokenizer.latent_shape,
    ).to(device)

    checkpoint = load_model_checkpoint(
        path=checkpoint_path,
        model=model,
        map_location=device,
    )
    model.eval()

    return tokenizer, model, checkpoint


@torch.no_grad()
def evaluate_loss(
    *,
    model: AutoregressiveApproach,
    tokenizer: ImageTokenizer,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
    max_batches: int | None,
) -> dict[str, float | int]:
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(loader, desc="evaluating loss", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model.training_step(batch, tokenizer)
            loss = output["loss"]

        total_loss += loss.item()
        num_batches += 1

        progress.set_postfix(loss=f"{loss.item():.4f}")

        if max_batches is not None and num_batches >= max_batches:
            break

    return {
        "loss": total_loss / num_batches,
        "num_batches": num_batches,
    }


@torch.no_grad()
def generate_samples(
    *,
    model: AutoregressiveApproach,
    tokenizer: ImageTokenizer,
    device: torch.device,
    sampling_cfg: SamplingConfig,
) -> torch.Tensor:
    return model.generate(
        tokenizer=tokenizer,
        batch_size=sampling_cfg.num_samples,
        labels=None,
        device=device,
        sampling_cfg=sampling_cfg,
    )


def main() -> None:
    args = parse_args()

    cfg = load_experiment_config(args.config)
    seed_everything(cfg.seed)

    device = get_device(cfg.device)
    mixed_precision = cfg.train.mixed_precision and device.type == "cuda"

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path("checkpoints") / "approaches" / cfg.approach.name / "best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Approach checkpoint does not exist: {checkpoint_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("runs") / "eval" / cfg.approach.name

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, checkpoint = load_models(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    loader = select_loader(cfg=cfg, split=args.split)
    sampling_cfg = make_sampling_config(cfg, args)

    loss_metrics = evaluate_loss(
        model=model,
        tokenizer=tokenizer,
        loader=loader,
        device=device,
        mixed_precision=mixed_precision,
        max_batches=args.max_loss_batches,
    )

    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sampling_cfg=sampling_cfg,
    )

    save_image_grid(
        samples.detach().cpu(),
        output_dir / "samples.png",
        nrow=8,
        normalized=True,
    )

    torch.save(samples.detach().cpu(), output_dir / "samples.pt")

    speed = measure_sampling_speed(
        sample_fn=lambda: generate_samples(
            model=model,
            tokenizer=tokenizer,
            device=device,
            sampling_cfg=sampling_cfg,
        ),
        num_samples=sampling_cfg.num_samples,
        device=device,
        warmup_steps=1,
        measured_steps=3,
    )

    metrics = {
        "approach": cfg.approach.name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "loss": loss_metrics,
        "sampling": {
            "num_samples_per_step": sampling_cfg.num_samples,
            "measured_total_samples": speed.num_samples,
            "total_seconds": speed.total_seconds,
            "samples_per_second": speed.samples_per_second,
            "temperature": sampling_cfg.temperature,
            "top_k": sampling_cfg.top_k,
            "top_p": sampling_cfg.top_p,
        },
        "parameters": {
            "trainable": count_parameters(model, trainable_only=True),
            "total": count_parameters(model, trainable_only=False),
        },
        "tokenizer": {
            "checkpoint": str(cfg.tokenizer.checkpoint_path),
            "vocab_size": tokenizer.vocab_size,
            "latent_shape": list(tokenizer.latent_shape),
        },
        "dataset": {
            "name": cfg.dataset.name,
            "size": cfg.dataset.size,
            "split": args.split,
        },
        "device": str(device),
        "mixed_precision": mixed_precision,
    }

    save_json(metrics, output_dir / "metrics.json")

    print("Evaluation complete")
    print("-------------------")
    print(f"approach:      {cfg.approach.name}")
    print(f"checkpoint:    {checkpoint_path}")
    print(f"split:         {args.split}")
    print(f"loss:          {loss_metrics['loss']:.4f}")
    print(f"samples/sec:   {speed.samples_per_second:.2f}")
    print(f"trainable params: {metrics['parameters']['trainable']:,}")
    print(f"output dir:    {output_dir}")


if __name__ == "__main__":
    main()
