"""
Evaluate the custom GRAFT-GS approach (Exp 1–3).

Mirrors the CLI interface of evaluate.py so run_pipeline.py can call it the same way.
Handles the GaussianPrimitiveAR model + GaussianSplatTokenizer which are incompatible
with the standard ExperimentConfig / AutoregressiveApproach pipeline.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
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
from ar_image_generation.config import load_yaml
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.data.medmnist import build_dataloaders, build_medmnist_dataloader
from ar_image_generation.engine.checkpointing import load_model_checkpoint
from ar_image_generation.evaluation.io import append_csv_row, save_json
from ar_image_generation.evaluation.metrics import count_parameters, measure_sampling_speed
from ar_image_generation.tokenizers.graft_gs import GaussianSplatTokenizer, GaussianTokenizerConfig
from ar_image_generation.utils.device import get_device
from ar_image_generation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the custom GRAFT-GS approach.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--quality-split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--quality-batch-size", type=int, default=64)
    parser.add_argument("--quality-num-samples", type=int, default=None)
    parser.add_argument("--speed-quality", action="store_true")
    parser.add_argument(
        "--speed-quality-temperatures",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
        metavar="T",
    )
    parser.add_argument("--tokenizer-quality", action="store_true")
    parser.add_argument("--tokenizer-quality-split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--max-loss-batches", type=int, default=None)
    return parser.parse_args()


def build_tokenizer(config: dict[str, Any]) -> GaussianSplatTokenizer:
    t = config["tokenizer"]
    return GaussianSplatTokenizer(
        GaussianTokenizerConfig(
            image_size=t["image_size"],
            image_channels=t["image_channels"],
            num_primitives=t["num_primitives"],
            primitive_feature_dim=t["primitive_feature_dim"],
            hidden_channels=t["hidden_channels"],
            min_scale=t["min_scale"],
            max_scale=t["max_scale"],
            max_position_offset=t["max_position_offset"],
            renderer_chunk_size=t["renderer_chunk_size"],
        )
    )


def build_prior(config: dict[str, Any]) -> GaussianPrimitiveAR:
    p = config["prior"]
    return GaussianPrimitiveAR(
        GaussianPrimitiveARConfig(
            num_primitives=p["num_primitives"],
            primitive_dim=p["primitive_dim"],
            dim=p["dim"],
            depth=p["depth"],
            num_heads=p["num_heads"],
            mlp_ratio=p["mlp_ratio"],
            dropout=p["dropout"],
            class_conditional=p["class_conditional"],
            num_classes=p["num_classes"],
            log_std_min=p["log_std_min"],
            log_std_max=p["log_std_max"],
            mean_loss_weight=p["mean_loss_weight"],
        )
    )


def move_batch(batch: ImageBatch, device: torch.device) -> ImageBatch:
    return ImageBatch(
        images=batch.images.to(device, non_blocking=True),
        labels=None if batch.labels is None else batch.labels.to(device, non_blocking=True),
        metadata=batch.metadata,
    )


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
    geometry_noise_scale: float,
    feature_noise_scale: float,
) -> torch.Tensor:
    normalized = prior.generate(
        batch_size=batch_size,
        labels=labels,
        device=device,
        temperature=temperature,
        geometry_noise_scale=geometry_noise_scale,
        feature_noise_scale=feature_noise_scale,
    ).clamp(-3.0, 3.0)

    flattened = normalizer.denormalize(normalized)
    primitives = primitives_from_flattened(flattened, tokenizer.primitive_cfg, anchors=tokenizer.encoder.anchors)
    return tokenizer.decode(primitives)


@torch.no_grad()
def evaluate_loss(
    *,
    prior: GaussianPrimitiveAR,
    tokenizer: GaussianSplatTokenizer,
    normalizer: PrimitiveNormalizer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mixed_precision: bool,
    max_batches: int | None,
) -> dict[str, float]:
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="evaluating loss", leave=False):
        batch = move_batch(batch, device)

        with autocast(device_type=device.type, enabled=mixed_precision):
            encoded = tokenizer.encode(batch.images)
            flattened = flatten_primitives(encoded)

        prims = normalizer.normalize(flattened)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = prior(prims, batch.labels)

        total_loss += output.loss.item()
        num_batches += 1

        if max_batches is not None and num_batches >= max_batches:
            break

    return {"loss": total_loss / num_batches, "num_batches": num_batches}


@torch.no_grad()
def _feed_quality_metrics(
    *,
    prior: GaussianPrimitiveAR,
    tokenizer: GaussianSplatTokenizer,
    normalizer: PrimitiveNormalizer,
    cfg: dict[str, Any],
    device: torch.device,
    quality_split: str,
    quality_batch_size: int,
    quality_num_samples: int | None,
    temperature: float,
    geometry_noise_scale: float,
    feature_noise_scale: float,
) -> dict[str, float | int]:
    cpu = torch.device("cpu")
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(cpu)
    is_metric = InceptionScore(normalize=False).to(cpu)

    from ar_image_generation.config import DatasetConfig
    dataset_cfg = DatasetConfig(**cfg["dataset"])
    quality_loader = build_medmnist_dataloader(dataset_cfg, split=quality_split, shuffle=False)
    num_real = (
        min(quality_num_samples, len(quality_loader.dataset))
        if quality_num_samples
        else len(quality_loader.dataset)
    )

    real_fed = 0
    for batch in tqdm(quality_loader, desc=f"real ({quality_split}) → FID", leave=False):
        remaining = num_real - real_fed
        if remaining <= 0:
            break
        images = batch.images[:remaining]
        real_u8 = (((images + 1.0) / 2.0).clamp(0.0, 1.0) * 255).to(torch.uint8)
        fid.update(real_u8, real=True)
        real_fed += len(images)

    generated = 0
    with tqdm(total=real_fed, desc="generating → FID/IS", leave=False) as pbar:
        while generated < real_fed:
            batch_size = min(quality_batch_size, real_fed - generated)
            samples = generate_images(
                prior=prior,
                tokenizer=tokenizer,
                normalizer=normalizer,
                batch_size=batch_size,
                labels=None,
                device=device,
                temperature=temperature,
                geometry_noise_scale=geometry_noise_scale,
                feature_noise_scale=feature_noise_scale,
            )
            fake_u8 = (((samples.cpu() + 1.0) / 2.0).clamp(0.0, 1.0) * 255).to(torch.uint8)
            fid.update(fake_u8, real=False)
            is_metric.update(fake_u8)
            generated += batch_size
            pbar.update(batch_size)

    fid_score = fid.compute().item()
    is_mean, is_std = is_metric.compute()
    return {
        "fid": fid_score,
        "inception_score_mean": is_mean.item(),
        "inception_score_std": is_std.item(),
        "num_real": real_fed,
        "num_generated": generated,
    }


@torch.no_grad()
def evaluate_tokenizer_quality(
    *,
    tokenizer: GaussianSplatTokenizer,
    cfg: dict[str, Any],
    device: torch.device,
    split: str,
    csv_path: Path | None,
) -> None:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    from ar_image_generation.config import DatasetConfig
    dataset_cfg = DatasetConfig(**cfg["dataset"])
    loader = build_medmnist_dataloader(dataset_cfg, split=split, shuffle=False)

    print(f"\nExperiment 3: Tokenizer quality  (split={split})")

    for batch in tqdm(loader, desc="tokenizer encode→decode", leave=False):
        images = batch.images.to(device)
        primitives = tokenizer.encode(images)
        reconstructed = tokenizer.decode(primitives)
        real = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
        recon = ((reconstructed + 1.0) / 2.0).clamp(0.0, 1.0)
        psnr_metric.update(recon, real)
        ssim_metric.update(recon, real)

    psnr = psnr_metric.compute().item()
    ssim = ssim_metric.compute().item()

    print(f"  PSNR={psnr:.2f} dB   SSIM={ssim:.4f}")

    row = {
        "approach": "custom",
        "tokenizer_name": cfg["tokenizer"].get("name", "graft_gs"),
        "tokenizer_checkpoint": str(cfg["tokenizer"]["checkpoint_path"]),
        "split": split,
        "psnr_db": round(psnr, 4),
        "ssim": round(ssim, 4),
    }
    if csv_path is not None:
        append_csv_row(row, csv_path)


def main() -> None:
    args = parse_args()

    config = load_yaml(args.config)
    seed_everything(config["seed"])

    device = get_device(config["device"])
    mixed_precision = config.get("train", {}).get("mixed_precision", False) and device.type == "cuda"

    run_name = config["logging"]["run_name"]

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path("runs") / run_name / "checkpoints" / "best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Custom approach checkpoint not found: {checkpoint_path}")

    output_dir = args.output_dir or Path("runs") / "eval" / "custom"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists():
            csv_path.unlink()

    tokenizer = build_tokenizer(config).to(device)
    load_model_checkpoint(
        path=config["tokenizer"]["checkpoint_path"],
        model=tokenizer,
        map_location=device,
    )
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    prior = build_prior(config).to(device)
    checkpoint = load_model_checkpoint(
        path=checkpoint_path,
        model=prior,
        map_location=device,
    )
    prior.eval()

    normalizer = PrimitiveNormalizer.from_state_dict(checkpoint["extra"]["normalizer"])

    sampling_cfg = config.get("sampling", {})
    temperature = sampling_cfg.get("temperature", 0.25)
    geometry_noise_scale = sampling_cfg.get("geometry_noise_scale", 0.7)
    feature_noise_scale = sampling_cfg.get("feature_noise_scale", 0.05)
    num_samples = sampling_cfg.get("num_samples", 64)

    from ar_image_generation.config import DatasetConfig
    dataset_cfg = DatasetConfig(**config["dataset"])
    _, val_loader, test_loader = build_dataloaders(dataset_cfg)
    loader = val_loader if args.split == "val" else test_loader

    loss_metrics = evaluate_loss(
        prior=prior,
        tokenizer=tokenizer,
        normalizer=normalizer,
        loader=loader,
        device=device,
        mixed_precision=mixed_precision,
        max_batches=args.max_loss_batches,
    )

    def _gen() -> torch.Tensor:
        return generate_images(
            prior=prior, tokenizer=tokenizer, normalizer=normalizer,
            batch_size=num_samples, labels=None, device=device,
            temperature=temperature,
            geometry_noise_scale=geometry_noise_scale,
            feature_noise_scale=feature_noise_scale,
        )

    speed = measure_sampling_speed(
        sample_fn=_gen,
        num_samples=num_samples,
        device=device,
        warmup_steps=1,
        measured_steps=3,
    )

    if args.quality:
        quality = _feed_quality_metrics(
            prior=prior, tokenizer=tokenizer, normalizer=normalizer,
            cfg=config, device=device,
            quality_split=args.quality_split,
            quality_batch_size=args.quality_batch_size,
            quality_num_samples=args.quality_num_samples,
            temperature=temperature,
            geometry_noise_scale=geometry_noise_scale,
            feature_noise_scale=feature_noise_scale,
        )
        if args.csv is not None:
            append_csv_row(
                {
                    "approach": "custom",
                    "loss": loss_metrics["loss"],
                    "samples_per_second": round(speed.samples_per_second, 4),
                    "trainable_params": count_parameters(prior, trainable_only=True),
                    "fid": round(quality["fid"], 4),
                    "inception_score_mean": round(quality["inception_score_mean"], 4),
                    "inception_score_std": round(quality["inception_score_std"], 4),
                },
                args.csv,
            )
        print(f"FID={quality['fid']:.4f}  IS={quality['inception_score_mean']:.4f}±{quality['inception_score_std']:.4f}")

    if args.speed_quality:
        print(f"\nExperiment 2: Speed–Quality sweep  (temperatures: {args.speed_quality_temperatures})")
        speed_sq = measure_sampling_speed(
            sample_fn=_gen,
            num_samples=num_samples,
            device=device,
            warmup_steps=1,
            measured_steps=3,
        )
        print(f"  speed: {speed_sq.samples_per_second:.2f} samples/sec")

        for temp in args.speed_quality_temperatures:
            quality = _feed_quality_metrics(
                prior=prior, tokenizer=tokenizer, normalizer=normalizer,
                cfg=config, device=device,
                quality_split=args.quality_split,
                quality_batch_size=args.quality_batch_size,
                quality_num_samples=args.quality_num_samples,
                temperature=temp,
                geometry_noise_scale=geometry_noise_scale,
                feature_noise_scale=feature_noise_scale,
            )
            row = {
                "approach": "custom",
                "temperature": temp,
                "top_k": "",
                "samples_per_second": round(speed_sq.samples_per_second, 4),
                "fid": round(quality["fid"], 4),
                "inception_score_mean": round(quality["inception_score_mean"], 4),
                "inception_score_std": round(quality["inception_score_std"], 4),
                "num_real": quality["num_real"],
                "num_generated": quality["num_generated"],
            }
            if args.csv is not None:
                append_csv_row(row, args.csv)
            print(
                f"  temp={temp:.2f}  "
                f"FID={quality['fid']:.2f}  "
                f"IS={quality['inception_score_mean']:.2f}±{quality['inception_score_std']:.2f}"
            )

    if args.tokenizer_quality:
        evaluate_tokenizer_quality(
            tokenizer=tokenizer,
            cfg=config,
            device=device,
            split=args.tokenizer_quality_split,
            csv_path=args.csv,
        )

    print("Evaluation complete")
    print(f"approach:   custom  ({run_name})")
    print(f"loss:       {loss_metrics['loss']:.4f}")
    print(f"samples/sec: {speed.samples_per_second:.2f}")


if __name__ == "__main__":
    main()
