import argparse
from pathlib import Path

import torch
from torch.amp import autocast
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

from ar_image_generation.approaches.base import AutoregressiveApproach, SamplingConfig
from ar_image_generation.approaches.factory import build_approach_from_config
from ar_image_generation.config import ExperimentConfig, load_experiment_config
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.data.medmnist import build_dataloaders, build_medmnist_dataloader
from ar_image_generation.engine.checkpointing import load_model_checkpoint
from ar_image_generation.evaluation.io import append_csv_row, save_json
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
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Compute FID and Inception Score against the val split (experiment 1).",
    )
    parser.add_argument(
        "--quality-split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split used as real images for FID/IS. Default: val (10,004 images).",
    )
    parser.add_argument(
        "--quality-batch-size",
        type=int,
        default=64,
        help="Batch size for generation during quality evaluation.",
    )
    parser.add_argument(
        "--quality-num-samples",
        type=int,
        default=None,
        help="Limit real+generated images for FID/IS (default: full split). "
             "Use e.g. 500 for a quick smoke test.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV file to append one result row to (shared across all approaches for one experiment).",
    )
    parser.add_argument(
        "--speed-quality",
        action="store_true",
        help="Experiment 2: sweep temperatures and record FID/IS + samples/sec at each "
             "to produce a speed-quality trade-off curve.",
    )
    parser.add_argument(
        "--speed-quality-temperatures",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
        metavar="T",
        help="Temperature values for the --speed-quality sweep (default: 0.5 0.75 1.0 1.25 1.5).",
    )
    parser.add_argument(
        "--tokenizer-quality",
        action="store_true",
        help="Experiment 3: encode→decode reconstruction quality (PSNR, SSIM). "
             "Independent of the generation approach — isolates tokenizer error.",
    )
    parser.add_argument(
        "--tokenizer-quality-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split for tokenizer reconstruction evaluation (default: test).",
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


@torch.no_grad()
def evaluate_quality(
    *,
    model: AutoregressiveApproach,
    tokenizer: ImageTokenizer,
    cfg: ExperimentConfig,
    device: torch.device,
    quality_split: str,
    quality_batch_size: int,
    quality_num_samples: int | None,
    sampling_cfg: SamplingConfig,
) -> dict[str, float | int]:
    """Experiment 1: FID + IS against the full quality_split (default val, 10,004 images)"""

    # FID/IS use float64 internally —> always run on CPU
    cpu = torch.device("cpu")
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(cpu)
    is_metric = InceptionScore(normalize=False).to(cpu)

    quality_loader = build_medmnist_dataloader(cfg.dataset, split=quality_split, shuffle=False)
    num_real = min(quality_num_samples, len(quality_loader.dataset)) if quality_num_samples else len(quality_loader.dataset)

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
    batch_cfg = SamplingConfig(
        temperature=sampling_cfg.temperature,
        top_k=sampling_cfg.top_k,
        top_p=sampling_cfg.top_p,
        num_samples=quality_batch_size,
    )
    with tqdm(total=real_fed, desc="generating → FID/IS", leave=False) as pbar:
        while generated < real_fed:
            batch_cfg.num_samples = min(quality_batch_size, real_fed - generated)
            samples = generate_samples(
                model=model,
                tokenizer=tokenizer,
                device=device,
                sampling_cfg=batch_cfg,
            )
            fake_u8 = (((samples.cpu() + 1.0) / 2.0).clamp(0.0, 1.0) * 255).to(torch.uint8)
            fid.update(fake_u8, real=False)
            is_metric.update(fake_u8)
            generated += batch_cfg.num_samples
            pbar.update(batch_cfg.num_samples)

    fid_score = fid.compute().item()
    is_mean, is_std = is_metric.compute()

    return {
        "fid": fid_score,
        "inception_score_mean": is_mean.item(),
        "inception_score_std": is_std.item(),
        "num_real": real_fed,
        "num_generated": generated,
        "quality_split": quality_split,
    }


@torch.no_grad()
def evaluate_speed_quality(
    *,
    model: AutoregressiveApproach,
    tokenizer: ImageTokenizer,
    cfg: ExperimentConfig,
    device: torch.device,
    temperatures: list[float],
    quality_split: str,
    quality_batch_size: int,
    quality_num_samples: int | None,
    base_sampling_cfg: SamplingConfig,
    csv_path: Path | None,
) -> None:
    """Experiment 2: speed measured once, then FID/IS swept over temperatures."""

    print(f"\nExperiment 2: Speed–Quality sweep  (temperatures: {temperatures})")

    speed = measure_sampling_speed(
        sample_fn=lambda: generate_samples(
            model=model,
            tokenizer=tokenizer,
            device=device,
            sampling_cfg=base_sampling_cfg,
        ),
        num_samples=base_sampling_cfg.num_samples,
        device=device,
        warmup_steps=1,
        measured_steps=3,
    )
    print(f"  speed: {speed.samples_per_second:.2f} samples/sec")

    for temperature in temperatures:
        sweep_cfg = SamplingConfig(
            temperature=temperature,
            top_k=base_sampling_cfg.top_k,
            top_p=base_sampling_cfg.top_p,
            num_samples=base_sampling_cfg.num_samples,
        )

        quality = evaluate_quality(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
            quality_split=quality_split,
            quality_batch_size=quality_batch_size,
            quality_num_samples=quality_num_samples,
            sampling_cfg=sweep_cfg,
        )

        row: dict = {
            "approach": cfg.approach.name,
            "temperature": temperature,
            "top_k": sweep_cfg.top_k if sweep_cfg.top_k is not None else "",
            "samples_per_second": round(speed.samples_per_second, 4),
            "fid": round(quality["fid"], 4),
            "inception_score_mean": round(quality["inception_score_mean"], 4),
            "inception_score_std": round(quality["inception_score_std"], 4),
            "num_real": quality["num_real"],
            "num_generated": quality["num_generated"],
        }

        if csv_path is not None:
            append_csv_row(row, csv_path)

        print(
            f"  temp={temperature:.2f}  "
            f"FID={quality['fid']:.2f}  "
            f"IS={quality['inception_score_mean']:.2f}±{quality['inception_score_std']:.2f}"
        )


@torch.no_grad()
def evaluate_tokenizer_quality(
    *,
    tokenizer: ImageTokenizer,
    cfg: ExperimentConfig,
    device: torch.device,
    split: str,
    csv_path: Path | None,
) -> dict:
    """Experiment 3: encode→decode reconstruction quality (PSNR, SSIM).

    Independent of the generation approach — measures only information lost
    in the tokenization step, before any generation error is introduced.
    Images are normalized to [0, 1] before metric computation.
    """
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    loader = build_medmnist_dataloader(cfg.dataset, split=split, shuffle=False)

    print(f"\nExperiment 3: Tokenizer quality  (split={split})")

    for batch in tqdm(loader, desc="tokenizer encode→decode", leave=False):
        images = batch.images.to(device)

        tokens = tokenizer.encode(images)
        reconstructed = tokenizer.decode(tokens)

        real = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
        recon = ((reconstructed + 1.0) / 2.0).clamp(0.0, 1.0)

        psnr_metric.update(recon, real)
        ssim_metric.update(recon, real)

    psnr = psnr_metric.compute().item()
    ssim = ssim_metric.compute().item()

    row = {
        "approach": cfg.approach.name,
        "tokenizer_name": cfg.tokenizer.name,
        "tokenizer_checkpoint": str(cfg.tokenizer.checkpoint_path),
        "split": split,
        "psnr_db": round(psnr, 4),
        "ssim": round(ssim, 4),
    }

    if csv_path is not None:
        append_csv_row(row, csv_path)

    print(f"  PSNR={psnr:.2f} dB   SSIM={ssim:.4f}")

    return row


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

    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists():
            csv_path.unlink()

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

    quality_metrics: dict | None = None
    if args.quality:
        quality_metrics = evaluate_quality(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
            quality_split=args.quality_split,
            quality_batch_size=args.quality_batch_size,
            quality_num_samples=args.quality_num_samples,
            sampling_cfg=sampling_cfg,
        )

    if args.speed_quality:
        evaluate_speed_quality(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
            temperatures=args.speed_quality_temperatures,
            quality_split=args.quality_split,
            quality_batch_size=args.quality_batch_size,
            quality_num_samples=args.quality_num_samples,
            base_sampling_cfg=sampling_cfg,
            csv_path=args.csv,
        )

    if args.tokenizer_quality:
        evaluate_tokenizer_quality(
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
            split=args.tokenizer_quality_split,
            csv_path=args.csv,
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

    if quality_metrics is not None:
        metrics["quality"] = quality_metrics

    save_json(metrics, output_dir / "metrics.json")

    if args.csv is not None and args.quality:
        csv_row = {
            "approach": cfg.approach.name,
            "loss": loss_metrics["loss"],
            "samples_per_second": speed.samples_per_second,
            "trainable_params": metrics["parameters"]["trainable"],
            "fid": quality_metrics["fid"] if quality_metrics else "",
            "inception_score_mean": quality_metrics["inception_score_mean"] if quality_metrics else "",
            "inception_score_std": quality_metrics["inception_score_std"] if quality_metrics else "",
        }
        append_csv_row(csv_row, args.csv)

    print("Evaluation complete")
    print("-------------------")
    print(f"approach:      {cfg.approach.name}")
    print(f"checkpoint:    {checkpoint_path}")
    print(f"split:         {args.split}")
    print(f"loss:          {loss_metrics['loss']:.4f}")
    print(f"samples/sec:   {speed.samples_per_second:.2f}")
    print(f"trainable params: {metrics['parameters']['trainable']:,}")
    if quality_metrics is not None:
        print(f"FID:           {quality_metrics['fid']:.4f}  (lower is better)")
        print(f"IS:            {quality_metrics['inception_score_mean']:.4f} ± {quality_metrics['inception_score_std']:.4f}  (higher is better)")
    print(f"output dir:    {output_dir}")


if __name__ == "__main__":
    main()
