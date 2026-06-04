import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from ar_image_generation.approaches.custom.primitives import (
    GaussianPrimitives,
    flatten_primitives,
)
from ar_image_generation.approaches.custom.residual_quantization import (
    ResidualFeatureCodebook,
)
from ar_image_generation.approaches.custom.template_code_model import (
    ResidualTemplateCodeAR,
    ResidualTemplateCodeARConfig,
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
    parser = argparse.ArgumentParser(
        description="Train GRAFT-GS template-conditioned code prior."
    )
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


def build_prior(config: dict[str, Any]) -> ResidualTemplateCodeAR:
    prior_cfg = config["prior"]

    return ResidualTemplateCodeAR(
        ResidualTemplateCodeARConfig(
            num_primitives=prior_cfg["num_primitives"],
            num_quantizers=prior_cfg["num_quantizers"],
            num_feature_codes=prior_cfg["num_feature_codes"],
            num_templates_per_class=prior_cfg["num_templates_per_class"],
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


def fit_small_kmeans(
    samples: torch.Tensor,
    *,
    num_clusters: int,
    num_iters: int = 30,
) -> torch.Tensor:
    if samples.ndim != 2:
        raise ValueError(f"Expected samples [N, D], got {tuple(samples.shape)}")

    if samples.shape[0] == 0:
        raise ValueError("Cannot fit k-means on empty samples.")

    if samples.shape[0] < num_clusters:
        repeat_count = num_clusters - samples.shape[0]
        repeated = samples[:1].expand(repeat_count, -1)
        return torch.cat([samples, repeated], dim=0)

    indices = torch.randperm(samples.shape[0], device=samples.device)[:num_clusters]
    centers = samples[indices].clone()

    for _ in range(num_iters):
        distances = torch.cdist(samples, centers)
        assignments = distances.argmin(dim=1)

        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(num_clusters, device=samples.device, dtype=samples.dtype)

        new_centers.index_add_(0, assignments, samples)
        counts.index_add_(
            0,
            assignments,
            torch.ones(
                assignments.shape[0], device=samples.device, dtype=samples.dtype
            ),
        )

        empty = counts == 0
        if empty.any():
            refill = torch.randperm(samples.shape[0], device=samples.device)[
                : int(empty.sum().item())
            ]
            new_centers[empty] = samples[refill]
            counts[empty] = 1.0

        centers = new_centers / counts[:, None]

    return centers


@torch.no_grad()
def collect_geometry_and_labels(
    *,
    tokenizer: GaussianSplatTokenizer,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
    num_classes: int,
) -> tuple[torch.Tensor, torch.LongTensor]:
    all_geometry: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="collecting geometry", leave=False):
        batch = move_batch_to_device(batch, device)

        if batch.labels is None:
            raise ValueError("Class labels are required for geometry templates.")

        labels = prepare_labels(batch.labels, num_classes)

        with autocast(device_type=device.type, enabled=mixed_precision):
            primitives = tokenizer.encode(batch.images)
            flattened = flatten_primitives(primitives)

        geometry = flattened[..., :6].detach().float().cpu()
        all_geometry.append(geometry)
        all_labels.append(labels.detach().cpu())

    return torch.cat(all_geometry, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def compute_geometry_templates(
    *,
    tokenizer: GaussianSplatTokenizer,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
    num_classes: int,
    templates_per_class: int,
) -> torch.Tensor:
    geometry, labels = collect_geometry_and_labels(
        tokenizer=tokenizer,
        loader=loader,
        device=device,
        mixed_precision=mixed_precision,
        num_classes=num_classes,
    )

    flat_geometry = geometry.reshape(geometry.shape[0], -1).to(device)
    labels = labels.to(device)

    templates = torch.empty(
        num_classes,
        templates_per_class,
        tokenizer.cfg.num_primitives,
        6,
        device=device,
        dtype=torch.float32,
    )

    global_centers = fit_small_kmeans(
        flat_geometry,
        num_clusters=templates_per_class,
        num_iters=30,
    )

    for class_index in range(num_classes):
        class_mask = labels == class_index

        if class_mask.any():
            class_samples = flat_geometry[class_mask]
            centers = fit_small_kmeans(
                class_samples,
                num_clusters=templates_per_class,
                num_iters=30,
            )
        else:
            centers = global_centers

        templates[class_index] = centers.reshape(
            templates_per_class,
            tokenizer.cfg.num_primitives,
            6,
        )

    return templates.detach().cpu()


@torch.no_grad()
def assign_template_ids(
    *,
    geometry: torch.Tensor,
    labels: torch.LongTensor,
    geometry_templates: torch.Tensor,
) -> torch.LongTensor:
    labels = labels.to(device=geometry.device, dtype=torch.long)
    templates = geometry_templates.to(device=geometry.device, dtype=geometry.dtype)

    batch_templates = templates[labels]
    distances = (geometry[:, None, :, :] - batch_templates).square().mean(dim=(2, 3))

    return distances.argmin(dim=1)


@torch.no_grad()
def encode_targets(
    *,
    tokenizer: GaussianSplatTokenizer,
    codebook: ResidualFeatureCodebook,
    geometry_templates: torch.Tensor,
    images: torch.Tensor,
    labels: torch.LongTensor,
    mixed_precision: bool,
    device: torch.device,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    with autocast(device_type=device.type, enabled=mixed_precision):
        primitives = tokenizer.encode(images)
        flattened = flatten_primitives(primitives)

    geometry = flattened[..., :6]
    features = flattened[..., 6:]

    template_ids = assign_template_ids(
        geometry=geometry,
        labels=labels,
        geometry_templates=geometry_templates,
    )

    codes = codebook.assign(features)

    return codes, template_ids


def train_one_epoch(
    *,
    model: ResidualTemplateCodeAR,
    tokenizer: GaussianSplatTokenizer,
    codebook: ResidualFeatureCodebook,
    geometry_templates: torch.Tensor,
    loader: torch.utils.data.DataLoader[ImageBatch],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    mixed_precision: bool,
    grad_clip_norm: float,
    epoch: int,
    num_classes: int,
) -> dict[str, float]:
    model.train()
    tokenizer.eval()

    total_loss = 0.0
    total_code_loss = 0.0
    num_batches = 0

    progress = tqdm(
        loader, desc=f"train template-code prior epoch {epoch}", leave=False
    )

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        if batch.labels is None:
            raise ValueError("Labels are required for template-code training.")

        labels = prepare_labels(batch.labels, num_classes)

        with torch.no_grad():
            codes, template_ids = encode_targets(
                tokenizer=tokenizer,
                codebook=codebook,
                geometry_templates=geometry_templates,
                images=batch.images,
                labels=labels,
                mixed_precision=mixed_precision,
                device=device,
            )

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(
                codes,
                labels=labels,
                template_ids=template_ids,
            )
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
    model: ResidualTemplateCodeAR,
    tokenizer: GaussianSplatTokenizer,
    codebook: ResidualFeatureCodebook,
    geometry_templates: torch.Tensor,
    loader: torch.utils.data.DataLoader[ImageBatch],
    device: torch.device,
    mixed_precision: bool,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    tokenizer.eval()

    total_loss = 0.0
    total_code_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if batch.labels is None:
            raise ValueError("Labels are required for template-code evaluation.")

        labels = prepare_labels(batch.labels, num_classes)

        codes, template_ids = encode_targets(
            tokenizer=tokenizer,
            codebook=codebook,
            geometry_templates=geometry_templates,
            images=batch.images,
            labels=labels,
            mixed_precision=mixed_precision,
            device=device,
        )

        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(
                codes,
                labels=labels,
                template_ids=template_ids,
            )

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
    prior: ResidualTemplateCodeAR,
    tokenizer: GaussianSplatTokenizer,
    codebook: ResidualFeatureCodebook,
    geometry_templates: torch.Tensor,
    batch_size: int,
    labels: torch.Tensor | None,
    template_ids: torch.Tensor | None,
    device: torch.device,
    code_temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    codes, generated_labels, generated_template_ids = prior.generate(
        batch_size=batch_size,
        labels=labels,
        template_ids=template_ids,
        device=device,
        code_temperature=code_temperature,
        top_k=top_k,
    )

    if generated_labels is None:
        generated_labels = torch.randint(
            low=0,
            high=geometry_templates.shape[0],
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    geometry = geometry_templates.to(device=device, dtype=torch.float32)[
        generated_labels,
        generated_template_ids,
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
    prior: ResidualTemplateCodeAR,
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
        template_ids=None,
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
        templates_per_class=config["geometry_templates"]["templates_per_class"],
    )

    torch.save(
        {"geometry_templates": geometry_templates},
        run_dir / "geometry_templates.pt",
    )

    print("GRAFT-GS template-code prior training")
    print("-------------------------------------")
    print(f"dataset:             {dataset_cfg.name}")
    print(f"image size:          {dataset_cfg.size}")
    print(f"batch size:          {dataset_cfg.batch_size}")
    print(f"primitives:          {config['prior']['num_primitives']}")
    print(f"quantizers:          {config['prior']['num_quantizers']}")
    print(f"feature codes:       {config['prior']['num_feature_codes']}")
    print(f"templates per class: {config['prior']['num_templates_per_class']}")
    print(f"model dim:           {config['prior']['dim']}")
    print(f"depth:               {config['prior']['depth']}")
    print(f"device:              {device}")
    print(f"mixed precision:     {mixed_precision}")
    print(f"epochs:              {epochs}")
    print(f"run dir:             {run_dir}")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=prior,
            tokenizer=tokenizer,
            codebook=codebook,
            geometry_templates=geometry_templates,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixed_precision=mixed_precision,
            grad_clip_norm=config["train"]["grad_clip_norm"],
            epoch=epoch,
            num_classes=config["prior"]["num_classes"],
        )

        val_metrics = evaluate(
            model=prior,
            tokenizer=tokenizer,
            codebook=codebook,
            geometry_templates=geometry_templates,
            loader=val_loader,
            device=device,
            mixed_precision=mixed_precision,
            num_classes=config["prior"]["num_classes"],
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
