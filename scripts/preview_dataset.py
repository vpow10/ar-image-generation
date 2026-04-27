import argparse
from pathlib import Path

import torch

from ar_image_generation.config import load_experiment_config
from ar_image_generation.data.medmnist import build_dataloaders
from ar_image_generation.utils.image_grid import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a MedMNIST dataset batch.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/debug/dataset_preview.png"),
        help="Where to save the preview image grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)

    train_loader, val_loader, test_loader = build_dataloaders(cfg.dataset)
    batch = next(iter(train_loader))

    images = batch.images
    labels = batch.labels

    print("Dataset preview")
    print("---------------")
    print(f"dataset:      {cfg.dataset.name}")
    print(f"size:         {cfg.dataset.size}")
    print(f"batch shape:  {tuple(images.shape)}")
    print(f"dtype:        {images.dtype}")
    print(f"min value:    {images.min().item():.4f}")
    print(f"max value:    {images.max().item():.4f}")
    print(f"labels shape: {None if labels is None else tuple(labels.shape)}")
    print(f"train batches:{len(train_loader)}")
    print(f"val batches:  {len(val_loader)}")
    print(f"test batches: {len(test_loader)}")

    save_image_grid(
        images[:64].detach().cpu(), args.output, nrow=8, normalized=cfg.dataset.normalize
    )
    print(f"saved grid:   {args.output}")

    if torch.cuda.is_available():
        print(f"cuda:         {torch.cuda.get_device_name(0)}")
    else:
        print("cuda:         unavailable")


if __name__ == "__main__":
    main()
