from collections.abc import Sequence
from pathlib import Path
from typing import Any

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset

from ar_image_generation.config import DatasetConfig
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.data.transforms import build_image_transform


def _resolve_medmnist_dataset_class(name: str) -> type[Dataset]:
    if name not in INFO:
        available = ", ".join(sorted(INFO))
        raise ValueError(f"Unknown MedMNIST dataset '{name}'. Available datasets: {available}")

    class_name = INFO[name]["python_class"]
    dataset_cls = getattr(medmnist, class_name, None)

    if dataset_cls is None:
        raise ValueError(f"MedMNIST class '{class_name}' could not be found.")

    return dataset_cls


def _collate_image_batch(samples: Sequence[tuple[torch.Tensor, Any]]) -> ImageBatch:
    images = torch.stack([sample[0] for sample in samples], dim=0)

    raw_labels = [sample[1] for sample in samples]
    labels_array = np.asarray(raw_labels)
    labels = torch.as_tensor(labels_array, dtype=torch.long)

    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)

    return ImageBatch(
        images=images,
        labels=labels,
        metadata=None,
    )


def build_medmnist_dataset(
    cfg: DatasetConfig,
    *,
    split: str,
) -> Dataset:
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)

    dataset_cls = _resolve_medmnist_dataset_class(cfg.name)
    transform = build_image_transform(normalize=cfg.normalize)

    return dataset_cls(
        split=split,
        transform=transform,
        download=cfg.download,
        as_rgb=cfg.as_rgb,
        root=str(root),
        size=cfg.size,
    )


def build_medmnist_dataloader(
    cfg: DatasetConfig,
    *,
    split: str,
    shuffle: bool,
) -> DataLoader[ImageBatch]:
    dataset = build_medmnist_dataset(cfg, split=split)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
        collate_fn=_collate_image_batch,
    )


def build_dataloaders(
    cfg: DatasetConfig,
) -> tuple[DataLoader[ImageBatch], DataLoader[ImageBatch], DataLoader[ImageBatch]]:
    train_loader = build_medmnist_dataloader(cfg, split="train", shuffle=True)
    val_loader = build_medmnist_dataloader(cfg, split="val", shuffle=False)
    test_loader = build_medmnist_dataloader(cfg, split="test", shuffle=False)

    return train_loader, val_loader, test_loader
