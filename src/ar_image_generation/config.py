from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "pathmnist"
    root: Path = Path("data/medmnist")
    size: int = 64
    batch_size: int = 32
    num_workers: int = 4
    download: bool = True
    normalize: bool = True
    as_rgb: bool = True


class TokenizerTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = 60
    lr: float = 2e-4
    weight_decay: float = 0.0

    commitment_cost: float = 0.25

    reconstruction_l1_weight: float = 1.0
    reconstruction_mse_weight: float = 0.25


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "vqvae_small"
    checkpoint_path: Path = Path("checkpoints/tokenizer/pathmnist64_vqvae.pt")
    vocab_size: int = 512
    embedding_dim: int = 128
    hidden_channels: int = 128
    downsample_factor: int = 8
    train: TokenizerTrainConfig = Field(default_factory=TokenizerTrainConfig)


class ApproachConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    log_every_steps: int = 50
    sample_every_epochs: int = 5
    save_every_epochs: int = 10


class SamplingConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = 1.0
    top_k: int | None = 100
    top_p: float | None = None
    num_samples: int = 64


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_wandb: bool = False
    project: str = "ar-image-generation"
    run_name: str = "debug"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"

    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    approach: ApproachConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    sampling: SamplingConfigModel = Field(default_factory=SamplingConfigModel)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML object: {path}")

    return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))
