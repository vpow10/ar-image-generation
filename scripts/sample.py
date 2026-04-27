import argparse
from pathlib import Path

import torch

from ar_image_generation.approaches.base import SamplingConfig
from ar_image_generation.approaches.factory import build_approach_from_config
from ar_image_generation.config import load_experiment_config
from ar_image_generation.engine.checkpointing import load_model_checkpoint
from ar_image_generation.tokenizers.checkpoint import load_tokenizer_checkpoint
from ar_image_generation.tokenizers.factory import build_tokenizer
from ar_image_generation.utils.device import get_device
from ar_image_generation.utils.image_grid import save_image_grid
from ar_image_generation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a trained approach.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment config YAML."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Approach checkpoint path. Defaults to checkpoints/approaches/<approach>/best.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/samples/generated.png"),
        help="Output image grid path.",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Override number of samples.")
    parser.add_argument(
        "--temperature", type=float, default=None, help="Override sampling temperature."
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k sampling.")
    parser.add_argument("--top-p", type=float, default=None, help="Override top-p sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_experiment_config(args.config)
    seed_everything(cfg.seed)

    device = get_device(cfg.device)

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

    model = build_approach_from_config(
        cfg.approach,
        vocab_size=tokenizer.vocab_size,
        latent_shape=tokenizer.latent_shape,
    ).to(device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path("checkpoints") / "approaches" / cfg.approach.name / "best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Approach checkpoint does not exist: {checkpoint_path}")

    load_model_checkpoint(
        path=checkpoint_path,
        model=model,
        map_location=device,
    )
    model.eval()

    sampling_cfg = SamplingConfig(
        temperature=args.temperature if args.temperature is not None else cfg.sampling.temperature,
        top_k=args.top_k if args.top_k is not None else cfg.sampling.top_k,
        top_p=args.top_p if args.top_p is not None else cfg.sampling.top_p,
        num_samples=args.num_samples if args.num_samples is not None else cfg.sampling.num_samples,
    )

    with torch.no_grad():
        images = model.generate(
            tokenizer=tokenizer,
            batch_size=sampling_cfg.num_samples,
            labels=None,
            device=device,
            sampling_cfg=sampling_cfg,
        )

    save_image_grid(
        images.detach().cpu(),
        args.output,
        nrow=8,
        normalized=True,
    )

    print("Sampling complete")
    print("-----------------")
    print(f"approach:    {cfg.approach.name}")
    print(f"checkpoint:  {checkpoint_path}")
    print(f"output:      {args.output}")
    print(f"samples:     {sampling_cfg.num_samples}")
    print(f"temperature: {sampling_cfg.temperature}")
    print(f"top_k:       {sampling_cfg.top_k}")
    print(f"top_p:       {sampling_cfg.top_p}")


if __name__ == "__main__":
    main()
