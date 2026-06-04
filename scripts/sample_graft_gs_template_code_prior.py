import argparse
from pathlib import Path

import torch

from ar_image_generation.approaches.custom.primitives import GaussianPrimitives
from ar_image_generation.approaches.custom.residual_quantization import (
    ResidualFeatureCodebook,
)
from ar_image_generation.approaches.custom.template_code_model import (
    ResidualTemplateCodeAR,
    ResidualTemplateCodeARConfig,
)
from ar_image_generation.config import load_yaml
from ar_image_generation.engine.checkpointing import load_model_checkpoint
from ar_image_generation.tokenizers.graft_gs import (
    GaussianSplatTokenizer,
    GaussianTokenizerConfig,
)
from ar_image_generation.utils.device import get_device
from ar_image_generation.utils.image_grid import save_image_grid
from ar_image_generation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from GRAFT-GS template-code prior."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--code-temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--template-id", type=int, default=None)
    return parser.parse_args()


def build_tokenizer(config: dict) -> GaussianSplatTokenizer:
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


def build_prior(config: dict) -> ResidualTemplateCodeAR:
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


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed_everything(config["seed"])

    device = get_device(config["device"])

    tokenizer = build_tokenizer(config).to(device)
    load_model_checkpoint(
        path=config["tokenizer"]["checkpoint_path"],
        model=tokenizer,
        map_location=device,
    )
    tokenizer.eval()

    prior = build_prior(config).to(device)
    checkpoint = load_model_checkpoint(
        path=args.checkpoint,
        model=prior,
        map_location=device,
    )
    prior.eval()

    codebook_state = torch.load(config["residual_codebook"]["path"], map_location="cpu")
    codebook = ResidualFeatureCodebook.from_state_dict(codebook_state)

    geometry_templates = checkpoint["extra"]["geometry_templates"][
        "geometry_templates"
    ].to(device)

    num_samples = args.num_samples or config["sampling"]["num_samples"]
    code_temperature = (
        args.code_temperature
        if args.code_temperature is not None
        else config["sampling"]["code_temperature"]
    )
    top_k = args.top_k if args.top_k is not None else config["sampling"]["top_k"]

    labels = None
    if args.label is not None:
        labels = torch.full((num_samples,), args.label, dtype=torch.long, device=device)

    template_ids = None
    if args.template_id is not None:
        template_ids = torch.full(
            (num_samples,),
            args.template_id,
            dtype=torch.long,
            device=device,
        )

    codes, generated_labels, generated_template_ids = prior.generate(
        batch_size=num_samples,
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
            size=(num_samples,),
            device=device,
            dtype=torch.long,
        )

    geometry = geometry_templates[generated_labels, generated_template_ids]
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

    images = tokenizer.decode(primitives)

    save_image_grid(images.detach().cpu(), args.output, nrow=8, normalized=True)

    print("GRAFT-GS template-code sampling complete")
    print("----------------------------------------")
    print(f"checkpoint:       {args.checkpoint}")
    print(f"output:           {args.output}")
    print(f"samples:          {num_samples}")
    print(f"code temperature: {code_temperature}")
    print(f"top_k:            {top_k}")
    print(f"label:            {args.label}")
    print(f"template id:      {args.template_id}")


if __name__ == "__main__":
    main()
