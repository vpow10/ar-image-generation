# Autoregressive Image Generation

This project compares autoregressive image generation methods on a small medical image dataset.

Target methods:

1. Piece-by-piece autoregressive generation.
2. Low-resolution to high-resolution autoregressive generation.
3. Random fragment filling / masked-token generation.
4. Custom autoregressive method.

The codebase is designed around a shared pipeline:

```text
image -> tokenizer -> discrete tokens -> generation approach -> generated tokens -> image
```
## Setup
Install dependencies:

```bash
uv sync
```

Run checks:

```bash
uv check
```

## Planned workflow
Train tokenizer:

```bash
uv run python scripts/train_tokenizer.py --config configs/experiment/var_pathmnist64_debug.yaml
```

Train generation approach:

```bash
uv run python scripts/train_approach.py --config configs/experiment/var_pathmnist64_debug.yaml
```

Sample:

```bash
uv run python scripts/sample.py --config configs/experiment/var_pathmnist64_debug.yaml
```

Evaluate:

```bash
uv run python scripts/evaluate.py --config configs/experiment/var_pathmnist64_debug.yaml
```
