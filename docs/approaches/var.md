# VAR-style Low-to-High Resolution Generation

This method generates image tokens progressively from coarse to fine resolution.

For PathMNIST 64x64 and a tokenizer downsampling factor of 8:

```text
image resolution: 64 x 64
latent resolution: 8 x 8
scale schedule: 1x1 -> 2x2 -> 4x4 -> 8x8
```

## Project implementation
The project uses a simplified VAR-style setup:

1. A VQ-VAE converts images into discrete 8x8 latent token grids.
2. The 8x8 token grid is downsampled with nearest-neighbor sampling to create coarse token grids.
3. The transformer predicts each next scale conditioned on previous scales.

The flattened sequence is:
```text
[BOS] [1x1 tokens] [2x2 tokens] [4x4 tokens] [8x8 tokens]
```
The attention rule is:
```text
tokens at scale k may attend to BOS and all previous scales, but not to the same scale or future scales
```
This avoids direct same-scale target leakage during training.

## Files
```text
src/ar_image_generation/approaches/var/schedule.py
src/ar_image_generation/approaches/var/multiscale.py
```

## Model implementation

The current implementation uses a simplified educational VAR objective.

For each scale:

```text
context = BOS + lower-resolution token grids
target = current-resolution token grid
```
The current scale is represented with learned query embeddings rather than ground-truth target token embeddings. This avoids target leakage through transformer residual connections.

Training objective:
```text
cross-entropy over all multiscale target tokens
```
Generation:
```text
1x1 tokens -> 2x2 tokens -> 4x4 tokens -> 8x8 tokens -> VQ-VAE decode
```
Implemented files:
```text
src/ar_image_generation/approaches/var/model.py
src/ar_image_generation/approaches/var/sampler.py
```
