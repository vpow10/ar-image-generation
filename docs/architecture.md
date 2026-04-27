# Architecture

The project uses a shared pipeline for all autoregressive image generation methods.

```text
images
  -> ImageTokenizer.encode
  -> discrete latent tokens
  -> AutoregressiveApproach
  -> generated discrete latent tokens
  -> ImageTokenizer.decode
  -> generated images
```
## Main abstractions
### ImageBatch

A batch of images returned by every dataset loader.

### ImageTokenizer

Encodes images into discrete latent tokens and decodes tokens back into images.

### AutoregressiveApproach

A generation method. Each approach defines its own training objective and sampling procedure but uses the same tokenizer, trainer, logger, and evaluation pipeline.

### Current target dataset

PathMNIST 64x64.

### Current target method

VAR-style low-resolution to high-resolution autoregressive generation.