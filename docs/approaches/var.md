# VAR-style Low-to-High Resolution Generation

This document describes the VAR-style approach implemented in this project.

The method is inspired by Visual Autoregressive Modeling (VAR), where image generation is formulated as progressive low-resolution-to-high-resolution token generation instead of standard raster-scan next-token prediction.

In our project, the method is adapted to small medical images from PathMNIST and to limited GPU resources.

---

## High-level idea

Standard image autoregression usually flattens an image token grid and predicts tokens one by one in raster order:

```text
token_0 -> token_1 -> token_2 -> ... -> token_N
```

This works, but the ordering is not very natural for images. Images usually have global structure first and local details later.

The VAR-style idea is different:

```text
coarse image representation -> finer image representation -> final detailed image representation
```

For our current setup:

```text
image resolution:      64 x 64
tokenizer latent grid: 16 x 16
scale schedule:        1x1 -> 2x2 -> 4x4 -> 8x8 -> 16x16
```

The model first generates coarse tokens, then progressively generates finer tokens. Only the final `16x16` token grid is decoded into the generated image.

---

## Full generation pipeline

The project uses a VQ-VAE tokenizer and a transformer-based autoregressive generator.

The full pipeline is:

```text
real image
  -> VQ-VAE encoder
  -> discrete image tokens
  -> VAR-style transformer
  -> generated discrete tokens
  -> VQ-VAE decoder
  -> generated image
```

The tokenizer is trained separately. The VAR model does not generate pixels directly. It generates discrete token IDs from the tokenizer codebook.

Current tokenizer setup:

```text
input image:        [3, 64, 64]
latent token grid:  [16, 16]
vocabulary size:    1024
downsample factor:  4
```

---

## Image-pyramid multiscale targets

An important part of the implementation is how the multiscale token targets are created.

Earlier experimental versions tried to create coarse token grids by downsampling token IDs directly. That was incorrect because token IDs are categorical indices, not continuous values. Token ID `10` is not numerically close to token ID `11`; both are just arbitrary codebook entries.

The final implementation therefore builds multiscale targets from an image pyramid.

For each image:

```text
64x64 image -> resize to 4x4   -> tokenizer.encode -> 1x1 tokens
64x64 image -> resize to 8x8   -> tokenizer.encode -> 2x2 tokens
64x64 image -> resize to 16x16 -> tokenizer.encode -> 4x4 tokens
64x64 image -> resize to 32x32 -> tokenizer.encode -> 8x8 tokens
64x64 image -> original 64x64  -> tokenizer.encode -> 16x16 tokens
```

Since the tokenizer downsampling factor is `4`, each resized image produces the expected token-grid size:

```text
4x4 image   -> 1x1 tokens
8x8 image   -> 2x2 tokens
16x16 image -> 4x4 tokens
32x32 image -> 8x8 tokens
64x64 image -> 16x16 tokens
```

The final multiscale sequence is:

```text
[1x1 tokens] [2x2 tokens] [4x4 tokens] [8x8 tokens] [16x16 tokens]
```

The total sequence length is:

```text
1 + 4 + 16 + 64 + 256 = 341 tokens
```

---

## Autoregressive training objective

The model is trained as a causal transformer over the flattened multiscale sequence.

Given the multiscale target sequence:

```text
[x0, x1, x2, ..., x340]
```

the transformer receives a shifted input:

```text
input:  [BOS, x0, x1, ..., x339]
target: [x0,  x1, x2, ..., x340]
```

The training objective is cross-entropy over the token vocabulary:

```text
loss = cross_entropy(predicted_token_logits, target_token_ids)
```

Because only the final `16x16` grid is decoded into the output image, the implementation gives more importance to the final scale.

Current regularized setup:

```text
final_scale_loss_weight: 2.0
```

This means errors on the final `16x16` tokens are weighted more strongly than errors on the earlier auxiliary coarse scales.

---

## Class conditioning

PathMNIST contains multiple tissue classes. Unconditional generation forced the model to learn one broad distribution covering all visual modes at once. This produced weaker and less coherent samples.

The final implementation therefore supports class-conditional generation.

During training, the model receives the image label and adds a learned class embedding to the token embeddings. Conceptually, the model learns:

```text
p(tokens | class)
```

instead of:

```text
p(tokens)
```

This reduces ambiguity and improves sample quality.

Current setup:

```text
class_conditional: true
num_classes: 9
```

During unconditional sampling from the script, if explicit labels are not provided, the model samples random class labels internally.

---

## Generation procedure

At inference time, there is no input image.

The model starts from a learned `BOS` embedding and generates the entire multiscale sequence token by token:

```text
generate token 0
generate token 1
generate token 2
...
generate token 340
```

These generated tokens correspond to all scales:

```text
1x1 + 2x2 + 4x4 + 8x8 + 16x16
```

Only the final scale is used for decoding:

```text
generated 16x16 tokens -> VQ-VAE decoder -> generated 64x64 image
```

The earlier scales act as coarse conditioning information for the later fine-scale generation.

---

## Current model configuration

The current best-performing VAR-style setup uses:

```yaml
tokenizer:
  vocab_size: 1024
  embedding_dim: 192
  hidden_channels: 192
  downsample_factor: 4

approach:
  name: var
  dim: 384
  depth: 6
  num_heads: 6
  mlp_ratio: 4
  dropout: 0.15
  scales: [1, 2, 4, 8, 16]
  class_conditional: true
  num_classes: 9
  final_scale_loss_weight: 2.0

train:
  lr: 0.0001
  weight_decay: 0.05
  batch_size: 16
```

The model has approximately:

```text
11.57M trainable parameters
```

---

## Implemented files

Main implementation files:

```text
src/ar_image_generation/approaches/var/model.py
src/ar_image_generation/approaches/var/multiscale.py
src/ar_image_generation/approaches/var/schedule.py
src/ar_image_generation/approaches/var/sampler.py
```

Shared components used by the approach:

```text
src/ar_image_generation/models/transformer.py
src/ar_image_generation/tokenizers/base.py
src/ar_image_generation/tokenizers/vqvae.py
src/ar_image_generation/engine/checkpointing.py
scripts/train_approach.py
scripts/sample.py
scripts/evaluate.py
```

---

## Differences from the original VAR paper

The implementation in this project is VAR-inspired, but it is not a full reproduction of the original large-scale VAR method.

The main differences are listed below.

---

### 1. Dataset scale

Original VAR is designed for large-scale natural image generation, especially ImageNet-scale experiments.

Our implementation is designed for:

```text
PathMNIST
64x64 medical images
small GPU budget
university project setting
```

Reason:

```text
The project requires a method that can be trained and tested on limited hardware, specifically an 8 GB VRAM GPU.
```

---

### 2. Tokenizer implementation

Original VAR uses a stronger visual tokenizer suitable for large-scale image generation.

Our implementation uses a custom VQ-VAE tokenizer trained directly on PathMNIST.

Current tokenizer:

```text
64x64 image -> 16x16 token grid
vocab size: 1024
```

Reason:

```text
A project-local tokenizer keeps the pipeline simple, reproducible, and fully controllable.
It also makes the method comparable with the other baseline approaches in the project.
```

---

### 3. Multiscale construction

Original VAR uses a carefully designed next-scale prediction formulation with scale-wise image token maps.

Our implementation creates multiscale targets using an image pyramid:

```text
resize image -> encode resized image -> obtain token grid for that scale
```

Reason:

```text
Directly downsampling token IDs is not meaningful because token IDs are categorical.
Using resized images produces meaningful low-resolution targets while keeping the implementation simple.
```

This was an important correction during development. Earlier versions that downsampled token IDs produced poor results.

---

### 4. Causal flattened sequence instead of full scale-wise parallel prediction

Original VAR emphasizes next-scale prediction, where the model predicts the next resolution level based on previous resolutions.

Our implementation flattens all scales into one causal sequence:

```text
[1x1 tokens] [2x2 tokens] [4x4 tokens] [8x8 tokens] [16x16 tokens]
```

and trains a causal transformer over this sequence.

Reason:

```text
This is simpler to implement and easier to integrate into the shared project pipeline.
It also allows tokens inside the final scale to depend on earlier final-scale tokens, which improved coherence in our experiments.
```

---

### 5. Class conditioning added for PathMNIST

Original VAR can be class-conditional in large-scale class-labeled datasets.

Our implementation uses class conditioning because PathMNIST has multiple tissue classes and unconditional generation gave unstable samples.

Reason:

```text
PathMNIST is visually multimodal.
Class conditioning reduces ambiguity and improves sample quality.
```

---

### 6. Final-scale loss weighting

Original VAR focuses on multiscale image generation with its own training formulation.

Our implementation adds explicit final-scale loss weighting:

```text
final_scale_loss_weight: 2.0
```

Reason:

```text
Only the final 16x16 token grid is decoded into the generated image.
Therefore, errors on final-scale tokens are more important than errors on auxiliary coarse-scale tokens.
```

Earlier experiments with too much final-scale weighting overfit quickly, so the final value is a compromise between image quality and generalization.

---

### 7. Smaller transformer architecture

Original VAR uses much larger models.

Our implementation uses a compact transformer:

```text
dim: 384
depth: 6
num_heads: 6
```

Reason:

```text
The model must fit into limited GPU memory and train in a reasonable amount of time.
```

---

### 8. Simpler sampling strategy

Original VAR uses a more mature sampling setup.

Our implementation uses standard token sampling controls:

```text
temperature
top_k
top_p
```

In practice, the best results came from moderate sampling, for example:

```text
temperature: 0.85
top_k: 100
```

Very strict sampling produced less diverse and sometimes worse outputs.

Reason:

```text
The model is relatively small, and overly strict sampling tends to collapse diversity.
```

---

## Summary

The implemented approach can be described as:

```text
a class-conditional, image-pyramid, multiscale causal autoregressive transformer over VQ-VAE image tokens
```

The method follows the core VAR intuition:

```text
generate images from coarse to fine resolution
```

but adapts it to the project constraints:

```text
small medical images
custom tokenizer
limited GPU memory
shared training/evaluation pipeline
```

The final implementation improved substantially over earlier versions after introducing:

```text
1. a stronger 16x16 VQ-VAE tokenizer,
2. image-pyramid multiscale targets,
3. causal multiscale token generation,
4. class conditioning,
5. final-scale loss weighting,
6. stronger regularization.
```