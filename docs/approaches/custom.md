# Custom Approach: GRAFT-GS

This document describes the planned custom image generation approach for the project.

The method is called:

```text
GRAFT-GS
Gaussian Rendering Autoregressive Flow Transformer
```

---

## High-level idea

Most baseline methods in this project generate images through discrete image-token grids.

For example:

```text
image
  -> VQ-VAE tokenizer
  -> discrete token grid
  -> autoregressive transformer
  -> generated token grid
  -> VQ-VAE decoder
  -> generated image
```

GRAFT-GS proposes a different representation.

Instead of generating a regular grid of tokens, the model generates a sequence of parametric 2D Gaussian primitives.

Each primitive describes a local visual element using parameters such as:

```text
position
scale
rotation
opacity
feature vector or feature code
level of detail
```

The generated primitives are then rendered with differentiable Gaussian splatting into a feature map. A decoder converts this feature map into the final RGB image.

The high-level pipeline is:

```text
image
  -> Gaussian Splat Tokenizer
  -> sequence of 2D Gaussian primitives
  -> autoregressive primitive generator
  -> generated 2D Gaussian primitives
  -> differentiable splatting renderer
  -> decoder
  -> generated image
```

---

## Motivation

Grid-token methods force every image to use approximately the same number of tokens.

For example, with a `16x16` VQ-VAE latent grid:

```text
every image = 256 tokens
```

This is simple, but not always efficient.

A mostly empty image and a highly detailed image still receive the same number of tokens. GRAFT-GS tries to make the representation adaptive.

The intended idea is:

```text
simple image regions  -> fewer primitives
complex image regions -> more primitives
```

This gives the method a natural quality-cost tradeoff.

Instead of asking:

```text
What token should be placed at this grid position?
```

GRAFT-GS asks:

```text
What visual primitive should be placed here, with what position, scale, orientation, opacity, and feature?
```

---

## Core representation

The basic generated unit is a 2D Gaussian primitive.

A primitive can be written conceptually as:

```text
z_i = (level_i, position_i, covariance_i, opacity_i, feature_i)
```

More explicitly:

```text
level_i       -> level of detail / hierarchy level
position_i    -> (x, y) position
scale_i       -> horizontal and vertical scale
rotation_i    -> orientation
opacity_i     -> alpha / transparency
feature_i     -> learned visual feature or codebook index
```

The primitive is not directly an RGB patch. It is a parametric element that contributes to a rendered feature map.

A set of generated primitives is rendered into an intermediate feature representation:

```text
{z_1, z_2, ..., z_N}
  -> Gaussian splatting renderer
  -> feature map
```

Then a CNN decoder converts the feature map into the final image:

```text
feature map -> decoder -> RGB image
```

---

## Why Gaussian primitives?

Gaussian primitives are spatially adaptive.

A large smooth region may be represented by a small number of large Gaussians. A complex region may require many smaller Gaussians.

This is different from grid-based tokenizers, where token positions are fixed.

Potential advantages:

```text
adaptive number of primitives
explicit spatial structure
controllable quality-cost tradeoff
natural support for level-of-detail generation
possible interpretability through primitive positions and scales
```

Potential disadvantages:

```text
harder tokenizer
continuous parameter modeling
more complex renderer
variable-length sequences
more difficult training
```

---

## Planned architecture

The planned GRAFT-GS architecture contains six main parts:

```text
1. GST: Gaussian Splat Tokenizer
2. QGen: hierarchical primitive allocation module
3. HSer: hierarchy serializer
4. AR-T: autoregressive transformer
5. Continuous/discrete prediction heads
6. Gaussian renderer + image decoder
```

---

## 1. GST: Gaussian Splat Tokenizer

The Gaussian Splat Tokenizer is responsible for converting a real image into a sequence or set of Gaussian primitives.

During tokenizer training:

```text
input image
  -> encoder
  -> Gaussian primitives
  -> splatting renderer
  -> decoder
  -> reconstructed image
```

The tokenizer should learn primitives that reconstruct the original image well.

The tokenizer output should contain information such as:

```text
position
scale
rotation
opacity
feature vector or codebook feature
optional level-of-detail information
```

The tokenizer is analogous to the VQ-VAE tokenizer used by the other approaches, but its output is not a regular token grid.

Instead of:

```text
tokens: [H, W]
```

it should produce something closer to:

```text
primitives: [N, parameter_dim]
```

where `N` may vary between images.

---

## 2. QGen: hierarchical primitive allocation

QGen is a planned module for deciding where primitives should be allocated.

Instead of dividing the image uniformly, it should create a hierarchy of spatial regions.

Conceptually:

```text
root region
  -> split into coarse cells
  -> allocate primitive budget per cell
  -> refine complex cells
  -> stop refining simple cells
```

This creates a level-of-detail structure.

Simple regions should receive fewer primitives. Complex regions should receive more primitives.

Example:

```text
smooth background region -> 1 or 2 large Gaussians
dense tissue region      -> many small Gaussians
edge/structure region    -> oriented anisotropic Gaussians
```

This is the main adaptive part of the method.

---

## 3. HSer: hierarchy serialization

Transformers operate on sequences.

Therefore, the hierarchy of regions and primitives must be serialized into a sequence.

Possible serialization strategies:

```text
row-major order
Hilbert curve order
learned spatial ordering
tree-depth order
```

The initial implementation should probably use the simplest reliable option:

```text
coarse-to-fine order + spatial row-major order
```

The more advanced Hilbert or learned ordering can be treated as future ablations.

The serialized sequence may look like:

```text
[global structure tokens]
[coarse cell primitives]
[finer cell primitives]
[EOS cell]
[EOS image]
```

---

## 4. AR-T: autoregressive primitive transformer

AR-T is the main generative model.

It should generate the primitive sequence autoregressively.

Conceptually:

```text
previous primitives -> predict next primitive
```

or, in a hierarchical version:

```text
previous cells and primitives -> predict next cell decision / primitive
```

The model may generate:

```text
split / stop decisions
number of primitives in a cell
primitive parameters
end-of-cell markers
end-of-image marker
```

Unlike VQ-token autoregression, the next item is not just one categorical token. It may contain both discrete and continuous components.

---

## 5. Continuous and discrete heads

A Gaussian primitive contains both discrete and continuous information.

Discrete parts may include:

```text
feature codebook index
cell decision
split / stop decision
EOS marker
primitive count
```

Continuous parts may include:

```text
x position
y position
scale_x
scale_y
rotation
opacity
continuous feature residual
```

Therefore, the model should not use only a single categorical classification head.

A possible design is:

```text
discrete head:
  cross-entropy loss

continuous head:
  Gaussian likelihood, mixture density, flow-matching, or diffusion-style prediction
```

For the first practical implementation, the simplest version will be used.

Planned first prototype:

```text
discretize or normalize primitive parameters
use simple MLP heads
use L1 / MSE loss for continuous parameters
use cross-entropy for discrete decisions
```

More advanced heads such as flow-matching or diffusion can be added later.

---

## 6. Renderer and image decoder

The generated primitives are not the final image.

They must first be rendered:

```text
generated Gaussians
  -> differentiable Gaussian splatting
  -> feature map
```

Then decoded:

```text
feature map
  -> CNN decoder
  -> RGB image
```

The renderer should be differentiable so that tokenizer training can use reconstruction losses.

The image decoder is responsible for converting the rendered feature map into a realistic image.

---

## Planned training stages

GRAFT-GS is too complex to train end-to-end from the beginning.

The recommended training plan is staged.

---

### Stage 1: train the Gaussian tokenizer

Goal:

```text
learn to reconstruct images using 2D Gaussian primitives
```

Training pipeline:

```text
image
  -> GST encoder
  -> Gaussian primitives
  -> renderer
  -> decoder
  -> reconstructed image
```

Possible losses:

```text
L1 reconstruction loss
MSE reconstruction loss
optional perceptual loss
primitive budget penalty
geometry regularization
opacity regularization
scale regularization
```

The most important output of Stage 1 is a stable primitive representation.

If the tokenizer is bad, the autoregressive model will also be bad.

---

### Stage 2: extract primitive sequences

After the tokenizer is trained, every training image should be converted into a primitive sequence.

For each image:

```text
image -> GST encoder -> primitive sequence
```

These sequences become the training data for the autoregressive prior.

The extracted dataset should contain:

```text
primitive parameters
cell / level information
optional class labels
sequence masks
EOS markers
```

---

### Stage 3: train the autoregressive prior

The autoregressive transformer learns to generate primitive sequences.

Training objective:

```text
predict next primitive or next primitive component from previous context
```

The loss may combine:

```text
cross-entropy for discrete components
L1 / MSE / likelihood loss for continuous parameters
EOS prediction loss
count / budget prediction loss
```

---

### Stage 4: generate and render

At inference time:

```text
1. Start from BOS.
2. Generate primitive sequence autoregressively.
3. Stop at EOS or maximum primitive budget.
4. Render generated primitives with Gaussian splatting.
5. Decode rendered features into RGB image.
```

---

## Initial project-scale simplification

For this project, the first implementation should be simplified.

Planned first version:

```text
fixed maximum number of primitives
fixed-size primitive tensor
no reinforcement learning
no learned serialization
no diffusion or flow head
simple continuous regression head
simple row-major or coarse-to-fine ordering
class-conditional optional
```

Initial representation:

```text
primitive_i = (
    x,
    y,
    scale_x,
    scale_y,
    rotation,
    opacity,
    feature_code
)
```

Initial losses:

```text
tokenizer:
  L1 reconstruction loss
  primitive budget regularization
  scale regularization
  opacity regularization

autoregressive prior:
  cross-entropy for feature_code
  L1 or MSE for continuous primitive parameters
  cross-entropy for EOS / active primitive mask
```

This makes the method implementable within the project while preserving the main idea.

---

## Differences from VAR and other baselines

### Difference from raster autoregression

Raster autoregression generates a fixed sequence of grid tokens:

```text
token_0 -> token_1 -> ... -> token_N
```

GRAFT-GS generates visual primitives:

```text
primitive_0 -> primitive_1 -> ... -> primitive_N
```

Raster tokens are tied to a grid. Gaussian primitives have continuous positions, scales, and rotations.

---

### Difference from VAR

VAR generates increasingly finer token grids:

```text
1x1 -> 2x2 -> 4x4 -> 8x8 -> 16x16
```

GRAFT-GS generates a variable or semi-variable set of primitives organized by spatial structure or level of detail.

VAR is still grid-token-based. GRAFT-GS is primitive-based.

---

### Difference from MaskGIT

MaskGIT starts with masked grid tokens and iteratively fills them.

GRAFT-GS does not fill a fixed token grid. It generates parametric primitives and renders them.

---

### Difference from VQ-VAE token generation

VQ-VAE token generation depends on a learned discrete codebook.

GRAFT-GS may use a feature codebook, but its core representation also includes continuous geometric parameters.

Therefore, the generated object is richer than a single token ID.

---

## Advantages expected from GRAFT-GS

Potential advantages:

```text
adaptive number of primitives
better quality-cost control
explicit spatial parameters
natural level-of-detail structure
possible interpretability
possible better handling of simple versus complex regions
```

For medical images, this may be useful because some regions are visually simple while others require detailed local structure.

---

## Summary

GRAFT-GS is a planned custom approach based on autoregressive generation of 2D Gaussian primitives.

The core idea is:

```text
generate images as sequences of renderable primitives instead of fixed-grid tokens
```

Compared with the existing VAR implementation, GRAFT-GS moves the representation from:

```text
multiscale discrete token grids
```

to:

```text
adaptive parametric Gaussian primitives
```

This makes the method more complex, but also more original.
