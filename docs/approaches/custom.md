# Custom Approach: GRAFT-GS Geometry-Bank Residual Feature-Code AR

This document describes the final custom image generation approach implemented in the project.

The original idea was called:

```text
GRAFT-GS
Gaussian Rendering Autoregressive Flow Transformer
```

During experiments, the method evolved into a more stable practical variant:

```text
GRAFT-GS Geometry-Bank Residual Feature-Code AR
```

The final method is a semi-parametric Gaussian primitive generator. It uses a learned Gaussian primitive tokenizer, samples realistic primitive geometry from a real geometry bank, and autoregressively generates residual-quantized primitive appearance codes.

---

## High-level idea

Most baseline methods in this project generate images through regular discrete image-token grids.

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

The custom approach uses a different representation.

Instead of representing an image as a grid of tokens, the image is represented as a set of anchored 2D Gaussian primitives.

Each primitive contains:

```text
position
scale
rotation
opacity
visual feature vector
```

The primitives are rendered with differentiable Gaussian splatting into a feature map. A CNN decoder then converts that feature map into the final RGB image.

The final generation pipeline is:

```text
class label
  -> sample real primitive geometry from geometry bank
  -> autoregressively generate residual feature codes
  -> reconstruct primitive features from residual codebooks
  -> render Gaussian primitives
  -> decode rendered feature map
  -> generated image
```

---

## Final pipeline

The final custom method consists of five main components:

```text
1. GRAFT-GS tokenizer
2. Residual feature codebook
3. Code-only autoregressive prior
4. Real geometry bank
5. Gaussian renderer + decoder
```

The complete pipeline is:

```text
training image
  -> GRAFT-GS tokenizer
  -> Gaussian primitives
  -> split into geometry and feature vectors
  -> residual feature-code extraction
  -> train AR prior over feature codes
```

At sampling time:

```text
sample class label
  -> sample real geometry from geometry bank
  -> generate residual feature codes autoregressively
  -> reconstruct continuous primitive features
  -> combine geometry + features
  -> render and decode image
```

---

## Core representation

Each image is represented as a fixed number of Gaussian primitives.

For the final PathMNIST setup:

```text
image resolution:       64 x 64
number of primitives:   256
primitive feature dim:  64
```

Each primitive can be written as:

```text
p_i = (g_i, f_i)
```

where:

```text
g_i = (x_i, y_i, scale_x_i, scale_y_i, rotation_i, opacity_i)
f_i = 64-dimensional learned visual feature
```

So the full primitive is:

```text
p_i = (
    x,
    y,
    scale_x,
    scale_y,
    rotation,
    opacity,
    feature_0,
    ...,
    feature_63
)
```

The geometry part controls where and how the primitive contributes to the rendered feature map. The feature vector controls what visual content the primitive carries.

---

## GRAFT-GS tokenizer

The tokenizer is the strongest component of the method.

It learns to reconstruct PathMNIST images using Gaussian primitives.

Tokenizer pipeline:

```text
image
  -> CNN encoder
  -> anchored Gaussian primitives
  -> differentiable Gaussian renderer
  -> rendered feature map
  -> CNN decoder
  -> reconstructed image
```

The tokenizer does not output a grid of discrete tokens. Instead, it outputs a fixed-size primitive tensor:

```text
[B, 256, 70]
```

where:

```text
70 = 6 geometry parameters + 64 feature parameters
```

The final tokenizer uses anchored local primitives. Instead of predicting all primitives from one global vector, it places primitive anchors on a regular spatial grid and predicts local primitive parameters from local CNN features.

This was essential for stable reconstruction.

Conceptually:

```text
regular anchor grid
  -> sample local CNN feature at each anchor
  -> predict local primitive geometry and feature
```

The tokenizer reconstructs PathMNIST much better than the VQ-VAE tokenizer used by the VAR baseline. This happens because the GRAFT-GS tokenizer is not bottlenecked by discrete VQ code IDs. It uses continuous geometry and continuous 64-dimensional primitive features.

---

## Residual feature codebook

The tokenizer reconstructs images using continuous primitive features. Directly generating these 64-dimensional feature vectors with a continuous autoregressive prior was unstable.

Therefore, the final method quantizes primitive features using residual quantization.

Instead of approximating a feature with one code:

```text
feature ≈ code_1
```

the final method uses two residual codes:

```text
feature ≈ codebook_1[code_1] + codebook_2[code_2]
```

For the final setup:

```text
num_quantizers: 2
codes per quantizer: 1024
feature dimension: 64
```

Each primitive feature is represented as:

```text
(code_1, code_2)
```

The residual codebook reduces the difficulty of the generative problem. The autoregressive prior no longer needs to generate a continuous 64-dimensional feature vector. It only needs to generate two discrete code IDs per primitive.

---

## Code-only autoregressive prior

The final generative model is a causal transformer trained over residual feature codes.

For each image, the target sequence is:

```text
[(code_1, code_2)_0,
 (code_1, code_2)_1,
 ...
 (code_1, code_2)_255]
```

The model predicts the next primitive feature codes autoregressively:

```text
previous primitive codes -> next primitive codes
```

The model is class-conditional. It receives the PathMNIST class label as a conditioning embedding.

Training objective:

```text
cross-entropy over residual feature-code IDs
```

For two residual quantizers, the loss is the average cross-entropy over both code streams:

```text
loss = 0.5 * CE(code_1) + 0.5 * CE(code_2)
```

The final prior does not generate geometry. This is intentional.

Earlier experiments showed that generating primitive geometry directly was unstable. Small errors in position, scale, and opacity accumulated across 256 primitives and produced chaotic images, white holes, purple blocks, and unstable tissue structure.

The final prior therefore generates only appearance codes.

---

## Geometry bank

To avoid unstable geometry generation, the final method uses a real geometry bank.

The geometry bank is built by encoding real training images with the trained GRAFT-GS tokenizer and storing their primitive geometry:

```text
image
  -> tokenizer.encode(image)
  -> extract geometry:
     x, y, scale_x, scale_y, rotation, opacity
  -> store geometry in class-specific bank
```

At sampling time:

```text
sample class label
  -> sample real geometry from that class geometry bank
  -> generate residual feature codes
  -> combine sampled geometry with generated features
```

This makes the method semi-parametric.

The geometry is not generated from scratch. It is sampled from the empirical geometry distribution learned by the tokenizer on the training set.

This was the most stable final compromise:

```text
geometry: sampled from real encoded geometry bank
appearance: generated autoregressively
```

---

## Final sampling procedure

Final sampling works as follows:

```text
1. Sample a class label.
2. Sample a real encoded primitive geometry from the geometry bank for that class.
3. Start the autoregressive code prior from BOS.
4. Generate residual feature codes for all 256 primitives.
5. Reconstruct continuous primitive features:
   feature = codebook_1[code_1] + codebook_2[code_2]
6. Combine sampled geometry and generated features.
7. Render Gaussian primitives into a feature map.
8. Decode the feature map into a 64x64 RGB image.
```

The final selected sampling configuration was:

```text
code_temperature: 0.95
top_k: 128
max_bank_per_class: 512
```

This version produced the most diverse and visually balanced samples among the tested custom variants.

---

## Final selected custom candidate

The final custom candidate is:

```text
GRAFT-GS tokenizer
+ residual feature codebook q2_k1024
+ code-only AR prior
+ real geometry-bank sampling
+ code_temperature = 0.95
+ top_k = 128
```

The final reproducible sampling command is:

```bash
uv run python scripts/sample_graft_gs_code_only_prior_geometry_bank.py \
  --config configs/experiment/custom_graft_gs_code_only_prior_pathmnist64.yaml \
  --checkpoint runs/graft_gs_code_only_prior_q2_k1024_60ep/checkpoints/best.pt \
  --output runs/final_candidates/custom/graft_gs_code_only_geometry_bank_t095_k128.png \
  --num-samples 64 \
  --code-temperature 0.95 \
  --top-k 128 \
  --max-bank-per-class 512
```

The final candidate checkpoint should be saved as:

```text
checkpoints/approaches/custom/graft_gs_code_only_prior_q2_k1024_best.pt
```

The tokenizer and residual codebook used by the final method are:

```text
checkpoints/tokenizer/pathmnist64_graft_gs_v2.pt
checkpoints/tokenizer/pathmnist64_graft_gs_residual_codebook_q2_k1024.pt
```

---

## Experiments and design evolution

Several variants were tested before reaching the final method.

---

### 1. Fully continuous primitive prior

The first prior attempted to generate the full primitive vector:

```text
x, y, scale_x, scale_y, rotation, opacity, feature_0 ... feature_63
```

This was trained with a continuous Gaussian likelihood.

The training loss decreased, but free sampling failed. The generated images were mostly chaotic, noisy, and visually unstable.

Reason:

```text
The model had to generate 256 high-dimensional continuous primitive vectors.
Small errors accumulated during autoregressive sampling.
The decoder received out-of-distribution primitive features.
```

Conclusion:

```text
discarded
```

---

### 2. Constrained continuous primitive prior

The next version generated constrained primitive vectors instead of raw primitive vectors.

The target representation was more interpretable:

```text
bounded position
bounded scale
bounded opacity
bounded feature values
```

This reduced pure noise but caused mean-image collapse. The samples became smooth, blurry, and repetitive.

Reason:

```text
The continuous Gaussian head regressed toward safe average primitive features.
Mean primitive features decode into blurry images.
```

Conclusion:

```text
discarded
```

---

### 3. Single-code hybrid prior

The next version quantized each primitive feature into one discrete feature code.

Representation:

```text
geometry + feature_code
```

This was more stable than continuous generation, but the samples were visibly blocky and quantized.

Reason:

```text
One feature code was too coarse to approximate the original continuous 64D primitive feature.
```

Conclusion:

```text
improved stability, but insufficient visual fidelity
```

---

### 4. Residual-code hybrid prior with generated geometry

The next version used residual quantization:

```text
feature ≈ code_1 + code_2
```

but still generated geometry with the transformer.

This improved the feature representation but did not solve the main instability. Generated geometry still produced artifacts, chaotic patches, and unstable tissue structure.

Conclusion:

```text
discarded
```

---

### 5. Code-only prior with class-average geometry

The next version stopped generating geometry.

It used:

```text
class-average geometry + autoregressively generated residual feature codes
```

This became the first stable custom generator.

It reduced chaotic geometry artifacts, but samples from the same class were too structurally similar. Some regions also became overly smooth or blocky because the geometry was averaged.

Conclusion:

```text
best stable internal model before geometry-bank sampling
```

---

### 6. Template-code prior with multiple geometry templates per class

A later version tried to improve diversity by computing multiple geometry templates per class:

```text
class -> one of K geometry templates
```

The model was conditioned on both class label and template ID.

This did not improve the samples. In practice, clustered geometry templates did not always form coherent visual layouts, and feature codes did not always match the selected template.

Conclusion:

```text
discarded
```

---

### 7. Code-only prior with real geometry-bank sampling

The final version used the trained code-only prior but replaced class-average geometry with real encoded geometry sampled from the training set.

Final representation:

```text
real sampled geometry + generated residual feature codes
```

This gave the best visual tradeoff:

```text
better diversity
less rigid layout
less chaotic than generated geometry
more realistic structure than class-average geometry
```

Conclusion:

```text
final selected custom approach
```

---

## Difference from the original theoretical GRAFT-GS idea

The original theoretical idea was a fully generative Gaussian primitive model.

It was supposed to generate:

```text
position
scale
rotation
opacity
feature
```

for every primitive.

The final implementation is more conservative:

```text
geometry is sampled from a real geometry bank
appearance is generated autoregressively
```

Therefore, the final method is not a fully parametric primitive generator. It is a semi-parametric generator.

Original theoretical approach:

```text
generate full Gaussian primitive sequence
```

Final implemented approach:

```text
sample realistic primitive geometry
generate residual feature codes
```

This change was made because direct geometry generation was empirically unstable.

---

## Difference from VAR

VAR is a low-to-high grid-token autoregressive model.

VAR pipeline:

```text
image
  -> VQ-VAE tokenizer
  -> multiscale discrete token grids
  -> transformer predicts tokens coarse-to-fine
  -> generated 16x16 token grid
  -> VQ-VAE decoder
  -> image
```

VAR generation order:

```text
1x1 -> 2x2 -> 4x4 -> 8x8 -> 16x16
```

The custom method is not grid-token based.

Custom pipeline:

```text
image
  -> Gaussian primitive tokenizer
  -> primitive geometry + primitive features
  -> residual feature codes
  -> AR transformer predicts feature codes
  -> Gaussian renderer + decoder
  -> image
```

Main difference:

```text
VAR generates a multiscale spatial token grid.
Custom method generates appearance codes for Gaussian primitives.
```

VAR has a simpler discrete representation and more direct control over image layout. The custom method has a stronger continuous renderer/tokenizer but requires special handling of geometry.

---

## Difference from raster autoregression

Raster autoregression generates a sequence of fixed-grid tokens:

```text
token_0 -> token_1 -> ... -> token_N
```

The custom method generates a sequence of primitive feature codes:

```text
primitive_code_0 -> primitive_code_1 -> ... -> primitive_code_255
```

Both are autoregressive, but the generated unit is different.

```text
raster AR: image-grid token
custom AR: Gaussian primitive feature code
```

The custom method does not generate pixels or VQ grid positions directly. It generates appearance codes for renderable Gaussian primitives.

---

## Difference from MaskGIT-style masked generation

MaskGIT-like methods usually start from a fully masked grid and iteratively fill missing tokens.

MaskGIT-style pipeline:

```text
masked grid tokens
  -> predict subset of missing tokens
  -> update confidence
  -> repeat until complete
```

The custom method is different:

```text
BOS
  -> generate primitive feature code 0
  -> generate primitive feature code 1
  -> ...
  -> generate primitive feature code 255
```

So the custom method is:

```text
causal autoregressive
primitive-based
not mask-based
not iterative refinement
```

---

## Strengths

The final method has several strengths.

```text
1. Very strong tokenizer reconstruction quality.
2. Non-grid representation based on renderable primitives.
3. Explicit primitive geometry and appearance separation.
4. Stable final generator after replacing free geometry generation with geometry-bank sampling.
5. Residual feature quantization gives a more expressive discrete appearance representation than single-code quantization.
6. The method is visibly different from all baseline approaches.
```

The main conceptual advantage is that the method shows a different image representation:

```text
not pixels
not VQ grid tokens
not multiscale token maps
but Gaussian primitive appearance codes rendered into an image
```

---

## Limitations

The final method also has important limitations.

---

### 1. Semi-parametric geometry

The model does not generate geometry from scratch.

Instead, geometry is sampled from a bank of real encoded geometries.

This improves visual stability but limits layout diversity:

```text
new samples reuse geometry layouts observed in the training set
```

---

### 2. Remaining blocky artifacts

Some generated images still contain blocky or one-color regions.

This is mostly caused by residual feature quantization.

The original tokenizer uses continuous primitive features:

```text
64D continuous feature
```

The generator uses:

```text
codebook_1[code_1] + codebook_2[code_2]
```

This is more stable for autoregressive generation, but less expressive than continuous features.

---

### 3. Appearance-geometry mismatch

Geometry is sampled from a real image, while feature codes are generated independently by the prior.

Sometimes the generated appearance codes do not perfectly match the sampled geometry.

This can produce:

```text
cloudy texture
locally inconsistent regions
overly purple or overly pale patches
```

---

### 4. No fully learned geometry prior

Attempts to generate geometry directly failed, so the final model avoids this problem rather than solving it.

A more complete version should learn a stable geometry prior.

---

### 5. Primitive sequence ordering is still simple

The model uses a fixed primitive sequence order.

Primitive autoregression may not have a natural left-to-right order like text or raster image tokens.

A better sequence order could improve consistency.

---

## Future work

The strongest future directions are listed below.

---

### 1. Learn a constrained geometry prior

Instead of fully free geometry generation, use:

```text
geometry template + small generated residual offsets
```

This would preserve stability while allowing more layout diversity.

A possible future representation:

```text
geometry = sampled_template_geometry + generated_geometry_residual
```

The residual should be strongly constrained to avoid the instability observed in fully generated geometry.

---

### 2. Condition appearance generation on geometry

The final method samples geometry and generates feature codes independently except for class conditioning.

A stronger future model should learn:

```text
p(feature_codes | class, geometry_embedding)
```

This would reduce appearance-geometry mismatch.

The geometry embedding could be produced by a small encoder over the sampled primitive geometry.

---

### 3. Improve residual feature quantization

Current feature reconstruction uses two residual codes:

```text
feature ≈ code_1 + code_2
```

Future work could use:

```text
feature ≈ code_1 + code_2 + code_3
```

or product quantization.

This could reduce blocky artifacts and improve local texture fidelity.

---

### 4. Train a tokenizer designed for discrete generation

The current GRAFT-GS tokenizer was optimized for continuous reconstruction.

A better generative tokenizer would be trained directly with quantized residual features in the loop, so the decoder learns the exact feature representation used by the prior.

Possible training path:

```text
image
  -> encoder
  -> primitive geometry + continuous feature
  -> residual quantization
  -> renderer
  -> decoder
  -> reconstruction
```

This could reduce the mismatch between tokenizer reconstruction and generative sampling.

---

### 5. Replace causal AR with masked primitive-code generation

Primitive feature codes may not have a natural causal order.

A MaskGIT-like model over primitive codes could be better:

```text
start with masked primitive codes
iteratively predict missing primitive codes
decode final primitive set
```

This may reduce exposure bias and improve global consistency.

---

### 6. Add a learned primitive ordering

The current primitive order is tied to the anchor/grid order.

Future work could test:

```text
Hilbert curve ordering
coarse-to-fine ordering
learned ordering
attention-based ordering
```

A better ordering may improve autoregressive modeling.

---

### 7. Add perceptual or texture-aware objectives

The tokenizer currently uses pixel-level reconstruction losses.

Histology images contain high-frequency local texture, so future training could include:

```text
perceptual loss
patch texture loss
adversarial loss
feature-matching loss
```

This could improve fine tissue texture.

---

## Final conclusion

The final custom method demonstrates that Gaussian primitive tokenization is a promising alternative to grid-token image representations.

The GRAFT-GS tokenizer reconstructs PathMNIST very accurately and provides an interpretable primitive-based representation. However, fully autoregressive generation of both primitive geometry and continuous primitive features was unstable.

The best-performing custom variant therefore uses a stabilized semi-parametric formulation:

```text
real geometry-bank sampling
+
autoregressive residual feature-code generation
+
Gaussian splatting renderer
+
learned decoder
```

This produces meaningful tissue-like samples and is visibly different from the baseline methods, but it still has limitations such as blocky artifacts and dependence on the geometry bank.

The method is best understood as a successful project-scale prototype of primitive-based autoregressive image generation, with clear directions for future improvement.
