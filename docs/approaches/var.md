# VAR-style Low-to-High Resolution Generation

This method generates image tokens progressively from coarse to fine resolution.

For PathMNIST 64x64 and a tokenizer downsampling factor of 8:

```text
image resolution: 64 x 64
latent resolution: 8 x 8
scale schedule: 1x1 -> 2x2 -> 4x4 -> 8x8
```
The implementation should live in:
```text
src/ar_image_gen/approaches/var/
```
