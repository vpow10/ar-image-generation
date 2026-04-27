# Adding new approach

This guide explains how to add a new generation approach to the existing project without changing shared infrastructure.

The project already provides:

- dataset loading
- config loading
- trained VQ-VAE tokenizer loading
- shared `ImageBatch`
- shared `ImageTokenizer`
- shared `AutoregressiveApproach`
- reusable transformer backbone
- checkpointing
- training script
- sampling script
- evaluation script
- metric/report saving

You need to implement your method as a new `AutoregressiveApproach`.

---

## 1. Start from a clean branch

```bash
git checkout main
git pull
uv sync
make test
```
Create a feature branch:

```bash
git checkout -b feat/<method-name>
```
## 2. Create approach files
Create the folder:

```bash
mkdir -p src/ar_image_generation/approaches/<method_name>
```

Create files:

```bash
touch src/ar_image_generation/approaches/<method_name>/__init__.py
touch src/ar_image_generation/approaches/<method_name>/model.py
touch src/ar_image_generation/approaches/<method_name>/sampler.py
touch tests/test_<method_name>_model.py
```

## 3. Implement the approach class
In:
```text
src/ar_image_generation/approaches/<method_name>/model.py
```
define one class:
```python
from ar_image_generation.approaches.base import AutoregressiveApproach, SamplingConfig
from ar_image_generation.approaches.registry import register_approach
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.tokenizers.base import ImageTokenizer


@register_approach("<method_name>")
class YourApproach(AutoregressiveApproach):
    name = "<method_name>"

    def __init__(
        self,
        *,
        vocab_size: int,
        latent_shape: tuple[int, int],
        dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        class_conditional: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_shape = latent_shape

        if class_conditional:
            raise NotImplementedError("This project currently uses unconditional generation.")

    def training_step(
        self,
        batch: ImageBatch,
        tokenizer: ImageTokenizer,
    ) -> dict[str, torch.Tensor]:
        ...

    @torch.no_grad()
    def generate(
        self,
        tokenizer: ImageTokenizer,
        batch_size: int,
        labels: torch.Tensor | None,
        device: torch.device,
        sampling_cfg: SamplingConfig,
    ) -> torch.Tensor:
        ...
```
The class must return a scalar loss from:
```python
training_step(...)
```
The class must return generated images from:
```python
generate(...)
```
Expected generated image shape:
```python
[B, 3, 64, 64]
```

## 4. Use the existing tokenizer
Do not train or load the tokenizer manually inside your approach.

Inside `training_step` use:
```python
tokenizer.eval()

with torch.no_grad():
    tokens = tokenizer.encode(batch.images)
```
The tokenizer returns:
```python
tokens: [B, H_latent, W_latent]
```
For the current default setup:
```python
tokens: [B, 8, 8]
```
Later this may become:
```python
tokens: [B, 16, 16]
```
Therefore, use `self.latent_shape` instead of hard-coding `8`.

## 5. Reuse shared modules
Use the shared transformer instead of creating a separate transformer implementation:
```python
from ar_image_generation.models.transformer import TransformerBackbone
```
Use the existing sampling helpers if needed:
```python
Use the existing sampling helpers if needed:
```
Do not reimplement:

- dataset loading
- tokenizer loading
- checkpoint saving
- image grid saving
- evaluation
- config parsing
- training loop

These are already handled by:
```text
scripts/train_approach.py
scripts/sample.py
scripts/evaluate.py
```
## 6. Register the approach
Open:
```text
src/ar_image_generation/approaches/registry.py
```
Inside `import_builtin_approaches()`, add your method import:
```python
import ar_image_generation.approaches.<method_name>.model  # noqa: F401
```
## 7. Add config
Use the existing experiment config pattern.

Example path:
```text
configs/experiment/<method_name>_pathmnist64_debug.yaml
```
The important part is:
```yaml
approach:
  name: <method_name>
  dim: 384
  depth: 6
  num_heads: 6
  mlp_ratio: 4
  dropout: 0.1
  class_conditional: false
```
