# Adding a New Approach

To add a new autoregressive generation method:

1. Create a folder:

```text
src/ar_image_generation/approaches/<method_name>/
```

2. Implement an approach class that subclasses `AutoregressiveApproach`.
3. Implement:
```python
training_step(...)
generate(...)
```
4. Register the approach in:
```text
src/ar_image_generation/approaches/registry.py
```
5. Add a config file:
```text
configs/approach/<method_name>.yaml
```
6. Add a smoke test:
```text
tests/test_<method_name>_shapes.py
```
7. Add documentation:
```text
docs/approaches/<method_name>.md
```
The method should not implement custom dataset loading, logging, checkpointing, or evaluation unless absolutely necessary.
