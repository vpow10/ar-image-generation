# Experiments

Every approach should be evaluated with the same pipeline.

## Required outputs

Each trained approach should produce:

```text
runs/<run_name>/
├── config.json
├── metrics.jsonl
├── latest_metrics.json
├── summary.json
├── checkpoints/
│   ├── best.pt
│   └── last.pt
└── samples/
    ├── epoch_001.png
    └── final.png
```
Each evaluation should produce:

```text
runs/eval/<approach_name>/
├── metrics.json
├── samples.png
└── samples.pt
```
# Metrics
Current baseline metrics:
1. Validation/test token prediction loss.
2. Trainable parameter count.
3. Total parameter count.
4. Sampling speed in samples per second.
5. Visual sample grid.
# Later optional metrics:
For the final report, we may add:

1. FID or clean-FID style metric.
2. Reconstruction quality of tokenizer.
3. Human visual inspection grid.
4. Training time.
5. GPU memory usage.