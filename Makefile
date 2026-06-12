install:
	uv sync

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts

test:
	uv run pytest tests

typecheck:
	uv run mypy src

check: lint test

# ── Quick mode ──────────────────────────────────────────────────────────

# All 3 approaches at once (auto-trains tiny models if checkpoints missing).
pipeline-quick:
	uv run python scripts/run_pipeline.py --quick

# Individual approaches — quick mode, small model.
pipeline-quick-raster:
	uv run python scripts/run_pipeline.py --quick --approaches configs/experiment/smoke_test_raster.yaml

pipeline-quick-maskgit:
	uv run python scripts/run_pipeline.py --quick --approaches configs/experiment/smoke_test_maskgit.yaml

pipeline-quick-var:
	uv run python scripts/run_pipeline.py --quick --approaches configs/experiment/smoke_test_var.yaml

# Quick eval on pre-trained full models (5 FID samples, no training).
pipeline-quick-full:
	uv run python scripts/run_pipeline.py --quick \
		--approaches configs/experiment/raster_pathmnist64_debug.yaml \
		             configs/experiment/maskgit_pathmnist64_debug.yaml \
		             configs/experiment/var_pathmnist64_d4.yaml

# ── Full mode ──────────────────────────────────────────────────────────

# All 3 approaches at once.
pipeline:
	uv run python scripts/run_pipeline.py

# Individual approaches — full mode.
pipeline-raster:
	uv run python scripts/run_pipeline.py --approaches configs/experiment/raster_pathmnist64_debug.yaml

pipeline-maskgit:
	uv run python scripts/run_pipeline.py --approaches configs/experiment/maskgit_pathmnist64_debug.yaml

pipeline-var:
	uv run python scripts/run_pipeline.py --approaches configs/experiment/var_pathmnist64_d4.yaml