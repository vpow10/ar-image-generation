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

# Quick experiment mode: auto-trains a small model if needed, runs pipeline with 5 quality samples.
pipeline-quick:
	uv run python scripts/run_pipeline.py --quick

# Normal experiment mode: requires pre-trained checkpoints. Experiments evaluate approaches on full val split.
pipeline:
	uv run python scripts/run_pipeline.py