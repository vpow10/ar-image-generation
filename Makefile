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

# All approaches at once: Exp 1–3 + Exp 4 
pipeline-quick:
	uv run python scripts/run_pipeline.py --quick

# All approaches at once: Exp 4
convergence-quick:
	uv run python scripts/run_pipeline.py --quick --convergence-only

# All *pre-trained full models* at once: Exp 1–3 + Exp 4
pipeline-quick-full:
	uv run python scripts/run_pipeline.py --quick \
		--approaches configs/experiment/raster_pathmnist64_debug.yaml \
		             configs/experiment/maskgit_pathmnist64_debug.yaml \
		             configs/experiment/var_pathmnist64_d4.yaml

# Individual approaches: Exp 1–3 only
pipeline-quick-raster:
	uv run python scripts/run_pipeline.py --quick --no-convergence \
		--approaches configs/experiment/smoke_test_raster.yaml

pipeline-quick-maskgit:
	uv run python scripts/run_pipeline.py --quick --no-convergence \
		--approaches configs/experiment/smoke_test_maskgit.yaml

pipeline-quick-var:
	uv run python scripts/run_pipeline.py --quick --no-convergence \
		--approaches configs/experiment/smoke_test_var.yaml


# ── Full mode ──────────────────────────────────────────────────────────

# All *pre-trained full models* at once: Exp 1–3 + Exp 4
pipeline:
	uv run python scripts/run_pipeline.py \
		--approaches configs/experiment/raster_pathmnist64_debug.yaml \
		             configs/experiment/maskgit_pathmnist64_debug.yaml \
		             configs/experiment/var_pathmnist64_d4.yaml

# Individual approaches: Exp 1–3 only
pipeline-raster:
	uv run python scripts/run_pipeline.py --no-convergence \
		--approaches configs/experiment/raster_pathmnist64_debug.yaml

pipeline-maskgit:
	uv run python scripts/run_pipeline.py --no-convergence \
		--approaches configs/experiment/maskgit_pathmnist64_debug.yaml

pipeline-var:
	uv run python scripts/run_pipeline.py --no-convergence \
		--approaches configs/experiment/var_pathmnist64_d4.yaml

# All *pre-trained full models* at once: Exp 4
convergence:
	uv run python scripts/run_pipeline.py --convergence-only \
		--approaches configs/experiment/raster_pathmnist64_debug.yaml \
		             configs/experiment/maskgit_pathmnist64_debug.yaml \
		             configs/experiment/var_pathmnist64_d4.yaml