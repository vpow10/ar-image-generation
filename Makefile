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


# ── Config shorthands ────────────────────────────────────────────────────────
RASTER   = configs/experiment/raster_pathmnist64_debug.yaml
MASKGIT  = configs/experiment/maskgit_pathmnist64_debug.yaml
VAR      = configs/experiment/var_pathmnist64_d4.yaml
CUSTOM   = configs/experiment/custom_graft_gs_prior_pathmnist64.yaml

RASTER_Q  = configs/experiment/smoke_test_raster.yaml
MASKGIT_Q = configs/experiment/smoke_test_maskgit.yaml
VAR_Q     = configs/experiment/smoke_test_var.yaml

FULL  = $(RASTER) $(MASKGIT) $(VAR) $(CUSTOM)


# ── Quick mode  (smoke-test models, FID/IS limited to 5 samples) ─────────────

# All 3 smoke models: Exp 1–3 + Exp 4
pipeline-quick:
	uv run python scripts/run_pipeline.py --quick

# Individual smoke model: Exp 1–3
pipeline-quick-raster:
	uv run python scripts/run_pipeline.py --quick --no-convergence --approaches $(RASTER_Q)

pipeline-quick-maskgit:
	uv run python scripts/run_pipeline.py --quick --no-convergence --approaches $(MASKGIT_Q)

pipeline-quick-var:
	uv run python scripts/run_pipeline.py --quick --no-convergence --approaches $(VAR_Q)

# All 3 smoke models: Exp 4
convergence-quick:
	uv run python scripts/run_pipeline.py --quick --convergence-only


# ── Quick mode  (full models, FID/IS limited to 5 samples) ─────────────

# All 4 full models: Exp 1–3 + Exp 4
pipeline-quick-full:
	uv run python scripts/run_pipeline.py --quick --approaches $(FULL)

# Individual full model: Exp 1–3
pipeline-quick-full-raster:
	uv run python scripts/run_pipeline.py --quick --no-convergence --approaches $(RASTER)

pipeline-quick-full-maskgit:
	uv run python scripts/run_pipeline.py --quick --no-convergence --approaches $(MASKGIT)

pipeline-quick-full-var:
	uv run python scripts/run_pipeline.py --quick --no-convergence --approaches $(VAR)

pipeline-quick-full-custom:
	uv run python scripts/run_pipeline.py --quick --no-convergence --approaches $(CUSTOM)

# All 4 full models: Exp 4
convergence-quick-full:
	uv run python scripts/run_pipeline.py --quick --convergence-only --approaches $(FULL)


# ── Full mode  (full val split, ~10k images) ─────────────────────────────────

# All 4 full models: Exp 1–3 + Exp 4
pipeline-full:
	uv run python scripts/run_pipeline.py --approaches $(FULL)

# All 4 full models: Exp 1,2  with limited samples number + Exp 3 + Exp 4
pipeline-full-1000:
	uv run python scripts/run_pipeline.py --approaches $(FULL) --quality-num-samples 1000

# Individual full model: Exp 1–3
pipeline-raster:
	uv run python scripts/run_pipeline.py --no-convergence --approaches $(RASTER)

pipeline-maskgit:
	uv run python scripts/run_pipeline.py --no-convergence --approaches $(MASKGIT)

pipeline-var:
	uv run python scripts/run_pipeline.py --no-convergence --approaches $(VAR)

pipeline-custom:
	uv run python scripts/run_pipeline.py --no-convergence --approaches $(CUSTOM)

# All 4 full models: Exp 4
convergence-full:
	uv run python scripts/run_pipeline.py --convergence-only --approaches $(FULL)
