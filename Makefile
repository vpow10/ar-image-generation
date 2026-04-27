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