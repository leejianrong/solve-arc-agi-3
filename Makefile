.PHONY: help install lint type test check smoke

help:
	@echo "install  - uv sync (dev group)"
	@echo "lint     - ruff check + ruff format --check"
	@echo "type     - mypy --strict over src"
	@echo "test     - pytest"
	@echo "check    - lint + type + test"
	@echo "smoke    - deterministic no-network baseline episode"

install:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run mypy src

test:
	uv run pytest -q

check: lint type test

smoke:
	uv run solve-arc-agi-3 smoke
