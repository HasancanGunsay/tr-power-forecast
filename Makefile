# Canonical commands for this project.
# On Windows without `make`, use the identical PowerShell wrapper: .\make.ps1 <target>

.PHONY: help setup lint format typecheck test check clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create the locked environment and install git hooks
	uv sync --all-groups
	uv run pre-commit install

lint:  ## Check style and catch likely bugs
	uv run ruff check .

format:  ## Auto-format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Static type analysis
	uv run mypy

test:  ## Run the test suite with coverage
	uv run pytest

check: lint typecheck test  ## Everything CI runs, locally

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build
