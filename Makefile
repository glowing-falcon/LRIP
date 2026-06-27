# Makefile

.venv:
	uv sync

install: .venv

format:
	.venv/bin/ruff check src --fix --show-fixes
