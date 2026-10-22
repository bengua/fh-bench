#!/usr/bin/env bash
# What must be true before a change lands: lint and tests, the same by hand and in CI.
# Uses uv when it is installed, else the current Python environment (pip install -e '.[dev]').
set -eu
cd "$(dirname "$0")"

if command -v uv >/dev/null; then
  uv sync --all-extras -q
  run() { uv run "$@"; }
else
  run() { "$@"; }
fi

run ruff check .
run python -m pytest -q
echo "OK"
