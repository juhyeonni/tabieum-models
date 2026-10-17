#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== stage 1: lint, lock, results, card, bench build"
uv run --only-dev ruff check .
uv lock --check
uv run --only-dev python scripts/check_results.py
(cd bench && swift build -c release)

if [ $# -eq 0 ]; then
    echo "== stage 2 skipped: pass a .mlpackage or its .zip (scripts/check.sh <package>)"
    exit 0
fi

echo "== stage 2: model fidelity and contract"
uv run --project eval python scripts/check_model.py "$1"
echo "== all checks passed"
