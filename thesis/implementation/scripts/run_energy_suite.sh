#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
THESIS_DIR="$(cd "$ROOT_DIR/.." && pwd)"
PYTHON_BIN="$THESIS_DIR/.venv/bin/python"
SUITE="${1:-configs/suites/local_energy.yml}"
OWNER_UID="$(id -u)"
OWNER_GID="$(id -g)"
PYTHONPATH_VALUE="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing thesis virtualenv Python at $PYTHON_BIN" >&2
  exit 1
fi

sudo --preserve-env=OMP_NUM_THREADS,OMP_PROC_BIND,OMP_PLACES env PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -m quantum_bench.bench run --suite "$SUITE"

sudo chown -R "$OWNER_UID:$OWNER_GID" "$ROOT_DIR/runs"
