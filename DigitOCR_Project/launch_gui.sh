#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ -x "${VENV_PYTHON}" ]]; then
  exec "${VENV_PYTHON}" "${PROJECT_ROOT}/gui_app.pyw"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Python 3.10+ is required to run the project." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/gui_app.pyw"
