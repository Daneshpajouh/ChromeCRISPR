#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.smoke-venv"
REQUIREMENTS_FILE="${ROOT_DIR}/workflow/requirements-smoke.txt"
STAMP_FILE="${VENV_DIR}/.requirements-smoke.stamp"

if [[ -n "${SMOKE_PYTHON:-}" ]]; then
  PYTHON_BIN="${SMOKE_PYTHON}"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
else
  PYTHON_BIN="$(command -v python3)"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "No usable Python interpreter found for the smoke environment." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
if [[ ! -f "${STAMP_FILE}" || "${REQUIREMENTS_FILE}" -nt "${STAMP_FILE}" ]]; then
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r "${REQUIREMENTS_FILE}"
  touch "${STAMP_FILE}"
fi

exec python "${ROOT_DIR}/scripts/run_checkpoint_smoke.py" "$@"
