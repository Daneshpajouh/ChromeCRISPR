#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.workflow-venv"
SNAKEFILE="${ROOT_DIR}/workflow/Snakefile"
CORES="${SNAKEMAKE_CORES:-1}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r "${ROOT_DIR}/workflow/requirements-snakemake.txt"

if [[ "$#" -eq 0 ]]; then
  set -- report
fi

exec snakemake \
  --directory "${ROOT_DIR}" \
  --snakefile "${SNAKEFILE}" \
  --cores "${CORES}" \
  "$@"
