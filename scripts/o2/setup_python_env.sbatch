#!/bin/bash
#SBATCH --job-name=tailbin_setup_python_env
#SBATCH -p short
#SBATCH -t 1:00:00
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results

RESULT_FILE="results/o2_python_env_check.txt"
exec > >(tee "$RESULT_FILE") 2>&1

echo "Tailbin O2 Python environment setup"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo

if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: module command not found. This setup must run on O2 with environment modules available." >&2
  exit 2
fi

module purge
module load gcc/14.2.0
module load python/3.13.1

echo "Loaded modules:"
module list 2>&1
echo

echo "Module Python:"
which python3
python3 --version
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"ERROR: Python >=3.10 required, got {sys.version.split()[0]}")
PY

if [ "${RESET_O2_VENV:-0}" = "1" ]; then
  echo "RESET_O2_VENV=1: removing existing .venv_o2"
  rm -rf .venv_o2
fi

if [ ! -d .venv_o2 ]; then
  echo "Creating .venv_o2 with loaded Python"
  python3 -m venv .venv_o2
else
  echo "Reusing existing .venv_o2"
fi

# shellcheck disable=SC1091
source .venv_o2/bin/activate

echo
echo "Virtualenv Python:"
which python
python --version
python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"ERROR: .venv_o2 Python must be >=3.10, got {sys.version.split()[0]}")
PY

echo
echo "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo
echo "Installing Tailbin package editable"
python -m pip install -e .

echo
echo "Import and version checks"
python - <<'PY'
import importlib

modules = ["numpy", "scipy", "yaml", "h5py", "numba", "tailbin_cache"]
for name in modules:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "unknown")
    print(f"{name}: OK version={version}")
PY

echo
echo "CLI check"
python -m tailbin_cache.cli --help

echo
echo "pip list"
python -m pip list

echo
echo "Setup complete: .venv_o2 is ready."
