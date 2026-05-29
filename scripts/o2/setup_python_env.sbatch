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

INSTALL_GPU_DEPS="${INSTALL_GPU_DEPS:-0}"
CUDA_MODULE="${CUDA_MODULE-cuda/12.8}"
GPU_PIP_PACKAGE="${GPU_PIP_PACKAGE:-cupy-cuda12x}"

if [ "$INSTALL_GPU_DEPS" = "1" ]; then
  if [ -n "$CUDA_MODULE" ]; then
    if ! module load "$CUDA_MODULE"; then
      echo "ERROR: failed to load CUDA module: $CUDA_MODULE" >&2
      echo "Run these discovery commands on O2:" >&2
      echo "  module spider cuda" >&2
      echo "  module load gcc/14.2.0" >&2
      echo "  module avail cuda" >&2
      exit 2
    fi
  else
    echo "WARNING: INSTALL_GPU_DEPS=1 but CUDA_MODULE is empty; skipping CUDA module load."
  fi
else
  echo "INSTALL_GPU_DEPS=0: GPU dependencies will be skipped."
fi

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

if [ "$INSTALL_GPU_DEPS" = "1" ]; then
  echo
  echo "Installing optional GPU dependency: $GPU_PIP_PACKAGE"
  python -m pip install "$GPU_PIP_PACKAGE"

  echo
  echo "CuPy import and runtime checks"
  python - <<'PY'
import cupy

print(f"cupy: OK version={cupy.__version__}")
try:
    print(f"CUDA runtime version: {cupy.cuda.runtime.runtimeGetVersion()}")
except Exception as exc:
    print(f"CUDA runtime version unavailable: {exc}")
try:
    count = cupy.cuda.runtime.getDeviceCount()
    print(f"CUDA device count: {count}")
    if count:
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        print(f"CUDA device 0: {name}")
except Exception as exc:
    print(f"CUDA device query unavailable: {exc}")
PY
else
  echo
  echo "Skipping CuPy install/import because INSTALL_GPU_DEPS=0."
fi

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
