#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: module command not found. Run this on O2 with environment modules available." >&2
  exit 2
fi

module purge
module load gcc/14.2.0
module load python/3.13.1

CHECK_GPU_DEPS="${CHECK_GPU_DEPS:-0}"
CUDA_MODULE="${CUDA_MODULE-cuda/12.8}"
if [ "$CHECK_GPU_DEPS" = "1" ]; then
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
    echo "WARNING: CHECK_GPU_DEPS=1 but CUDA_MODULE is empty; skipping CUDA module load."
  fi
fi

if [ ! -f .venv_o2/bin/activate ]; then
  echo "ERROR: Missing .venv_o2. First run: bash scripts/o2/submit_setup_env.sh" >&2
  exit 2
fi

# shellcheck disable=SC1091
source .venv_o2/bin/activate

echo "Loaded modules:"
module list 2>&1
echo

echo "Python path: $(which python)"
echo "Python version: $(python --version 2>&1)"
python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"ERROR: .venv_o2 Python must be >=3.10, got {sys.version.split()[0]}")
print("Python version check: OK")
PY

python - <<'PY'
import importlib

modules = ["numpy", "scipy", "yaml", "h5py", "numba", "tailbin_cache"]
for name in modules:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "unknown")
    print(f"{name}: OK version={version}")
PY

if [ "$CHECK_GPU_DEPS" = "1" ]; then
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
  echo "CHECK_GPU_DEPS=0: skipping CuPy checks."
fi

python -m tailbin_cache.cli --help

echo "O2 Python environment check passed."
