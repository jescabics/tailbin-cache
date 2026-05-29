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

python -m tailbin_cache.cli --help

echo "O2 Python environment check passed."
