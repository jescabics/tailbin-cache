#!/bin/bash
#SBATCH --job-name=tailbin_cpu_smoke
#SBATCH -p short
#SBATCH -t 1:00:00
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs outputs results tasks outputs/o2_smoke results/o2_smoke

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: module command not found. This job must run on O2 with environment modules available." >&2
  exit 2
fi

module purge
module load gcc/14.2.0
module load python/3.13.1

if [ ! -f .venv_o2/bin/activate ]; then
  echo "Missing .venv_o2. First run: bash scripts/o2/submit_setup_env.sh" >&2
  exit 2
fi

# shellcheck disable=SC1091
source .venv_o2/bin/activate

python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"ERROR: .venv_o2 Python must be >=3.10, got {sys.version.split()[0]}")
PY

CONFIG="${CPU_SMOKE_CONFIG:-examples/kmax2000_cpu_smoke.yaml}"
RESULT_DIR="results/o2_smoke"
PLAN_DIR="outputs/o2_smoke/plan"

if [ ! -f "$CONFIG" ]; then
  echo "Missing CPU smoke config: $CONFIG" >&2
  exit 2
fi

{
  echo "Job ID: ${SLURM_JOB_ID:-unknown}"
  echo "Host: $(hostname)"
  echo "Started: $(date)"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Loaded modules:"
  module list 2>&1
  echo "Python path: $(which python)"
  echo "Python version: $(python --version 2>&1)"
  echo "Config: $CONFIG"
  echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
} | tee "$RESULT_DIR/cpu_smoke_metadata.txt"

echo
echo "python -m tailbin_cache.cli --help"
python -m tailbin_cache.cli --help

echo
echo "Running CPU smoke estimate"
python -m tailbin_cache.cli estimate --config "$CONFIG" | tee "$RESULT_DIR/cpu_smoke_estimate.json"

echo
echo "Running tiny CPU smoke plan with --limit-bundles 1"
python -m tailbin_cache.cli plan --config "$CONFIG" --output-dir "$PLAN_DIR" --limit-bundles 1 | tee "$RESULT_DIR/cpu_smoke_plan_summary.json"

echo "Finished: $(date)" | tee -a "$RESULT_DIR/cpu_smoke_metadata.txt"
