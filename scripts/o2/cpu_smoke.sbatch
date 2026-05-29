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
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="python3"
if [ -f /home/jae37/pythonEnv_3.10.11/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/jae37/pythonEnv_3.10.11/bin/activate
  PYTHON_BIN="python"
fi

TAILBIN_CMD=(tailbin-cache)
if ! command -v tailbin-cache >/dev/null 2>&1; then
  TAILBIN_CMD=("$PYTHON_BIN" -m tailbin_cache.cli)
fi

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
  echo "Python: $("$PYTHON_BIN" --version 2>&1)"
  echo "Config: $CONFIG"
  echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
} | tee "$RESULT_DIR/cpu_smoke_metadata.txt"

echo
echo "tailbin-cache --help"
"${TAILBIN_CMD[@]}" --help

echo
echo "Running CPU smoke estimate"
"${TAILBIN_CMD[@]}" estimate --config "$CONFIG" | tee "$RESULT_DIR/cpu_smoke_estimate.json"

echo
echo "Running tiny CPU smoke plan with --limit-bundles 1"
"${TAILBIN_CMD[@]}" plan --config "$CONFIG" --output-dir "$PLAN_DIR" --limit-bundles 1 | tee "$RESULT_DIR/cpu_smoke_plan_summary.json"

echo "Finished: $(date)" | tee -a "$RESULT_DIR/cpu_smoke_metadata.txt"
