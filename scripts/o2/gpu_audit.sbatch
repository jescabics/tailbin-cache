#!/bin/bash
#SBATCH --job-name=tailbin_gpu_audit
#SBATCH -p gpu_quad
#SBATCH -t 2:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_doublep
#SBATCH --mem=16G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs outputs results tasks outputs/o2_gpu_audit results/o2_gpu_audit

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
module load cuda/11.7
module load python/3.13.1

RESULT_DIR="results/o2_gpu_audit"
GPU_MONITOR_PID=""
if [ -x /n/cluster/bin/job_gpu_monitor.sh ]; then
  (cd "$RESULT_DIR" && /n/cluster/bin/job_gpu_monitor.sh) &
  GPU_MONITOR_PID="$!"
else
  echo "GPU monitor not found at /n/cluster/bin/job_gpu_monitor.sh"
fi

cleanup() {
  if [ -n "${GPU_MONITOR_PID:-}" ]; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

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

AUDIT_SCRIPT="${GPU_AUDIT_SCRIPT:-examples/gpu_backend_audit.py}"
GPU_CONFIG="${GPU_AUDIT_CONFIG:-examples/kmax20_o2_gpu_probe.yaml}"

if [ ! -f "$AUDIT_SCRIPT" ]; then
  echo "Missing GPU audit script: $AUDIT_SCRIPT" >&2
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
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "Audit script: $AUDIT_SCRIPT"
  echo "Reference GPU config: $GPU_CONFIG"
  echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
} | tee "$RESULT_DIR/gpu_audit_metadata.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$RESULT_DIR/nvidia_smi.txt"
else
  echo "nvidia-smi not found" | tee "$RESULT_DIR/nvidia_smi.txt"
fi

# examples/gpu_backend_audit.py is currently self-contained and does not accept
# a config argument. Keep examples/kmax20_o2_gpu_probe.yaml in the result bundle
# as the matching GPU probe config for review.
echo
echo "Running GPU backend audit"
python -u "$AUDIT_SCRIPT" | tee "$RESULT_DIR/gpu_backend_audit.json"

echo "Finished: $(date)" | tee -a "$RESULT_DIR/gpu_audit_metadata.txt"
