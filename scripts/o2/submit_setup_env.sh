#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

mkdir -p logs results

SETUP_PARTITION="${SETUP_PARTITION:-short}"
SETUP_TIME="${SETUP_TIME:-1:00:00}"
SETUP_MEM="${SETUP_MEM:-4G}"
SETUP_CPUS="${SETUP_CPUS:-1}"
RESET_O2_VENV="${RESET_O2_VENV:-0}"
INSTALL_GPU_DEPS="${INSTALL_GPU_DEPS:-0}"
CUDA_MODULE="${CUDA_MODULE-cuda/12.8}"
GPU_PIP_PACKAGE="${GPU_PIP_PACKAGE:-cupy-cuda12x}"

job_raw="$(sbatch --parsable \
  --job-name=tailbin_setup_python_env \
  --partition="$SETUP_PARTITION" \
  --time="$SETUP_TIME" \
  --cpus-per-task="$SETUP_CPUS" \
  --mem="$SETUP_MEM" \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --export="ALL,RESET_O2_VENV=${RESET_O2_VENV},INSTALL_GPU_DEPS=${INSTALL_GPU_DEPS},CUDA_MODULE=${CUDA_MODULE},GPU_PIP_PACKAGE=${GPU_PIP_PACKAGE}" \
  scripts/o2/setup_python_env.sbatch)"
jobid="${job_raw%%;*}"

echo "Submitted O2 Python environment setup job: $jobid"
echo
echo "Monitor:"
echo "  squeue -j $jobid"
echo "  O2_jobs_report -j $jobid"
echo "  sacct -j $jobid --format=JobId,JobName,Partition,NCPUS,State,ReqMem,MaxRSS,Elapsed,TimeLimit,ExitCode"
echo
echo "After it completes, run:"
echo "  bash scripts/o2/check_python_env.sh"
echo
echo "Examples:"
echo "  bash scripts/o2/submit_setup_env.sh"
echo "  INSTALL_GPU_DEPS=1 bash scripts/o2/submit_setup_env.sh"
echo "  RESET_O2_VENV=1 INSTALL_GPU_DEPS=1 bash scripts/o2/submit_setup_env.sh"
