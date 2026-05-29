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

job_raw="$(sbatch --parsable \
  --job-name=tailbin_setup_python_env \
  --partition="$SETUP_PARTITION" \
  --time="$SETUP_TIME" \
  --cpus-per-task="$SETUP_CPUS" \
  --mem="$SETUP_MEM" \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --export="ALL,RESET_O2_VENV=${RESET_O2_VENV}" \
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
