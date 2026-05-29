#!/bin/bash
#SBATCH --job-name=tailbin_collect_representative_cal
#SBATCH -p short
#SBATCH -t 0:30:00
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs outputs results tasks

RUN_ID="${RUN_ID:-}"
if [ -z "$RUN_ID" ]; then
  echo "RUN_ID is required for representative calibration collection." >&2
  exit 2
fi

run_dir="results/o2_representative_calibration/${RUN_ID}"
bundle_path="results/tailbin_o2_representative_calibration_${RUN_ID}.tgz"

mkdir -p "$run_dir"/{accounting,docs,examples,logs/current,gpu_audit}
shopt -s nullglob

copy_file_if_exists() {
  local src="$1"
  local dest="$2"
  if [ -f "$src" ]; then
    mkdir -p "$(dirname "$dest")"
    cp -a "$src" "$dest"
  else
    echo "missing file: $src" >> "$run_dir/missing_files.txt"
  fi
}

copy_matching_files() {
  local dest="$1"
  local label="$2"
  shift 2
  local copied=0
  mkdir -p "$dest"
  for pattern in "$@"; do
    local matches=( $pattern )
    for src in "${matches[@]}"; do
      if [ -f "$src" ]; then
        cp -a "$src" "$dest/$(basename "$src")"
        copied=1
      fi
    done
  done
  if [ "$copied" -eq 0 ]; then
    echo "No files matched for $label: $*" >> "$run_dir/missing_files.txt"
  fi
}

collect_accounting() {
  local jobid="$1"
  local label="$2"
  local outfile="$run_dir/accounting/${label}_${jobid:-missing}.txt"

  if [ -z "$jobid" ]; then
    echo "No job ID provided for $label" > "$outfile"
    return 0
  fi

  {
    echo "Accounting for $label job $jobid"
    if command -v O2_jobs_report >/dev/null 2>&1; then
      O2_jobs_report -j "$jobid"
    elif command -v sacct >/dev/null 2>&1; then
      sacct -j "$jobid" --format=JobId,JobName,Partition,NCPUS,State,ReqMem,MaxRSS,Elapsed,CPUTime,TimeLimit,ExitCode,Start,End
    else
      echo "Neither O2_jobs_report nor sacct is available."
    fi
  } > "$outfile" 2>&1 || true
}

{
  echo "Representative calibration collection"
  echo "Run ID: ${RUN_ID}"
  echo "Created: $(date)"
  echo "Repository: ${SLURM_SUBMIT_DIR:-$PWD}"
  echo "Submit git commit: ${SUBMIT_GIT_COMMIT:-unknown}"
  echo "Current git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Calibration job ID: ${CAL_JOB_ID:-}"
  echo "GPU job ID: ${GPU_JOB_ID:-}"
  echo "Collector job ID: ${SLURM_JOB_ID:-}"
  echo "Calibration config: ${CAL_CONFIG:-examples/local34_diag_v1_k10000_1k.yaml}"
} > "$run_dir/collection_metadata.txt"

git rev-parse HEAD > "$run_dir/git_commit.txt" 2>/dev/null || echo unknown > "$run_dir/git_commit.txt"
git status --short --branch -uall > "$run_dir/git_status_after_collection.txt" 2>&1 || true

copy_file_if_exists docs/O2_RUNBOOK.md "$run_dir/docs/O2_RUNBOOK.md"
copy_file_if_exists docs/O2_RESOURCE_CALIBRATION.md "$run_dir/docs/O2_RESOURCE_CALIBRATION.md"
copy_file_if_exists docs/WORKFLOW.md "$run_dir/docs/WORKFLOW.md"
copy_file_if_exists docs/ITERATION_HANDOFF_TEMPLATE.md "$run_dir/docs/ITERATION_HANDOFF_TEMPLATE.md"
copy_file_if_exists docs/PRODUCTION_TARGET_GRID.md "$run_dir/docs/PRODUCTION_TARGET_GRID.md"
copy_file_if_exists examples/local34_diag_v1_k10000_1k.yaml "$run_dir/examples/local34_diag_v1_k10000_1k.yaml"
copy_file_if_exists examples/full100k_v1_k50000.yaml "$run_dir/examples/full100k_v1_k50000.yaml"
copy_file_if_exists examples/kmax20_o2_gpu_probe.yaml "$run_dir/examples/kmax20_o2_gpu_probe.yaml"

if [ -n "${CAL_JOB_ID:-}" ]; then
  copy_matching_files "$run_dir/logs/current" "representative calibration job logs" "logs/*${CAL_JOB_ID}*"
fi
if [ -n "${GPU_JOB_ID:-}" ]; then
  copy_matching_files "$run_dir/logs/current" "GPU audit job logs" "logs/*${GPU_JOB_ID}*"
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
  copy_matching_files "$run_dir/logs/current" "collector job logs" "logs/*${SLURM_JOB_ID}*"
fi
if [ "${INCLUDE_ALL_LOGS:-0}" = "1" ]; then
  mkdir -p "$run_dir/logs/all_logs"
  cp -a logs/* "$run_dir/logs/all_logs/" 2>/dev/null || true
fi

gpu_meta_current=0
if [ -n "${GPU_JOB_ID:-}" ] && [ -f results/o2_gpu_audit/gpu_audit_metadata.txt ]; then
  if grep -q "Job ID: ${GPU_JOB_ID}" results/o2_gpu_audit/gpu_audit_metadata.txt; then
    gpu_meta_current=1
  fi
fi

if [ "$gpu_meta_current" -eq 1 ]; then
  copy_file_if_exists results/o2_gpu_audit/gpu_audit_metadata.txt "$run_dir/gpu_audit/gpu_audit_metadata.txt"
  copy_file_if_exists results/o2_gpu_audit/gpu_backend_audit.json "$run_dir/gpu_audit/gpu_backend_audit.json"
  copy_file_if_exists results/o2_gpu_audit/nvidia_smi.txt "$run_dir/gpu_audit/nvidia_smi.txt"
else
  echo "No current-run GPU audit metadata found for GPU_JOB_ID=${GPU_JOB_ID:-unset}; generic GPU audit artifacts were not copied." > "$run_dir/gpu_audit/README.txt"
fi

if [ -n "${GPU_JOB_ID:-}" ] && [ -f "results/o2_gpu_audit/${GPU_JOB_ID}.gpulog" ]; then
  copy_file_if_exists "results/o2_gpu_audit/${GPU_JOB_ID}.gpulog" "$run_dir/gpu_audit/current_gpu_monitor/${GPU_JOB_ID}.gpulog"
else
  echo "No current-run GPU monitor log found for GPU_JOB_ID=${GPU_JOB_ID:-unset}" >> "$run_dir/gpu_audit/README.txt"
fi

collect_accounting "${CAL_JOB_ID:-}" "representative_calibration"
collect_accounting "${GPU_JOB_ID:-}" "gpu_audit"

if [ -f .venv_o2/bin/activate ]; then
  module purge
  module load gcc/14.2.0
  module load python/3.13.1
  source .venv_o2/bin/activate
fi

python scripts/o2/summarize_representative_calibration.py --run-id "$RUN_ID" \
  > "$run_dir/logs/current/summarize_collection_stdout.txt" \
  2> "$run_dir/logs/current/summarize_collection_stderr.txt" || true

tar -czf "$bundle_path" -C "$(dirname "$run_dir")" "$(basename "$run_dir")"

echo "Created representative calibration bundle: $bundle_path"
echo "Summary: $run_dir/summary.md"
