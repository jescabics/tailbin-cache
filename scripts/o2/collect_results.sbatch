#!/bin/bash
#SBATCH --job-name=tailbin_collect_smoke_audit
#SBATCH -p short
#SBATCH -t 0:30:00
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs outputs results tasks

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
bundle_name="tailbin_o2_smoke_audit_${timestamp}"
bundle_dir="results/${bundle_name}"
bundle_path="results/${bundle_name}.tgz"

mkdir -p "$bundle_dir"/{accounting,docs,examples,gpu_logs}
shopt -s nullglob

copy_file_if_exists() {
  local src="$1"
  local dest="$2"
  if [ -f "$src" ]; then
    mkdir -p "$(dirname "$dest")"
    cp -a "$src" "$dest"
  else
    echo "missing file: $src" >> "$bundle_dir/missing_files.txt"
  fi
}

copy_dir_if_exists() {
  local src="$1"
  local dest="$2"
  if [ -d "$src" ]; then
    mkdir -p "$(dirname "$dest")"
    cp -a "$src" "$dest"
  else
    echo "missing dir: $src" >> "$bundle_dir/missing_files.txt"
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
    echo "No files matched for $label: $*" >> "$bundle_dir/missing_files.txt"
  fi
}

copy_dir_contents_if_exists() {
  local src="$1"
  local dest="$2"
  if [ -d "$src" ]; then
    mkdir -p "$dest"
    local matches=( "$src"/* )
    if [ "${#matches[@]}" -gt 0 ]; then
      cp -a "${matches[@]}" "$dest/"
    else
      echo "empty dir: $src" >> "$bundle_dir/missing_files.txt"
    fi
  else
    echo "missing dir: $src" >> "$bundle_dir/missing_files.txt"
  fi
}

collect_accounting() {
  local jobid="$1"
  local label="$2"
  local outfile="$bundle_dir/accounting/${label}_${jobid}.txt"

  if [ -z "$jobid" ]; then
    echo "No job ID provided for $label" > "$outfile"
    return 0
  fi

  {
    echo "Accounting for $label job $jobid"
    if command -v O2_jobs_report >/dev/null 2>&1; then
      O2_jobs_report -j "$jobid"
    elif command -v sacct >/dev/null 2>&1; then
      sacct -j "$jobid" --format=JobId,NNodes,Partition,NCPUS,State,ReqMem,MaxRSS,Elapsed,CPUTime,TimeLimit,ExitCode,Start,End
    else
      echo "Neither O2_jobs_report nor sacct is available."
    fi
  } > "$outfile" 2>&1 || true
}

{
  echo "Bundle: $bundle_name"
  echo "Created: $(date)"
  echo "Repository: ${SLURM_SUBMIT_DIR:-$PWD}"
  echo "Submit git commit: ${SUBMIT_GIT_COMMIT:-unknown}"
  echo "Current git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "CPU job ID: ${CPU_JOB_ID:-}"
  echo "GPU job ID: ${GPU_JOB_ID:-}"
  echo "Collector job ID: ${SLURM_JOB_ID:-}"
  echo "CPU smoke config: ${CPU_SMOKE_CONFIG:-examples/kmax2000_cpu_smoke.yaml}"
  echo "GPU audit config: ${GPU_AUDIT_CONFIG:-examples/kmax20_o2_gpu_probe.yaml}"
  echo "INCLUDE_ALL_LOGS: ${INCLUDE_ALL_LOGS:-0}"
} > "$bundle_dir/metadata.txt"

git rev-parse HEAD > "$bundle_dir/git_commit.txt" 2>/dev/null || echo unknown > "$bundle_dir/git_commit.txt"
git status --short --branch -uall > "$bundle_dir/git_status.txt" 2>&1 || true

copy_file_if_exists docs/O2_RUNBOOK.md "$bundle_dir/docs/O2_RUNBOOK.md"
copy_file_if_exists docs/O2_RESOURCE_CALIBRATION.md "$bundle_dir/docs/O2_RESOURCE_CALIBRATION.md"
copy_file_if_exists docs/WORKFLOW.md "$bundle_dir/docs/WORKFLOW.md"
copy_file_if_exists docs/ITERATION_HANDOFF_TEMPLATE.md "$bundle_dir/docs/ITERATION_HANDOFF_TEMPLATE.md"
copy_file_if_exists examples/kmax2000_cpu_smoke.yaml "$bundle_dir/examples/kmax2000_cpu_smoke.yaml"
copy_file_if_exists examples/kmax20_o2_gpu_probe.yaml "$bundle_dir/examples/kmax20_o2_gpu_probe.yaml"
copy_file_if_exists examples/gpu_backend_audit.py "$bundle_dir/examples/gpu_backend_audit.py"

if [ -n "${CPU_JOB_ID:-}" ]; then
  copy_matching_files "$bundle_dir/logs/current" "CPU job logs" "logs/*${CPU_JOB_ID}*"
fi
if [ -n "${GPU_JOB_ID:-}" ]; then
  copy_matching_files "$bundle_dir/logs/current" "GPU job logs" "logs/*${GPU_JOB_ID}*"
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
  copy_matching_files "$bundle_dir/logs/current" "collector job logs" "logs/*${SLURM_JOB_ID}*"
fi
if [ "${INCLUDE_ALL_LOGS:-0}" = "1" ]; then
  copy_dir_if_exists logs "$bundle_dir/logs/all_logs"
fi

copy_dir_contents_if_exists outputs/o2_smoke/plan "$bundle_dir/outputs/o2_smoke/plan"
copy_dir_contents_if_exists results/o2_smoke "$bundle_dir/results/o2_smoke"
copy_file_if_exists results/o2_gpu_audit/gpu_audit_metadata.txt "$bundle_dir/results/o2_gpu_audit/gpu_audit_metadata.txt"
copy_file_if_exists results/o2_gpu_audit/gpu_backend_audit.json "$bundle_dir/results/o2_gpu_audit/gpu_backend_audit.json"
copy_file_if_exists results/o2_gpu_audit/nvidia_smi.txt "$bundle_dir/results/o2_gpu_audit/nvidia_smi.txt"

if [ -n "${GPU_JOB_ID:-}" ] && [ -f "results/o2_gpu_audit/${GPU_JOB_ID}.gpulog" ]; then
  copy_file_if_exists "results/o2_gpu_audit/${GPU_JOB_ID}.gpulog" "$bundle_dir/results/o2_gpu_audit/${GPU_JOB_ID}.gpulog"
  copy_file_if_exists "results/o2_gpu_audit/${GPU_JOB_ID}.gpulog" "$bundle_dir/gpu_logs/current/${GPU_JOB_ID}.gpulog"
else
  echo "No current-run GPU monitor log found for GPU_JOB_ID=${GPU_JOB_ID:-unset}" > "$bundle_dir/gpu_logs/README.txt"
fi

collect_accounting "${CPU_JOB_ID:-}" "cpu_smoke"
collect_accounting "${GPU_JOB_ID:-}" "gpu_audit"

tar -czf "$bundle_path" -C results "$bundle_name"

echo "Created result bundle: $bundle_path"
