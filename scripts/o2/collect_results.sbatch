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
  echo "CPU smoke config: ${CPU_SMOKE_CONFIG:-examples/kmax2000_cpu_smoke.yaml}"
  echo "GPU audit config: ${GPU_AUDIT_CONFIG:-examples/kmax20_o2_gpu_probe.yaml}"
} > "$bundle_dir/metadata.txt"

git rev-parse HEAD > "$bundle_dir/git_commit.txt" 2>/dev/null || echo unknown > "$bundle_dir/git_commit.txt"
git status --short --branch -uall > "$bundle_dir/git_status.txt" 2>&1 || true

copy_file_if_exists docs/O2_RUNBOOK.md "$bundle_dir/docs/O2_RUNBOOK.md"
copy_file_if_exists docs/O2_RESOURCE_CALIBRATION.md "$bundle_dir/docs/O2_RESOURCE_CALIBRATION.md"
copy_file_if_exists docs/WORKFLOW.md "$bundle_dir/docs/WORKFLOW.md"
copy_file_if_exists examples/kmax2000_cpu_smoke.yaml "$bundle_dir/examples/kmax2000_cpu_smoke.yaml"
copy_file_if_exists examples/kmax20_o2_gpu_probe.yaml "$bundle_dir/examples/kmax20_o2_gpu_probe.yaml"
copy_file_if_exists examples/gpu_backend_audit.py "$bundle_dir/examples/gpu_backend_audit.py"

copy_dir_if_exists logs "$bundle_dir/logs"
copy_dir_if_exists outputs/o2_smoke "$bundle_dir/outputs/o2_smoke"
copy_dir_if_exists outputs/o2_gpu_audit "$bundle_dir/outputs/o2_gpu_audit"
copy_dir_if_exists results/o2_smoke "$bundle_dir/results/o2_smoke"
copy_dir_if_exists results/o2_gpu_audit "$bundle_dir/results/o2_gpu_audit"

collect_accounting "${CPU_JOB_ID:-}" "cpu_smoke"
collect_accounting "${GPU_JOB_ID:-}" "gpu_audit"

found_gpu_logs=0
while IFS= read -r -d '' gpulog; do
  found_gpu_logs=1
  cp -a "$gpulog" "$bundle_dir/gpu_logs/$(basename "$gpulog")"
done < <(find . -path "./results/tailbin_o2_smoke_audit_*" -prune -o -type f -name "*.gpulog" -print0)

if [ "$found_gpu_logs" -eq 0 ]; then
  echo "No *.gpulog files found." > "$bundle_dir/gpu_logs/README.txt"
fi

tar -czf "$bundle_path" -C results "$bundle_name"

echo "Created result bundle: $bundle_path"
