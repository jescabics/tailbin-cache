#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

mkdir -p logs outputs results tasks

git_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

CPU_PARTITION="${CPU_PARTITION:-short}"
CPU_TIME="${CPU_TIME:-1:00:00}"
CPU_MEM="${CPU_MEM:-2G}"
CPU_CPUS="${CPU_CPUS:-1}"

GPU_PARTITION="${GPU_PARTITION:-gpu_quad}"
GPU_TIME="${GPU_TIME:-2:00:00}"
GPU_MEM="${GPU_MEM:-16G}"
GPU_CPUS="${GPU_CPUS:-4}"
GPU_GRES="${GPU_GRES:-gpu:1}"
GPU_CONSTRAINT="${GPU_CONSTRAINT-gpu_doublep}"

COLLECT_PARTITION="${COLLECT_PARTITION:-short}"
COLLECT_TIME="${COLLECT_TIME:-0:30:00}"
COLLECT_MEM="${COLLECT_MEM:-1G}"
COLLECT_CPUS="${COLLECT_CPUS:-1}"

CPU_SMOKE_CONFIG="${CPU_SMOKE_CONFIG:-examples/kmax2000_cpu_smoke.yaml}"
GPU_AUDIT_SCRIPT="${GPU_AUDIT_SCRIPT:-examples/gpu_backend_audit.py}"
GPU_AUDIT_CONFIG="${GPU_AUDIT_CONFIG:-examples/kmax20_o2_gpu_probe.yaml}"

echo "Tailbin O2 smoke/audit submitter"
echo "Repository: $repo_root"
echo "Git commit: $git_commit"
echo

cpu_job_raw="$(sbatch --parsable \
  --job-name=tailbin_cpu_smoke \
  --partition="$CPU_PARTITION" \
  --time="$CPU_TIME" \
  --cpus-per-task="$CPU_CPUS" \
  --mem="$CPU_MEM" \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --export="ALL,CPU_SMOKE_CONFIG=${CPU_SMOKE_CONFIG}" \
  scripts/o2/cpu_smoke.sbatch)"
cpu_jobid="${cpu_job_raw%%;*}"

gpu_args=(
  --parsable
  --job-name=tailbin_gpu_audit
  --partition="$GPU_PARTITION"
  --time="$GPU_TIME"
  --cpus-per-task="$GPU_CPUS"
  --mem="$GPU_MEM"
  --gres="$GPU_GRES"
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
  --export="ALL,GPU_AUDIT_SCRIPT=${GPU_AUDIT_SCRIPT},GPU_AUDIT_CONFIG=${GPU_AUDIT_CONFIG}"
)
if [ -n "$GPU_CONSTRAINT" ]; then
  gpu_args+=(--constraint="$GPU_CONSTRAINT")
fi

gpu_job_raw="$(sbatch "${gpu_args[@]}" scripts/o2/gpu_audit.sbatch)"
gpu_jobid="${gpu_job_raw%%;*}"

collect_job_raw="$(sbatch --parsable \
  --job-name=tailbin_collect_smoke_audit \
  --partition="$COLLECT_PARTITION" \
  --time="$COLLECT_TIME" \
  --cpus-per-task="$COLLECT_CPUS" \
  --mem="$COLLECT_MEM" \
  --dependency="afterany:${cpu_jobid}:${gpu_jobid}" \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --export="ALL,CPU_JOB_ID=${cpu_jobid},GPU_JOB_ID=${gpu_jobid},SUBMIT_GIT_COMMIT=${git_commit},CPU_SMOKE_CONFIG=${CPU_SMOKE_CONFIG},GPU_AUDIT_CONFIG=${GPU_AUDIT_CONFIG}" \
  scripts/o2/collect_results.sbatch)"
collect_jobid="${collect_job_raw%%;*}"

echo "Submitted jobs:"
echo "  CPU smoke:      $cpu_jobid"
echo "  GPU audit:      $gpu_jobid"
echo "  Result collect: $collect_jobid"
echo
echo "Monitor:"
echo '  squeue -u "$USER"'
echo "  squeue -j $cpu_jobid"
echo "  squeue -j $gpu_jobid"
echo "  squeue -j $collect_jobid"
echo "  O2_jobs_report -j $cpu_jobid"
echo "  O2_jobs_report -j $gpu_jobid"
echo "  O2_jobs_report -j $collect_jobid"
echo
echo "Results will be collected after both smoke/audit jobs finish."
