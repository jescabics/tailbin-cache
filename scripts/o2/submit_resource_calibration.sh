#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

mkdir -p logs outputs results tasks

if [ ! -f .venv_o2/bin/activate ]; then
  echo "Missing .venv_o2. First run: bash scripts/o2/submit_setup_env.sh" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [ -n "${RUN_ID:-}" ]; then
  run_id="$RUN_ID"
elif [ -n "${RUN_LABEL:-}" ]; then
  run_id="${RUN_LABEL}_${timestamp}"
else
  run_id="$timestamp"
fi
run_id="${run_id//[^A-Za-z0-9_.-]/_}"

git_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

CAL_PARTITION="${CAL_PARTITION:-short}"
CAL_TIME="${CAL_TIME:-2:00:00}"
CAL_MEM="${CAL_MEM:-4G}"
CAL_CPUS="${CAL_CPUS:-1}"

RUN_GPU_AUDIT="${RUN_GPU_AUDIT:-1}"
GPU_PARTITION="${GPU_PARTITION:-gpu_quad}"
GPU_TIME="${GPU_TIME:-1:00:00}"
GPU_MEM="${GPU_MEM:-16G}"
GPU_CPUS="${GPU_CPUS:-4}"
GPU_GRES="${GPU_GRES:-gpu:1}"
GPU_CONSTRAINT="${GPU_CONSTRAINT-gpu_doublep}"
CUDA_MODULE="${CUDA_MODULE-cuda/12.8}"
GPU_AUDIT_SCRIPT="${GPU_AUDIT_SCRIPT:-examples/gpu_backend_audit.py}"
GPU_AUDIT_CONFIG="${GPU_AUDIT_CONFIG:-examples/kmax20_o2_gpu_probe.yaml}"

COLLECT_PARTITION="${COLLECT_PARTITION:-short}"
COLLECT_TIME="${COLLECT_TIME:-0:30:00}"
COLLECT_MEM="${COLLECT_MEM:-1G}"
COLLECT_CPUS="${COLLECT_CPUS:-1}"

SMOKE_CONFIG="${SMOKE_CONFIG:-examples/kmax2000_cpu_smoke.yaml}"
CAL_CONFIG="${CAL_CONFIG:-examples/o2_resource_calibration.yaml}"
SMOKE_PLAN_LIMIT_BUNDLES="${SMOKE_PLAN_LIMIT_BUNDLES:-1}"
CAL_PLAN_LIMIT_BUNDLES="${CAL_PLAN_LIMIT_BUNDLES:-4}"
CAL_SHARDS="${CAL_SHARDS:-4}"
RUN_TINY_BUILD="${RUN_TINY_BUILD:-1}"
TINY_BUILD_LIMIT_BASE_POINTS="${TINY_BUILD_LIMIT_BASE_POINTS:-1}"

echo "Tailbin O2 resource calibration submitter"
echo "Repository: $repo_root"
echo "Git commit: $git_commit"
echo "Run ID: $run_id"
echo
echo "CPU calibration resources: partition=$CAL_PARTITION time=$CAL_TIME mem=$CAL_MEM cpus=$CAL_CPUS"
echo "GPU audit enabled: $RUN_GPU_AUDIT"
if [ "$RUN_GPU_AUDIT" = "1" ]; then
  echo "GPU audit resources: partition=$GPU_PARTITION time=$GPU_TIME mem=$GPU_MEM cpus=$GPU_CPUS gres=$GPU_GRES constraint=${GPU_CONSTRAINT:-none} cuda=${CUDA_MODULE:-none}"
fi
echo "Collector resources: partition=$COLLECT_PARTITION time=$COLLECT_TIME mem=$COLLECT_MEM cpus=$COLLECT_CPUS"
echo

cal_job_raw="$(sbatch --parsable \
  --job-name=tailbin_resource_calibration \
  --partition="$CAL_PARTITION" \
  --time="$CAL_TIME" \
  --cpus-per-task="$CAL_CPUS" \
  --mem="$CAL_MEM" \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --export="ALL,RUN_ID=${run_id},SUBMIT_GIT_COMMIT=${git_commit},SMOKE_CONFIG=${SMOKE_CONFIG},CAL_CONFIG=${CAL_CONFIG},SMOKE_PLAN_LIMIT_BUNDLES=${SMOKE_PLAN_LIMIT_BUNDLES},CAL_PLAN_LIMIT_BUNDLES=${CAL_PLAN_LIMIT_BUNDLES},CAL_SHARDS=${CAL_SHARDS},RUN_TINY_BUILD=${RUN_TINY_BUILD},TINY_BUILD_LIMIT_BASE_POINTS=${TINY_BUILD_LIMIT_BASE_POINTS}" \
  scripts/o2/resource_calibration.sbatch)"
cal_jobid="${cal_job_raw%%;*}"

gpu_jobid=""
if [ "$RUN_GPU_AUDIT" = "1" ]; then
  gpu_args=(
    --parsable
    --job-name=tailbin_gpu_calibration_audit
    --partition="$GPU_PARTITION"
    --time="$GPU_TIME"
    --cpus-per-task="$GPU_CPUS"
    --mem="$GPU_MEM"
    --gres="$GPU_GRES"
    --output=logs/%x_%j.out
    --error=logs/%x_%j.err
    --export="ALL,GPU_AUDIT_SCRIPT=${GPU_AUDIT_SCRIPT},GPU_AUDIT_CONFIG=${GPU_AUDIT_CONFIG},CUDA_MODULE=${CUDA_MODULE}"
  )
  if [ -n "$GPU_CONSTRAINT" ]; then
    gpu_args+=(--constraint="$GPU_CONSTRAINT")
  fi
  gpu_job_raw="$(sbatch "${gpu_args[@]}" scripts/o2/gpu_audit.sbatch)"
  gpu_jobid="${gpu_job_raw%%;*}"
fi

dependency="afterany:${cal_jobid}"
if [ -n "$gpu_jobid" ]; then
  dependency="${dependency}:${gpu_jobid}"
fi

collect_job_raw="$(sbatch --parsable \
  --job-name=tailbin_collect_resource_cal \
  --partition="$COLLECT_PARTITION" \
  --time="$COLLECT_TIME" \
  --cpus-per-task="$COLLECT_CPUS" \
  --mem="$COLLECT_MEM" \
  --dependency="$dependency" \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --export="ALL,RUN_ID=${run_id},CAL_JOB_ID=${cal_jobid},GPU_JOB_ID=${gpu_jobid},SUBMIT_GIT_COMMIT=${git_commit},SMOKE_CONFIG=${SMOKE_CONFIG},CAL_CONFIG=${CAL_CONFIG}" \
  scripts/o2/collect_resource_calibration.sbatch)"
collect_jobid="${collect_job_raw%%;*}"

echo "Submitted jobs:"
echo "  Resource calibration: $cal_jobid"
if [ -n "$gpu_jobid" ]; then
  echo "  GPU audit:            $gpu_jobid"
else
  echo "  GPU audit:            skipped (RUN_GPU_AUDIT=0)"
fi
echo "  Result collect:       $collect_jobid"
echo
echo "Monitor:"
echo '  squeue -u "$USER"'
echo "  squeue -j $cal_jobid"
if [ -n "$gpu_jobid" ]; then
  echo "  squeue -j $gpu_jobid"
fi
echo "  squeue -j $collect_jobid"
echo "  O2_jobs_report -j $cal_jobid"
if [ -n "$gpu_jobid" ]; then
  echo "  O2_jobs_report -j $gpu_jobid"
fi
echo "  O2_jobs_report -j $collect_jobid"
echo "  sacct -j $cal_jobid --format=JobId,JobName,Partition,NCPUS,State,ReqMem,MaxRSS,Elapsed,TimeLimit,ExitCode"
echo
echo "After completion, inspect:"
echo "  results/o2_resource_calibration/${run_id}/summary.md"
echo "  results/o2_resource_calibration/${run_id}/summary.json"
echo "  results/tailbin_o2_resource_calibration_${run_id}.tgz"
echo
echo "This workflow is calibration only and does not launch full production."
