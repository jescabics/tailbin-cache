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
  run_id="representative_${timestamp}"
fi
run_id="${run_id//[^A-Za-z0-9_.-]/_}"

git_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

CAL_CONFIG="${CAL_CONFIG:-examples/local34_diag_v1_k10000_1k.yaml}"
CAL_FULL_PLAN="${CAL_FULL_PLAN:-1}"
CAL_SHARDS="${CAL_SHARDS:-8}"
CAL_BUILD_SAMPLE_BASE_POINTS="${CAL_BUILD_SAMPLE_BASE_POINTS:-40}"
CAL_BUILD_SAMPLE_STRATEGY="${CAL_BUILD_SAMPLE_STRATEGY:-representative_hard}"

CAL_PARTITION="${CAL_PARTITION:-short}"
CAL_TIME="${CAL_TIME:-12:00:00}"
CAL_MEM="${CAL_MEM:-8G}"
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

echo "Tailbin O2 representative calibration submitter"
echo "Repository: $repo_root"
echo "Git commit: $git_commit"
echo "Run ID: $run_id"
echo "Config: $CAL_CONFIG"
echo "Sample base points: $CAL_BUILD_SAMPLE_BASE_POINTS"
echo "Strategy: $CAL_BUILD_SAMPLE_STRATEGY"
echo "This submits resource calibration only, not production."
echo

cal_job_raw="$(sbatch --parsable \
  --job-name=tailbin_representative_cal \
  --partition="$CAL_PARTITION" \
  --time="$CAL_TIME" \
  --cpus-per-task="$CAL_CPUS" \
  --mem="$CAL_MEM" \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --export="ALL,RUN_ID=${run_id},SUBMIT_GIT_COMMIT=${git_commit},CAL_CONFIG=${CAL_CONFIG},CAL_FULL_PLAN=${CAL_FULL_PLAN},CAL_SHARDS=${CAL_SHARDS},CAL_BUILD_SAMPLE_BASE_POINTS=${CAL_BUILD_SAMPLE_BASE_POINTS},CAL_BUILD_SAMPLE_STRATEGY=${CAL_BUILD_SAMPLE_STRATEGY}" \
  scripts/o2/representative_calibration.sbatch)"
cal_jobid="${cal_job_raw%%;*}"

gpu_jobid=""
if [ "$RUN_GPU_AUDIT" = "1" ]; then
  gpu_args=(
    --parsable
    --job-name=tailbin_gpu_representative_audit
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
  --job-name=tailbin_collect_representative_cal \
  --partition="$COLLECT_PARTITION" \
  --time="$COLLECT_TIME" \
  --cpus-per-task="$COLLECT_CPUS" \
  --mem="$COLLECT_MEM" \
  --dependency="$dependency" \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --export="ALL,RUN_ID=${run_id},CAL_JOB_ID=${cal_jobid},GPU_JOB_ID=${gpu_jobid},SUBMIT_GIT_COMMIT=${git_commit},CAL_CONFIG=${CAL_CONFIG}" \
  scripts/o2/collect_representative_calibration.sbatch)"
collect_jobid="${collect_job_raw%%;*}"

echo "Submitted jobs:"
echo "  Representative calibration: $cal_jobid"
if [ -n "$gpu_jobid" ]; then
  echo "  GPU audit:                  $gpu_jobid"
else
  echo "  GPU audit:                  skipped (RUN_GPU_AUDIT=0)"
fi
echo "  Result collect:             $collect_jobid"
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
echo
echo "After completion, inspect:"
echo "  results/o2_representative_calibration/${run_id}/summary.md"
echo "  results/o2_representative_calibration/${run_id}/summary.json"
echo "  results/tailbin_o2_representative_calibration_${run_id}.tgz"
