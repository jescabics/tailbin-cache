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
CAL_RUN_SHARD_PLAN="${CAL_RUN_SHARD_PLAN:-0}"
CAL_SHARDS="${CAL_SHARDS:-8}"
CAL_BUILD_STAGE="${CAL_BUILD_STAGE:-easy_smoke}"

case "$CAL_BUILD_STAGE" in
  easy_smoke)
    CAL_BUILD_SAMPLE_BASE_POINTS="${CAL_BUILD_SAMPLE_BASE_POINTS:-1}"
    CAL_BUILD_SAMPLE_STRATEGY="${CAL_BUILD_SAMPLE_STRATEGY:-easy_first}"
    ;;
  stratified_probe)
    CAL_BUILD_SAMPLE_BASE_POINTS="${CAL_BUILD_SAMPLE_BASE_POINTS:-8}"
    CAL_BUILD_SAMPLE_STRATEGY="${CAL_BUILD_SAMPLE_STRATEGY:-representative_stratified}"
    ;;
  representative_sample)
    CAL_BUILD_SAMPLE_BASE_POINTS="${CAL_BUILD_SAMPLE_BASE_POINTS:-40}"
    CAL_BUILD_SAMPLE_STRATEGY="${CAL_BUILD_SAMPLE_STRATEGY:-representative_hard}"
    ;;
  *)
    echo "Unknown CAL_BUILD_STAGE=${CAL_BUILD_STAGE}; expected easy_smoke, stratified_probe, or representative_sample." >&2
    exit 2
    ;;
esac

PLAN_PARTITION="${PLAN_PARTITION:-short}"
PLAN_TIME="${PLAN_TIME:-2:00:00}"
PLAN_MEM="${PLAN_MEM:-4G}"
PLAN_CPUS="${PLAN_CPUS:-1}"

RUN_GPU_BUILD="${RUN_GPU_BUILD:-1}"
BUILD_PARTITION="${BUILD_PARTITION:-gpu_quad}"
BUILD_TIME="${BUILD_TIME:-1:00:00}"
BUILD_MEM="${BUILD_MEM:-16G}"
BUILD_CPUS="${BUILD_CPUS:-4}"
BUILD_GRES="${BUILD_GRES:-gpu:1}"
BUILD_GPU_CONSTRAINT="${BUILD_GPU_CONSTRAINT-gpu_doublep}"
CAL_BUILD_TIMEOUT_SECONDS="${CAL_BUILD_TIMEOUT_SECONDS:-600}"
CAL_MAX_SECONDS_PER_BASE_POINT="${CAL_MAX_SECONDS_PER_BASE_POINT:-300}"
CAL_WARN_ONLY_TIMEOUT="${CAL_WARN_ONLY_TIMEOUT:-0}"

RUN_GPU_AUDIT="${RUN_GPU_AUDIT:-1}"
GPU_PARTITION="${GPU_PARTITION:-gpu_quad}"
GPU_TIME="${GPU_TIME:-1:00:00}"
GPU_MEM="${GPU_MEM:-16G}"
GPU_CPUS="${GPU_CPUS:-4}"
GPU_GRES="${GPU_GRES:-gpu:1}"
GPU_CONSTRAINT="${GPU_CONSTRAINT-gpu_doublep}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
GPU_AUDIT_SCRIPT="${GPU_AUDIT_SCRIPT:-examples/gpu_backend_audit.py}"
GPU_AUDIT_CONFIG="${GPU_AUDIT_CONFIG:-examples/kmax20_o2_gpu_probe.yaml}"

COLLECT_PARTITION="${COLLECT_PARTITION:-short}"
COLLECT_TIME="${COLLECT_TIME:-0:30:00}"
COLLECT_MEM="${COLLECT_MEM:-1G}"
COLLECT_CPUS="${COLLECT_CPUS:-1}"

echo "Tailbin O2 staged representative calibration submitter"
echo "Repository: $repo_root"
echo "Git commit: $git_commit"
echo "Run ID: $run_id"
echo "Config: $CAL_CONFIG"
echo "Build stage: $CAL_BUILD_STAGE"
echo "Build sample points: $CAL_BUILD_SAMPLE_BASE_POINTS"
echo "Build sample strategy: $CAL_BUILD_SAMPLE_STRATEGY"
echo "Run shard plan: $CAL_RUN_SHARD_PLAN"
echo "Run GPU build: $RUN_GPU_BUILD"
echo "Run GPU audit: $RUN_GPU_AUDIT"
echo "This submits resource calibration only, not production."
echo

plan_job_raw="$(sbatch --parsable \
  --job-name=tailbin_representative_plan \
  --partition="$PLAN_PARTITION" \
  --time="$PLAN_TIME" \
  --cpus-per-task="$PLAN_CPUS" \
  --mem="$PLAN_MEM" \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --export="ALL,RUN_ID=${run_id},SUBMIT_GIT_COMMIT=${git_commit},CAL_CONFIG=${CAL_CONFIG},CAL_FULL_PLAN=${CAL_FULL_PLAN},CAL_RUN_SHARD_PLAN=${CAL_RUN_SHARD_PLAN},CAL_SHARDS=${CAL_SHARDS},CAL_BUILD_STAGE=${CAL_BUILD_STAGE},CAL_BUILD_SAMPLE_BASE_POINTS=${CAL_BUILD_SAMPLE_BASE_POINTS},CAL_BUILD_SAMPLE_STRATEGY=${CAL_BUILD_SAMPLE_STRATEGY}" \
  scripts/o2/representative_calibration.sbatch)"
plan_jobid="${plan_job_raw%%;*}"

build_args=(
  --parsable
  --job-name=tailbin_representative_build
  --partition="$BUILD_PARTITION"
  --time="$BUILD_TIME"
  --cpus-per-task="$BUILD_CPUS"
  --mem="$BUILD_MEM"
  --dependency="afterok:${plan_jobid}"
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
  --export="ALL,RUN_ID=${run_id},SUBMIT_GIT_COMMIT=${git_commit},CAL_CONFIG=${CAL_CONFIG},CAL_BUILD_STAGE=${CAL_BUILD_STAGE},RUN_GPU_BUILD=${RUN_GPU_BUILD},CUDA_MODULE=${CUDA_MODULE},CAL_BUILD_TIMEOUT_SECONDS=${CAL_BUILD_TIMEOUT_SECONDS},CAL_MAX_SECONDS_PER_BASE_POINT=${CAL_MAX_SECONDS_PER_BASE_POINT},CAL_WARN_ONLY_TIMEOUT=${CAL_WARN_ONLY_TIMEOUT}"
)
if [ "$RUN_GPU_BUILD" = "1" ]; then
  build_args+=(--gres="$BUILD_GRES")
  if [ -n "$BUILD_GPU_CONSTRAINT" ]; then
    build_args+=(--constraint="$BUILD_GPU_CONSTRAINT")
  fi
fi
build_job_raw="$(sbatch "${build_args[@]}" scripts/o2/representative_build.sbatch)"
build_jobid="${build_job_raw%%;*}"

gpu_audit_jobid=""
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
  gpu_audit_jobid="${gpu_job_raw%%;*}"
fi

dependency="afterany:${plan_jobid}:${build_jobid}"
if [ -n "$gpu_audit_jobid" ]; then
  dependency="${dependency}:${gpu_audit_jobid}"
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
  --export="ALL,RUN_ID=${run_id},PLAN_JOB_ID=${plan_jobid},BUILD_JOB_ID=${build_jobid},CAL_JOB_ID=${plan_jobid},GPU_AUDIT_JOB_ID=${gpu_audit_jobid},GPU_JOB_ID=${gpu_audit_jobid},SUBMIT_GIT_COMMIT=${git_commit},CAL_CONFIG=${CAL_CONFIG}" \
  scripts/o2/collect_representative_calibration.sbatch)"
collect_jobid="${collect_job_raw%%;*}"

echo "Submitted jobs:"
echo "  Plan/select:          $plan_jobid"
echo "  Representative build: $build_jobid"
if [ -n "$gpu_audit_jobid" ]; then
  echo "  GPU audit:            $gpu_audit_jobid"
else
  echo "  GPU audit:            skipped (RUN_GPU_AUDIT=0)"
fi
echo "  Result collect:       $collect_jobid"
echo
echo "Monitor:"
echo '  squeue -u "$USER"'
echo "  squeue -j $plan_jobid"
echo "  squeue -j $build_jobid"
if [ -n "$gpu_audit_jobid" ]; then
  echo "  squeue -j $gpu_audit_jobid"
fi
echo "  squeue -j $collect_jobid"
echo "  sacct -j ${plan_jobid},${build_jobid}${gpu_audit_jobid:+,${gpu_audit_jobid}},${collect_jobid} --format=JobId,JobName,Partition,NCPUS,State,ReqMem,MaxRSS,Elapsed,TimeLimit,ExitCode"
echo "  tail -f logs/*${build_jobid}*"
echo
echo "After completion, inspect:"
echo "  results/o2_representative_calibration/${run_id}/summary.md"
echo "  results/o2_representative_calibration/${run_id}/summary.json"
echo "  results/o2_representative_calibration/${run_id}/progress/build_representative_sample.progress.jsonl"
echo "  results/tailbin_o2_representative_calibration_${run_id}.tgz"
