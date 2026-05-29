#!/bin/bash
#SBATCH --job-name=tailbin_representative_plan
#SBATCH -p short
#SBATCH -t 2:00:00
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs outputs results tasks

if [ ! -f .venv_o2/bin/activate ]; then
  echo "Missing .venv_o2. First run: bash scripts/o2/submit_setup_env.sh" >&2
  exit 2
fi

if [ ! -x /usr/bin/time ]; then
  echo "Missing /usr/bin/time; representative calibration requires /usr/bin/time -v." >&2
  exit 3
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CAL_CONFIG="${CAL_CONFIG:-examples/local34_diag_v1_k10000_1k.yaml}"
CAL_FULL_PLAN="${CAL_FULL_PLAN:-1}"
CAL_RUN_SHARD_PLAN="${CAL_RUN_SHARD_PLAN:-0}"
CAL_SHARDS="${CAL_SHARDS:-8}"
CAL_BUILD_STAGE="${CAL_BUILD_STAGE:-easy_smoke}"
CAL_BUILD_SAMPLE_BASE_POINTS="${CAL_BUILD_SAMPLE_BASE_POINTS:-1}"
CAL_BUILD_SAMPLE_STRATEGY="${CAL_BUILD_SAMPLE_STRATEGY:-easy_first}"
CAL_STRATIFIED_SAMPLE_BASE_POINTS="${CAL_STRATIFIED_SAMPLE_BASE_POINTS:-8}"
CAL_REPRESENTATIVE_SAMPLE_BASE_POINTS="${CAL_REPRESENTATIVE_SAMPLE_BASE_POINTS:-40}"
stratified_sample_size="$CAL_STRATIFIED_SAMPLE_BASE_POINTS"
representative_sample_size="$CAL_REPRESENTATIVE_SAMPLE_BASE_POINTS"
if [ "$CAL_BUILD_STAGE" = "stratified_probe" ]; then
  stratified_sample_size="$CAL_BUILD_SAMPLE_BASE_POINTS"
fi
if [ "$CAL_BUILD_STAGE" = "representative_sample" ]; then
  representative_sample_size="$CAL_BUILD_SAMPLE_BASE_POINTS"
fi

run_dir="results/o2_representative_calibration/${RUN_ID}"
timing_dir="${run_dir}/timing"
json_dir="${run_dir}/json"
plan_dir="${run_dir}/plans"
sample_dir="${run_dir}/sample"
log_dir="${run_dir}/logs"

mkdir -p "$timing_dir" "$json_dir" "$plan_dir" "$sample_dir" "$log_dir"

module purge
module load gcc/14.2.0
module load python/3.13.1

source .venv_o2/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

{
  echo "Representative calibration plan/select stage"
  echo "Run ID: ${RUN_ID}"
  echo "Job ID: ${SLURM_JOB_ID:-unknown}"
  echo "Host: $(hostname)"
  echo "Started: $(date)"
  echo "Repository: ${SLURM_SUBMIT_DIR:-$PWD}"
  echo "Submit git commit: ${SUBMIT_GIT_COMMIT:-unknown}"
  echo "Current git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Calibration config: ${CAL_CONFIG}"
  echo "Full plan enabled: ${CAL_FULL_PLAN}"
  echo "Run shard plan: ${CAL_RUN_SHARD_PLAN}"
  echo "Shard count if requested: ${CAL_SHARDS}"
  echo "Build stage requested: ${CAL_BUILD_STAGE}"
  echo "Requested build sample size: ${CAL_BUILD_SAMPLE_BASE_POINTS}"
  echo "Requested build sample strategy: ${CAL_BUILD_SAMPLE_STRATEGY}"
  echo "SLURM partition: ${SLURM_JOB_PARTITION:-unknown}"
  echo "SLURM CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
  echo "SLURM mem per node: ${SLURM_MEM_PER_NODE:-unknown}"
  echo "SLURM time limit: ${SLURM_TIMELIMIT:-unknown}"
  echo "Threads: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS}"
} | tee "$run_dir/metadata.txt"

module list 2>&1 | tee "$run_dir/module_list.txt"
which python | tee "$run_dir/python_path.txt"
python --version 2>&1 | tee "$run_dir/python_version.txt"
python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >=3.10 is required; found " + sys.version)
PY

git status --short --branch -uall > "$run_dir/git_status.txt" 2>&1 || true
git rev-parse HEAD > "$run_dir/git_commit.txt" 2>/dev/null || echo unknown > "$run_dir/git_commit.txt"

run_timed() {
  local name="$1"
  local stdout_file="$2"
  shift 2

  local timing_file="${timing_dir}/${name}.time.txt"
  local stderr_file="${log_dir}/${name}.stderr.txt"
  local status_file="${timing_dir}/${name}.status.txt"
  local command_file="${timing_dir}/${name}.command.txt"

  mkdir -p "$(dirname "$stdout_file")"
  {
    echo "name: ${name}"
    echo "started: $(date)"
    echo "command:"
    printf '  %q' "$@"
    echo
  } > "$command_file"

  echo "Running ${name}"
  set +e
  /usr/bin/time -v -o "$timing_file" "$@" > "$stdout_file" 2> "$stderr_file"
  local status=$?
  set -e

  echo "$status" > "$status_file"
  {
    echo "finished: $(date)"
    echo "exit_status: ${status}"
  } >> "$command_file"

  if [ "$status" -ne 0 ]; then
    echo "Command failed: ${name}" >&2
    echo "See: ${stderr_file}" >&2
    exit "$status"
  fi
}

run_timed "cli_help" "${log_dir}/cli_help.txt" \
  python -m tailbin_cache.cli --help

run_timed "estimate_grid_b" "${json_dir}/estimate_grid_b.json" \
  python -m tailbin_cache.cli estimate --config "$CAL_CONFIG"

if [ "$CAL_FULL_PLAN" = "1" ]; then
  run_timed "plan_grid_b_full" "${json_dir}/plan_grid_b_full.json" \
    python -m tailbin_cache.cli plan --config "$CAL_CONFIG" \
      --output-dir "${plan_dir}/full_plan"
else
  echo "CAL_FULL_PLAN must be 1 for representative calibration selection." >&2
  exit 2
fi

if [ "$CAL_RUN_SHARD_PLAN" = "1" ]; then
  run_timed "plan_shards_grid_b_from_full_plan" "${json_dir}/plan_shards_grid_b_from_full_plan.json" \
    python -m tailbin_cache.cli plan-shards \
      --from-plan-dir "${plan_dir}/full_plan" \
      --output-dir "${plan_dir}/full_shards" \
      --n-shards "$CAL_SHARDS"
else
  echo "Shard planning skipped by CAL_RUN_SHARD_PLAN=${CAL_RUN_SHARD_PLAN}" | tee "${plan_dir}/full_shards.SKIPPED.txt"
fi

run_timed "select_easy_first" "${json_dir}/select_easy_first.json" \
  python -m tailbin_cache.cli select-representative \
    --config "$CAL_CONFIG" \
    --plan-dir "${plan_dir}/full_plan" \
    --output-csv "${sample_dir}/selected_easy_first.csv" \
    --output-json "${sample_dir}/selected_easy_first.json" \
    --sample-size 1 \
    --strategy easy_first

run_timed "select_stratified_8" "${json_dir}/select_stratified_8.json" \
  python -m tailbin_cache.cli select-representative \
    --config "$CAL_CONFIG" \
    --plan-dir "${plan_dir}/full_plan" \
    --output-csv "${sample_dir}/selected_stratified_8.csv" \
    --output-json "${sample_dir}/selected_stratified_8.json" \
    --sample-size "$stratified_sample_size" \
    --strategy representative_stratified

run_timed "select_representative_40" "${json_dir}/select_representative_40.json" \
  python -m tailbin_cache.cli select-representative \
    --config "$CAL_CONFIG" \
    --plan-dir "${plan_dir}/full_plan" \
    --output-csv "${sample_dir}/selected_representative_40.csv" \
    --output-json "${sample_dir}/selected_representative_40.json" \
    --sample-size "$representative_sample_size" \
    --strategy representative_hard

case "$CAL_BUILD_STAGE" in
  easy_smoke)
    cp "${sample_dir}/selected_easy_first.csv" "${sample_dir}/selected_base_points.csv"
    cp "${sample_dir}/selected_easy_first.json" "${sample_dir}/selected_base_points.json"
    ;;
  stratified_probe)
    cp "${sample_dir}/selected_stratified_8.csv" "${sample_dir}/selected_base_points.csv"
    cp "${sample_dir}/selected_stratified_8.json" "${sample_dir}/selected_base_points.json"
    ;;
  representative_sample)
    cp "${sample_dir}/selected_representative_40.csv" "${sample_dir}/selected_base_points.csv"
    cp "${sample_dir}/selected_representative_40.json" "${sample_dir}/selected_base_points.json"
    ;;
  *)
    echo "Unknown CAL_BUILD_STAGE=${CAL_BUILD_STAGE}; expected easy_smoke, stratified_probe, or representative_sample." >&2
    exit 2
    ;;
esac

{
  echo "selected_manifest_csv=${sample_dir}/selected_base_points.csv"
  echo "selected_manifest_json=${sample_dir}/selected_base_points.json"
  echo "selected_stage=${CAL_BUILD_STAGE}"
} > "${sample_dir}/selected_manifest_for_build.env"

python scripts/o2/summarize_representative_calibration.py --run-id "$RUN_ID" \
  > "${log_dir}/summarize_plan_stdout.txt" \
  2> "${log_dir}/summarize_plan_stderr.txt" || true

echo "Finished: $(date)" | tee -a "$run_dir/metadata.txt"
echo "Selected manifest: ${sample_dir}/selected_base_points.csv"
echo "Summary so far: ${run_dir}/summary.md"
