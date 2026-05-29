#!/bin/bash
#SBATCH --job-name=tailbin_resource_calibration
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
  echo "Missing /usr/bin/time; resource calibration requires /usr/bin/time -v." >&2
  exit 3
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_CONFIG="${SMOKE_CONFIG:-examples/kmax2000_cpu_smoke.yaml}"
CAL_CONFIG="${CAL_CONFIG:-examples/o2_resource_calibration.yaml}"
SMOKE_PLAN_LIMIT_BUNDLES="${SMOKE_PLAN_LIMIT_BUNDLES:-1}"
CAL_PLAN_LIMIT_BUNDLES="${CAL_PLAN_LIMIT_BUNDLES:-4}"
CAL_SHARDS="${CAL_SHARDS:-4}"
RUN_TINY_BUILD="${RUN_TINY_BUILD:-1}"
TINY_BUILD_LIMIT_BASE_POINTS="${TINY_BUILD_LIMIT_BASE_POINTS:-1}"

run_dir="results/o2_resource_calibration/${RUN_ID}"
timing_dir="${run_dir}/timing"
json_dir="${run_dir}/json"
plan_dir="${run_dir}/plans"
build_dir="${run_dir}/build"
log_dir="${run_dir}/logs"

mkdir -p "$timing_dir" "$json_dir" "$plan_dir" "$build_dir" "$log_dir"

module purge
module load gcc/14.2.0
module load python/3.13.1

source .venv_o2/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

{
  echo "Run ID: ${RUN_ID}"
  echo "Job ID: ${SLURM_JOB_ID:-unknown}"
  echo "Host: $(hostname)"
  echo "Started: $(date)"
  echo "Repository: ${SLURM_SUBMIT_DIR:-$PWD}"
  echo "Submit git commit: ${SUBMIT_GIT_COMMIT:-unknown}"
  echo "Current git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Smoke config: ${SMOKE_CONFIG}"
  echo "Calibration config: ${CAL_CONFIG}"
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

  {
    echo "$status"
  } > "$status_file"
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

run_timed "estimate_smoke" "${json_dir}/estimate_smoke.json" \
  python -m tailbin_cache.cli estimate --config "$SMOKE_CONFIG"

run_timed "plan_smoke_limit1" "${json_dir}/plan_smoke_limit1.json" \
  python -m tailbin_cache.cli plan --config "$SMOKE_CONFIG" \
    --output-dir "${plan_dir}/smoke_limit1" \
    --limit-bundles "$SMOKE_PLAN_LIMIT_BUNDLES"

run_timed "estimate_calibration" "${json_dir}/estimate_calibration.json" \
  python -m tailbin_cache.cli estimate --config "$CAL_CONFIG"

run_timed "plan_calibration_limit4" "${json_dir}/plan_calibration_limit4.json" \
  python -m tailbin_cache.cli plan --config "$CAL_CONFIG" \
    --output-dir "${plan_dir}/calibration_limit4" \
    --limit-bundles "$CAL_PLAN_LIMIT_BUNDLES"

run_timed "plan_shards_calibration" "${json_dir}/plan_shards_calibration.json" \
  python -m tailbin_cache.cli plan-shards --config "$CAL_CONFIG" \
    --output-dir "${plan_dir}/calibration_shards" \
    --n-shards "$CAL_SHARDS" \
    --limit-bundles "$CAL_PLAN_LIMIT_BUNDLES"

if [ "$RUN_TINY_BUILD" = "1" ]; then
  run_timed "build_tiny_smoke" "${json_dir}/build_tiny_smoke.json" \
    python -m tailbin_cache.cli build-hdf5 --config "$SMOKE_CONFIG" \
      --output "${build_dir}/tiny_smoke_shard0.h5" \
      --limit-base-points "$TINY_BUILD_LIMIT_BASE_POINTS" \
      --n-shards 1 \
      --shard-index 0 \
      --compression gzip \
      --compression-opts 4
else
  echo "Tiny build skipped because RUN_TINY_BUILD=${RUN_TINY_BUILD}" > "${log_dir}/build_tiny_smoke.skipped.txt"
fi

python scripts/o2/summarize_resource_calibration.py --run-id "$RUN_ID" \
  > "${log_dir}/summarize_stdout.txt" \
  2> "${log_dir}/summarize_stderr.txt"

echo "Finished: $(date)" | tee -a "$run_dir/metadata.txt"
echo "Summary: ${run_dir}/summary.md"
