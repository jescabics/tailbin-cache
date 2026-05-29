#!/bin/bash
#SBATCH --job-name=tailbin_representative_build
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -c 4
#SBATCH --mem=16G
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
  echo "Missing /usr/bin/time; representative build requires /usr/bin/time -v." >&2
  exit 3
fi

RUN_ID="${RUN_ID:?RUN_ID is required}"
CAL_CONFIG="${CAL_CONFIG:-examples/local34_diag_v1_k10000_1k.yaml}"
CAL_BUILD_STAGE="${CAL_BUILD_STAGE:-easy_smoke}"
RUN_GPU_BUILD="${RUN_GPU_BUILD:-1}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
CAL_BUILD_TIMEOUT_SECONDS="${CAL_BUILD_TIMEOUT_SECONDS:-600}"
CAL_MAX_SECONDS_PER_BASE_POINT="${CAL_MAX_SECONDS_PER_BASE_POINT:-300}"
CAL_WARN_ONLY_TIMEOUT="${CAL_WARN_ONLY_TIMEOUT:-0}"

run_dir="results/o2_representative_calibration/${RUN_ID}"
timing_dir="${run_dir}/timing"
json_dir="${run_dir}/json"
build_dir="${run_dir}/build"
progress_dir="${run_dir}/progress"
log_dir="${run_dir}/logs"
gpu_build_dir="${run_dir}/gpu_build"
sample_dir="${run_dir}/sample"

mkdir -p "$timing_dir" "$json_dir" "$build_dir" "$progress_dir" "$log_dir" "$gpu_build_dir"

module purge
module load gcc/14.2.0
module load python/3.13.1
if [ "$RUN_GPU_BUILD" = "1" ]; then
  if [ -z "$CUDA_MODULE" ]; then
    echo "RUN_GPU_BUILD=1 requires CUDA_MODULE, usually cuda/12.8." >&2
    exit 2
  fi
  module load "$CUDA_MODULE"
fi

source .venv_o2/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

manifest_csv="${sample_dir}/selected_base_points.csv"
if [ ! -f "$manifest_csv" ]; then
  echo "Missing selected manifest ${manifest_csv}; plan/select stage must finish first." >&2
  exit 2
fi

build_backend="batched"
require_backend=""
if [ "$RUN_GPU_BUILD" = "1" ]; then
  build_backend="cupy"
  require_backend="cupy"
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "RUN_GPU_BUILD=1 but CUDA_VISIBLE_DEVICES is unset. This job does not appear to have a GPU allocation." >&2
    exit 2
  fi
fi

{
  echo "Representative HDF5 build stage"
  echo "Run ID: ${RUN_ID}"
  echo "Job ID: ${SLURM_JOB_ID:-unknown}"
  echo "Host: $(hostname)"
  echo "Started: $(date)"
  echo "Repository: ${SLURM_SUBMIT_DIR:-$PWD}"
  echo "Submit git commit: ${SUBMIT_GIT_COMMIT:-unknown}"
  echo "Current git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Calibration config: ${CAL_CONFIG}"
  echo "Build stage: ${CAL_BUILD_STAGE}"
  echo "Manifest: ${manifest_csv}"
  echo "RUN_GPU_BUILD: ${RUN_GPU_BUILD}"
  echo "Requested build backend: ${build_backend}"
  echo "Required backend: ${require_backend:-none}"
  echo "CUDA module: ${CUDA_MODULE:-none}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "SLURM partition: ${SLURM_JOB_PARTITION:-unknown}"
  echo "SLURM CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
  echo "SLURM mem per node: ${SLURM_MEM_PER_NODE:-unknown}"
  echo "SLURM time limit: ${SLURM_TIMELIMIT:-unknown}"
  echo "Timeout seconds: ${CAL_BUILD_TIMEOUT_SECONDS}"
  echo "Max seconds per base point: ${CAL_MAX_SECONDS_PER_BASE_POINT}"
} | tee "$gpu_build_dir/gpu_build_metadata.txt"

module list 2>&1 | tee "$gpu_build_dir/module_list.txt"
which python | tee "$gpu_build_dir/python_path.txt"
python --version 2>&1 | tee "$gpu_build_dir/python_version.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$gpu_build_dir/nvidia_smi.txt"
else
  echo "nvidia-smi not found" | tee "$gpu_build_dir/nvidia_smi.txt"
fi

if [ "$RUN_GPU_BUILD" = "1" ]; then
  python - <<'PY' | tee "$gpu_build_dir/cupy_check.txt"
import cupy as cp
print("cupy_version=" + str(cp.__version__))
print("device_count=" + str(cp.cuda.runtime.getDeviceCount()))
dev = cp.cuda.Device()
print("device_id=" + str(dev.id))
props = cp.cuda.runtime.getDeviceProperties(dev.id)
name = props.get("name", b"unknown")
if isinstance(name, bytes):
    name = name.decode("utf-8", errors="replace")
print("device_name=" + str(name))
PY
fi

if [ -x /n/cluster/bin/job_gpu_monitor.sh ] && [ "$RUN_GPU_BUILD" = "1" ]; then
  pushd "$gpu_build_dir" >/dev/null
  /n/cluster/bin/job_gpu_monitor.sh &
  echo $! > gpu_monitor.pid
  popd >/dev/null
fi

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

build_args=(
  python -m tailbin_cache.cli build-hdf5
  --config "$CAL_CONFIG"
  --output "${build_dir}/representative_sample.h5"
  --base-point-manifest "$manifest_csv"
  --pgf-backend "$build_backend"
  --progress-path "${progress_dir}/build_representative_sample.progress.jsonl"
  --timeout-seconds "$CAL_BUILD_TIMEOUT_SECONDS"
  --max-seconds-per-base-point "$CAL_MAX_SECONDS_PER_BASE_POINT"
  --compression gzip
  --compression-opts 4
)
if [ -n "$require_backend" ]; then
  build_args+=(--require-pgf-backend "$require_backend")
fi
if [ "$CAL_WARN_ONLY_TIMEOUT" = "1" ]; then
  build_args+=(--warn-only-timeout)
fi

run_timed "build_representative_sample" "${json_dir}/build_representative_sample.json" "${build_args[@]}"

python scripts/o2/summarize_representative_calibration.py --run-id "$RUN_ID" \
  > "${log_dir}/summarize_build_stdout.txt" \
  2> "${log_dir}/summarize_build_stderr.txt" || true

echo "Finished: $(date)" | tee -a "$gpu_build_dir/gpu_build_metadata.txt"
echo "Progress: ${progress_dir}/build_representative_sample.progress.jsonl"
echo "Summary: ${run_dir}/summary.md"
