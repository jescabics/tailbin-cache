# O2 Smoke/Audit Workflow

These scripts submit a first-pass Tailbin O2 smoke/audit run. They are not a production cache build.

## Intended O2 Location

Use this repository path on O2 unless the project location changes:

```bash
/n/data1/hms/sysbio/hormoz/users/javi/tailbin-cache
```

## First-Time Sequence

From the repository root on O2:

```bash
cd /n/data1/hms/sysbio/hormoz/users/javi/tailbin-cache
git pull
```

Create the CPU-only project-local Python environment through SLURM:

```bash
bash scripts/o2/submit_setup_env.sh
```

For a GPU-capable environment, install CuPy in the setup job:

```bash
INSTALL_GPU_DEPS=1 bash scripts/o2/submit_setup_env.sh
```

Monitor setup:

```bash
squeue -u "$USER"
```

After setup completes, verify the environment from the login node:

```bash
bash scripts/o2/check_python_env.sh
```

Check GPU dependencies after a GPU-capable setup:

```bash
CHECK_GPU_DEPS=1 bash scripts/o2/check_python_env.sh
```

Then submit the CPU smoke job, GPU audit job, and dependent collector job:

```bash
bash scripts/o2/submit_smoke_audit.sh
```

Monitor:

```bash
squeue -u "$USER"
```

Find result bundles:

```bash
ls -lh results/*.tgz
```

Result bundles are intended to contain current-run logs only: CPU smoke logs, GPU audit logs, and collector logs matched by the submitted job IDs. Current-run GPU monitor logs are selected by `GPU_JOB_ID`, using `results/o2_gpu_audit/${GPU_JOB_ID}.gpulog` when present. For deeper debugging, include all historical `logs/` files by setting:

```bash
INCLUDE_ALL_LOGS=1 bash scripts/o2/submit_smoke_audit.sh
```

## Resource Calibration Sequence

Smoke/audit is a functional check: it proves the Python environment, CPU CLI, GPU backend, and SLURM dependency path work. Resource calibration is the next decision-making step before production.

From the repository root on O2:

```bash
git pull
bash scripts/o2/check_python_env.sh
bash scripts/o2/submit_resource_calibration.sh
squeue -u "$USER"
```

After the calibration and collector finish, inspect:

```bash
ls -lh results/tailbin_o2_resource_calibration_*.tgz
less results/o2_resource_calibration/<run_id>/summary.md
python -m json.tool results/o2_resource_calibration/<run_id>/summary.json | less
```

The calibration workflow runs small/moderate commands only:

* `tailbin_cache.cli --help`
* `estimate` on `examples/kmax2000_cpu_smoke.yaml`
* `plan --limit-bundles 1` on `examples/kmax2000_cpu_smoke.yaml`
* `estimate`, `plan`, and `plan-shards` on `examples/o2_resource_calibration.yaml`
* a tiny `build-hdf5 --limit-base-points 1` under `results/o2_resource_calibration/<run_id>/build/`
* a current GPU audit by default, unless `RUN_GPU_AUDIT=0` is set

Each important CPU command is wrapped in `/usr/bin/time -v` because `sacct MaxRSS` can be empty or unreliable for small jobs. Timing files are written under:

```bash
results/o2_resource_calibration/<run_id>/timing/
```

The summary helper writes:

```bash
results/o2_resource_calibration/<run_id>/summary.md
results/o2_resource_calibration/<run_id>/summary.json
```

Resource calibration overrides:

```bash
RUN_LABEL=first_calibration \
CAL_PARTITION=short CAL_TIME=2:00:00 CAL_MEM=4G CAL_CPUS=1 \
RUN_GPU_AUDIT=1 GPU_PARTITION=gpu_quad GPU_TIME=1:00:00 GPU_MEM=16G GPU_CPUS=4 \
COLLECT_PARTITION=short COLLECT_TIME=0:30:00 COLLECT_MEM=1G COLLECT_CPUS=1 \
bash scripts/o2/submit_resource_calibration.sh
```

Use `RUN_GPU_AUDIT=0` for CPU-only calibration. Do not start full production until `summary.md`, `summary.json`, accounting, and any GPU audit output have been reviewed.

`scripts/o2/submit_production_pilot.sh` is currently a documented placeholder. It intentionally exits without submitting jobs until calibration determines safe pilot parameters.

## Representative Grid B Calibration

For the named `local34_diag_v1_k10000_1k` target, do not use a two-base-point build as serious calibration, and do not jump straight to a 40-point hard sample. Start with a staged GPU easy smoke:

```bash
RUN_LABEL=local34_diag_v1_k10000_1k_gpu_easy_smoke \
CAL_CONFIG=examples/local34_diag_v1_k10000_1k.yaml \
CAL_FULL_PLAN=1 \
CAL_RUN_SHARD_PLAN=0 \
CAL_SHARDS=8 \
CAL_BUILD_STAGE=easy_smoke \
CAL_BUILD_SAMPLE_BASE_POINTS=1 \
CAL_BUILD_SAMPLE_STRATEGY=easy_first \
RUN_GPU_BUILD=1 \
RUN_GPU_AUDIT=1 \
CUDA_MODULE=cuda/12.8 \
bash scripts/o2/submit_representative_calibration.sh
```

This submits a CPU plan/select job, a dependent representative build job, an optional GPU audit job, and a dependent collector. The CPU plan/select job:

* plans all Grid B base points and alpha tables;
* writes the full adaptive plan;
* skips shard planning by default (`CAL_RUN_SHARD_PLAN=0`);
* writes ordered manifests `selected_easy_first.csv`, `selected_stratified_8.csv`, and `selected_representative_40.csv`;
* copies the stage-selected manifest to `sample/selected_base_points.csv`.

The representative build job:

* runs after plan/select succeeds;
* runs on a GPU partition by default when `RUN_GPU_BUILD=1`;
* passes `build-hdf5 --base-point-manifest ... --pgf-backend cupy --require-pgf-backend cupy`;
* writes live JSONL progress to `progress/build_representative_sample.progress.jsonl`;
* prints flushed progress lines so `tail -f logs/*<GPU_BUILD_JOBID>*` shows movement;
* writes easy/moderate/hard/problematic classification in the summary.

Generated representative calibration outputs are under:

```bash
results/o2_representative_calibration/<run_id>/
results/tailbin_o2_representative_calibration_<run_id>.tgz
```

`RUN_GPU_AUDIT=1` is only a separate GPU health/correctness audit. `RUN_GPU_BUILD=1` is what makes the representative HDF5 build request a GPU node and use the CuPy backend. The checked-in scientific config remains `pgf_backend: batched`; the O2 workflow applies the CuPy backend as a run-time override and fails before expensive work if the effective backend is not CuPy.

After the 1-point `easy_smoke` succeeds and the summary shows real GPU progress, run `stratified_probe` with 8 points. Do not run the 40-point `representative_sample` until the 1-point GPU easy smoke is reviewed.

## Python Environment

The O2 environment is repo-local at `.venv_o2/` and ignored by Git.

O2 scripts intentionally load this module stack before creating or activating `.venv_o2`:

```bash
module purge
module load gcc/14.2.0
module load python/3.13.1
```

O2 currently exposes `cuda/12.8` after `gcc/14.2.0`; `cuda/11.7` should not be hard-coded. GPU scripts default to:

```bash
CUDA_MODULE=cuda/12.8
GPU_PIP_PACKAGE=cupy-cuda12x
```

The old `/home/jae37/pythonEnv_3.10.11` environment should not be used for this project.

If CuPy does not install under Python 3.13, capture `logs/` and `results/o2_python_env_check.txt` and do not run GPU audit until the environment strategy is revised.

To recreate the environment:

```bash
RESET_O2_VENV=1 bash scripts/o2/submit_setup_env.sh
```

To recreate it and install GPU dependencies:

```bash
RESET_O2_VENV=1 INSTALL_GPU_DEPS=1 bash scripts/o2/submit_setup_env.sh
```

To rediscover available CUDA modules from the login node:

```bash
bash scripts/o2/discover_cuda.sh
```

## Submitter Resource Overrides

Environment setup defaults are modest:

```bash
SETUP_PARTITION=short SETUP_TIME=1:00:00 SETUP_MEM=4G SETUP_CPUS=1 \
bash scripts/o2/submit_setup_env.sh
```

Optional setup variables:

* `INSTALL_GPU_DEPS=1` installs the GPU package into `.venv_o2`.
* `CUDA_MODULE` defaults to `cuda/12.8`.
* `GPU_PIP_PACKAGE` defaults to `cupy-cuda12x`.

Smoke/audit resources can also be overridden:

```bash
CPU_PARTITION=short CPU_TIME=1:00:00 CPU_MEM=2G CPU_CPUS=1 \
GPU_PARTITION=gpu_quad GPU_TIME=2:00:00 GPU_MEM=16G GPU_CPUS=4 GPU_GRES=gpu:1 GPU_CONSTRAINT=gpu_doublep CUDA_MODULE=cuda/12.8 \
COLLECT_PARTITION=short COLLECT_TIME=0:30:00 COLLECT_MEM=1G COLLECT_CPUS=1 \
bash scripts/o2/submit_smoke_audit.sh
```

You can omit `GPU_CONSTRAINT` by setting it to an empty string:

```bash
GPU_CONSTRAINT="" bash scripts/o2/submit_smoke_audit.sh
```

## Jobs

`submit_smoke_audit.sh` submits:

* `cpu_smoke.sbatch`: runs `python -m tailbin_cache.cli estimate` and a tiny `plan --limit-bundles 1` with `examples/kmax2000_cpu_smoke.yaml`.
* `gpu_audit.sbatch`: loads `CUDA_MODULE` when set, starts the O2 GPU monitor if available, and runs `python -u examples/gpu_backend_audit.py`.
* `collect_results.sbatch`: runs after both jobs with `afterany:<cpu_jobid>:<gpu_jobid>` and creates `results/tailbin_o2_smoke_audit_<timestamp>.tgz`.
* `submit_resource_calibration.sh`: submits the CPU resource calibration job, optionally submits a current GPU audit job, and submits a dependent resource-calibration collector.
* `resource_calibration.sbatch`: runs bounded estimate/plan/shard-plan/tiny-build commands with `/usr/bin/time -v`.
* `collect_resource_calibration.sbatch`: collects current calibration logs, accounting, GPU audit artifacts when present, and writes `results/tailbin_o2_resource_calibration_<run_id>.tgz`.
* `submit_representative_calibration.sh`: submits staged Grid B plan/select, representative build, optional GPU audit, and collector jobs.
* `representative_calibration.sbatch`: runs CPU-only full Grid B plan and ordered sample selection; shard planning is optional with `CAL_RUN_SHARD_PLAN=1`.
* `representative_build.sbatch`: runs the selected representative HDF5 build, using CuPy/GPU when `RUN_GPU_BUILD=1`, with live progress logging.
* `collect_representative_calibration.sbatch`: collects current representative calibration logs/artifacts, accounting, optional current GPU audit output, and writes `results/tailbin_o2_representative_calibration_<run_id>.tgz`.
* `submit_production_pilot.sh`: placeholder only; it does not submit production until calibration results are reviewed.

## Outputs

Generated files stay in ignored project directories:

* `logs/`
* `outputs/o2_smoke/`
* `outputs/o2_gpu_audit/`
* `results/o2_smoke/`
* `results/o2_gpu_audit/`
* `results/o2_resource_calibration/`
* `results/o2_representative_calibration/`
* `results/tailbin_o2_smoke_audit_<timestamp>.tgz`
* `results/tailbin_o2_resource_calibration_<run_id>.tgz`
* `results/tailbin_o2_representative_calibration_<run_id>.tgz`

The collector bundles current-run logs, selected outputs/results, O2 docs, example configs, git metadata, accounting output, and the GPU monitor log matching `GPU_JOB_ID` when present.

By default, the collector only includes logs matching the current CPU, GPU, and collector job IDs. Set `INCLUDE_ALL_LOGS=1` to add the full `logs/` directory under `logs/all_logs/` inside the bundle. This does not sweep historical GPU monitor logs from `results/o2_gpu_audit/`; GPU monitor logs remain selected by `GPU_JOB_ID`.

## Retrieve The Bundle

After the collector finishes, send back or download the newest bundle from:

```bash
results/tailbin_o2_smoke_audit_<timestamp>.tgz
```

For example, from your laptop you can use `scp` with your normal O2 access method, or attach/share the bundle through the agreed result-review workflow.

## Login Node Boundary

Do not run heavy Tailbin planners, HDF5 builds, GPU audits, resource calibration, or production jobs directly on login nodes. Use `bash scripts/o2/submit_smoke_audit.sh` or `bash scripts/o2/submit_resource_calibration.sh` to submit work through SLURM.
