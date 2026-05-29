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

## Outputs

Generated files stay in ignored project directories:

* `logs/`
* `outputs/o2_smoke/`
* `outputs/o2_gpu_audit/`
* `results/o2_smoke/`
* `results/o2_gpu_audit/`
* `results/tailbin_o2_smoke_audit_<timestamp>.tgz`

The collector bundles logs, selected outputs/results, O2 docs, example configs, git metadata, accounting output, and GPU monitor logs when present.

## Retrieve The Bundle

After the collector finishes, send back or download the newest bundle from:

```bash
results/tailbin_o2_smoke_audit_<timestamp>.tgz
```

For example, from your laptop you can use `scp` with your normal O2 access method, or attach/share the bundle through the agreed result-review workflow.

## Login Node Boundary

Do not run heavy Tailbin planners, HDF5 builds, GPU audits, or production jobs directly on login nodes. Use `bash scripts/o2/submit_smoke_audit.sh` to submit work through SLURM.
