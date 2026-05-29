# O2 Runbook

This document records O2-specific execution rules for the Tailbin project. It guides future SLURM scripts, resource requests, monitoring, GPU usage, result collection, and cluster-safe execution.

## Project Location on O2

Preferred O2 repository/work directory:

```bash
/n/data1/hms/sysbio/hormoz/users/javi/tailbin-cache
```

This path should be treated as the main O2 working copy unless explicitly changed.

Large generated outputs, HDF5 caches, result bundles, benchmark logs, and temporary scratch artifacts should not be committed to Git. They should be written under project output directories, scratch, or another agreed O2 storage location.

## Core O2 Rule

Do not run computationally intensive Tailbin work directly on O2 login nodes.

Login nodes are only for lightweight commands such as:

```bash
git pull
git status
ls
cat
head
tail
less
mkdir
sbatch
squeue
O2_jobs_report
O2sacct
```

Do not run planner sweeps, HDF5 builds, GPU audits, large Python jobs, or multi-core Python processes directly on login nodes. Use `sbatch` or an interactive `srun` allocation instead.

If a command may take more than a few minutes or use significant CPU, RAM, or GPU resources, submit it through SLURM.

## Development and Execution Workflow

Normal workflow:

1. Edit code locally using Codex.
2. Review the local diff.
3. Commit and push reviewed changes to GitHub.
4. Pull the repository on O2.
5. Submit O2 jobs with SLURM.
6. Collect logs, summaries, resource reports, and result bundles.
7. Bring result bundles back to ChatGPT for analysis.

Codex edits the local repository. O2 runs computational jobs.

## Resource Efficiency Policy

O2 monitors memory, CPU, GPU, and wall-time efficiency. Over-requesting resources can reduce job priority.

The Tailbin project must avoid defaulting to excessive memory, CPU, GPU, or wall-time requests.

Important observed issue:

* Prior jobs requested much more memory than they used.
* Example report: average requested memory was approximately 84 GB, average used memory was approximately 1 GB, and maximum observed memory was approximately 4.5 GB.
* Future scripts should avoid large memory defaults unless measurements justify them.

General policy:

* Start with modest resource requests.
* Increase memory only if jobs fail with OUT_OF_MEMORY or measured MaxRSS approaches the requested limit.
* Prefer separate job tiers over one huge request for every job.
* Do not request many CPU cores unless the code actually uses them.
* Do not request maximum wall-time by default.
* Short jobs should be grouped into batches that run at least roughly 15 minutes when practical.
* Long or hard jobs should be submitted separately with larger requests, not mixed into the same array as easy jobs.
* Resource requests should be revised after every calibration or production batch using measured accounting data.

## Recommended Initial Resource Tiers

These are starting points, not permanent truths. Adjust after checking O2 resource reports.

### Lightweight setup or metadata jobs

Use for task generation, small summaries, config checks, and collection scripts.

```bash
-p short
-t 0:30:00
-c 1
--mem=1G
```

### CPU smoke tests

Use for tiny planner/build probes.

```bash
-p short
-t 1:00:00
-c 1
--mem=2G
```

### CPU planner census

Use for planner/preflight-only jobs that may take longer but should not require large memory.

```bash
-p short
-t 4:00:00
-c 1
--mem=4G
```

### CPU build shards

Use for production cache-building shards when running CPU-only.

```bash
-p short or medium
-t 4:00:00 to 12:00:00
-c 1
--mem=4G to 8G
```

Use higher memory only after measured MaxRSS requires it.

### Hard CPU shards

Use for known hard rows or long build shards.

```bash
-p medium
-t 12:00:00 to 24:00:00
-c 1
--mem=8G to 16G
```

Do not apply this tier to all jobs unless measured usage justifies it.

### GPU audit jobs

Use for CPU-vs-GPU correctness and speed tests.

```bash
-p gpu or gpu_quad
--gres=gpu:1
-c 2 to 4
--mem=8G to 16G
-t 1:00:00 to 4:00:00
```

For early correctness testing, prefer double-precision-capable GPUs if needed:

```bash
--constraint=gpu_doublep
```

### GPU production shards

Use only after GPU audit confirms real speedup and numerical agreement.

```bash
-p gpu_quad or gpu_requeue
--gres=gpu:1
-c 2 to 4
--mem=16G to 32G
-t based on measured shard runtime
```

Avoid requesting multiple GPUs per task unless the code explicitly uses multiple GPUs.

## CPU Core Policy

Request `-c 1` by default unless the job is known to use multiple CPU cores.

If requesting multiple cores:

* The program must actually use multiple threads or processes.
* Set thread environment variables to match SLURM allocation when relevant:

```bash
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
```

Do not request extra cores just because a job is slow. First determine whether the code can use them.

## Wall-Time Policy

Wall-time does not directly count like memory, but excessive wall-time can reduce backfill opportunities and increase pending time.

Use measured runtimes to set time limits.

Bad pattern:

```bash
# requesting 12 hours for jobs that normally run under 5 minutes
-t 12:00:00
```

Better pattern:

* Use short wall-time for easy shards.
* Use separate hard-row queues for the small number of slow shards.
* Resubmit failed or timed-out shards with larger limits.
* Keep job arrays stratified by expected hardness.

## Job Arrays and Sharding

Prefer SLURM arrays for many similar shards.

Use array concurrency limits to avoid overwhelming fairshare or active resource caps:

```bash
#SBATCH --array=0-99%10
```

Each array task should write an independent output file. This makes jobs restartable and safe for `gpu_requeue`.

Good output naming pattern:

```bash
outputs/shard_${SLURM_ARRAY_TASK_ID}.h5
logs/%x_%A_%a.out
logs/%x_%A_%a.err
```

Do not make many array tasks write to the same HDF5 file concurrently.

## GPU Rules

Do not manually set or overwrite `CUDA_VISIBLE_DEVICES`.

SLURM sets `CUDA_VISIBLE_DEVICES` correctly. Changing it may cause the job to run without the allocated GPU.

Load CUDA modules inside GPU jobs. Example:

```bash
module load gcc/9.2.0
module load cuda/11.7
```

If using CuPy, the installed CuPy package must match the CUDA runtime. For CUDA 11.x, this is commonly:

```bash
cupy-cuda11x
```

GPU partitions:

* `gpu` is broadly available but has active limits.
* `gpu_quad` is preferred for larger production GPU runs if access is available.
* `gpu_requeue` can be useful for opportunistic throughput, but jobs may be killed and requeued.

The open `gpu` partition has active per-user limits, including GPU-hour, CPU-core, and memory limits. Scripts should avoid excessive concurrent GPU jobs on this partition.

Use GPU utilization monitoring in GPU jobs when appropriate:

```bash
/n/cluster/bin/job_gpu_monitor.sh &
```

This creates a GPU utilization log that helps determine whether the job is actually using the GPU.

## Double Precision GPU Testing

The Tailbin PGF/CDF code may require double precision or careful mixed-precision validation.

For early correctness testing, prefer A100, V100, or V100s where possible.

If double precision is required, include:

```bash
--constraint=gpu_doublep
```

Only use L40S, RTX8000, A40, or similar single-precision-oriented cards for production after CPU-vs-GPU audits show acceptable numerical agreement and certification behavior.

## Monitoring Commands

Useful commands:

```bash
squeue -u "$USER"
squeue -t RUNNING -u "$USER"
squeue -t PENDING -u "$USER"
O2_jobs_report
O2sacct --help
sacct -j <jobid> --format=JobId,NNodes,Partition,NCPUS,State,ReqMem,MaxRSS,Elapsed,CPUTime,TimeLimit,ExitCode,Start,End
sstat -j <jobid> --format JobID,MaxRSS,MaxVMSize,NTasks
sprio -l -u "$USER"
sshare -u "$USER" -U
```

For completed jobs, collect at least:

* job ID
* partition
* requested memory
* MaxRSS
* elapsed time
* time limit
* requested CPUs
* CPU efficiency if available
* GPU utilization log if GPU job
* job state and exit code

## Efficiency Acceptance

After any benchmark or production batch, check resource usage.

A good O2 result bundle should include:

* SLURM stdout/stderr logs
* config files used
* planner/build summary JSON
* certification summary
* O2_jobs_report or O2sacct output
* GPU utilization log for GPU runs
* exact git commit hash
* package version or branch name

Resource requests should be adjusted based on measured MaxRSS, elapsed time, and GPU utilization.

## Tailbin Numerical Acceptance

For production inference, speed is not enough. Required rows must be present and certified.

A production cache is acceptable only if:

* every required row exists;
* every required row has `certified=true`;
* no missing, deferred, or uncertified rows are silently accepted;
* numerical error indicators are within configured tolerance;
* all configs and logs are saved with the result bundle.

## Guidance for Future Codex-Generated Scripts

When Codex creates O2 scripts for this project, scripts should:

* use `sbatch` scripts rather than heavy login-node commands;
* create `logs/`, `outputs/`, `results/`, and `tasks/` directories as needed;
* use conservative memory defaults;
* expose resource requests through environment variables where reasonable;
* avoid hard-coding excessive memory or wall-time;
* use job arrays for shardable workloads;
* include array concurrency limits;
* write one output per shard;
* collect O2 accounting information after jobs finish;
* include the git commit hash in result metadata;
* avoid committing generated outputs;
* include GPU monitoring for GPU jobs;
* never overwrite `CUDA_VISIBLE_DEVICES`.

## Example CPU sbatch Skeleton

```bash
#!/bin/bash
#SBATCH -c 1
#SBATCH -t 4:00:00
#SBATCH -p short
#SBATCH --mem=4G
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs outputs results tasks

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"

# Run project command here.
# Use one output file per shard.

echo "Finished: $(date)"
```

## Example GPU sbatch Skeleton

```bash
#!/bin/bash
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

mkdir -p logs outputs results

module load gcc/9.2.0
module load cuda/11.7

/n/cluster/bin/job_gpu_monitor.sh &

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started: $(date)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"

# Run GPU audit or GPU shard command here.

echo "Finished: $(date)"
```
