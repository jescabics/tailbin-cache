# O2 Resource Calibration

This document describes how Tailbin jobs should choose O2 memory, CPU, GPU, and wall-time requests without over-requesting resources.

The guiding rule is:

> Start modestly, measure actual usage, adjust by shard class, and resubmit only the jobs that need larger resources.

Scripts should not assume one large resource request for all jobs.

## Why This Exists

O2 may reduce job priority for users who consistently request substantially more memory than their jobs use.

The Tailbin project has already seen reports where jobs requested much more memory and wall-time than they used. Future O2 scripts must therefore be measurement-driven.

## Required Resource Loop

Every substantial O2 workflow should follow this loop:

1. Run a small smoke job.
2. Run a calibration batch across representative shard classes.
3. Collect accounting data.
4. Compute recommended memory and wall-time by class.
5. Submit production arrays using class-specific resource tiers.
6. Collect accounting again.
7. Resubmit only failed, timed-out, or OUT_OF_MEMORY shards with a larger tier.
8. Update the recommended tiers based on measured data.

## Shard Classes

Tailbin jobs should be stratified whenever possible.

Useful classes include:

* `metadata`
* `planner`
* `compact_prefix`
* `short_prefix`
* `medium_prefix`
* `near_full_prefix`
* `full_or_kmax`
* `gpu_audit`
* `gpu_production`

Do not mix easy and hard shards into one array with one oversized memory and wall-time request.

## Metrics to Collect

For every completed, failed, timed-out, or OOM job, collect:

* job ID
* array task ID if applicable
* git commit hash
* config path
* shard class
* partition
* requested CPUs
* requested memory
* requested wall-time
* MaxRSS
* elapsed time
* CPU efficiency if available
* GPU utilization if applicable
* GPU VRAM usage if applicable
* job state
* exit code
* output file path
* summary JSON path if available

Useful commands include:

```bash
O2_jobs_report
O2sacct --help
sacct -j <jobid> --format=JobId,NNodes,Partition,NCPUS,State,ReqMem,MaxRSS,Elapsed,CPUTime,TimeLimit,ExitCode,Start,End
```

## Resource Recommendation Rule

After a calibration batch, choose resource requests by class.

Recommended memory:

```text
recommended_memory = max(minimum_tier_memory, ceil_to_allowed_unit(1.5 * p95_MaxRSS))
```

Recommended wall-time:

```text
recommended_walltime = max(minimum_tier_time, round_up(2.0 * p95_elapsed))
```

Then manually review the recommendation before launching large production batches.

Do not increase all jobs because a few hard jobs failed. Split hard jobs into a separate class and resource tier.

## Memory Guidance

Use measured MaxRSS.

Starting points:

* metadata: 1G
* smoke: 2G
* planner: 4G
* normal CPU build: 4G to 8G
* hard CPU build: 8G to 16G
* GPU audit: 8G to 16G
* GPU production: 16G to 32G only after measurement

If a class uses less than half the requested memory in nearly all jobs, lower the request unless the request is already very small.

If a job fails with OUT_OF_MEMORY, resubmit that shard or class with a larger memory tier.

## CPU Guidance

Use `-c 1` unless the job is known to use multiple cores.

If using multiple cores, set thread variables:

```bash
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
```

Average CPU efficiency should be reviewed after batches. If CPU efficiency is poor, reduce requested cores or fix the code so it uses the requested cores.

## Wall-Time Guidance

Avoid requesting maximum wall-time for all jobs.

Good policy:

* easy shards: short wall-time
* hard shards: separate longer wall-time
* failed timeout shards: resubmit with larger wall-time
* production batches: use measured p95 elapsed by class

Short jobs should be grouped when practical so each array task runs at least roughly 15 minutes.

## GPU Guidance

GPU resources should be requested only when the code actually uses the GPU backend.

GPU audit jobs must report:

* CPU baseline time
* GPU time
* speedup
* CPU-vs-GPU numerical agreement
* certification status
* GPU utilization
* GPU VRAM usage

If GPU utilization is low, do not scale to many GPUs until the backend is fixed.

Use one GPU per shard unless the code explicitly supports multiple GPUs.

Do not overwrite `CUDA_VISIBLE_DEVICES`.

## Production Submission Policy

Production submissions should be staged:

1. Smoke.
2. Calibration.
3. Small production array.
4. Full production array.
5. Failed-shard resubmission.

Each stage should produce a result bundle.

Do not launch a full 10,000 or 100,000 point build until the smaller calibration and production samples show acceptable certification and resource efficiency.

## Result Bundle Requirements

Every calibration or production run should bundle:

* logs/
* configs/
* tasks/
* summaries/
* O2 accounting reports
* GPU monitor logs if applicable
* git commit hash
* package version or branch name
* failed shard list
* timed-out shard list
* OOM shard list
* certification summary

The bundle should be sent back to ChatGPT for analysis.

## Failure Handling

If a shard succeeds:

* keep its output;
* include it in the final merge if certified.

If a shard times out:

* classify it as a hard shard;
* resubmit only that shard or class with a larger wall-time tier;
* do not increase wall-time for all easy shards.

If a shard fails with OUT_OF_MEMORY:

* resubmit only that shard or class with a larger memory tier;
* do not increase memory for all easy shards.

If a shard is uncertified:

* do not silently accept it;
* inspect numerical diagnostics;
* either refine settings or mark it as requiring method development.

If a GPU shard has low GPU utilization:

* do not scale GPU production;
* benchmark and fix the GPU backend first.

## Codex Script Requirements

Future Codex-generated O2 scripts should implement or support:

* `smoke` mode;
* `calibration` mode;
* class-stratified task generation;
* SLURM array submission with concurrency limits;
* one output per shard;
* resource requests controlled by environment variables;
* result collection;
* accounting collection;
* failed-shard list creation;
* timed-out-shard list creation;
* OOM-shard list creation;
* GPU utilization log collection for GPU jobs;
* no generated outputs committed to Git.

Scripts should make resource iteration easy. They should not pretend to know perfect resource requests before measurement.

## Acceptance Standard

A resource setup is acceptable when:

* jobs complete and certify;
* most jobs are not over-requesting memory by large factors;
* wall-time requests are close enough to measured runtimes to allow backfill;
* CPU core requests match actual CPU usage;
* GPU jobs show meaningful GPU utilization;
* failed or hard shards are isolated into appropriate tiers;
* resource accounting is included in every result bundle.
