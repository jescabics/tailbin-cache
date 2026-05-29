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

## Current Tailbin Calibration Command

After smoke/audit passes, run the resource calibration workflow from the O2 repository root:

```bash
git pull
bash scripts/o2/check_python_env.sh
bash scripts/o2/submit_resource_calibration.sh
squeue -u "$USER"
```

After completion, inspect:

```bash
results/o2_resource_calibration/<run_id>/summary.md
results/o2_resource_calibration/<run_id>/summary.json
results/tailbin_o2_resource_calibration_<run_id>.tgz
```

Smoke/audit is a functional check. Resource calibration is the next decision-making step: it measures small estimate/plan/shard-plan/tiny-build commands and, by default, submits a current GPU audit so GPU behavior can be reviewed beside CPU timing data.

Do not launch full production until the calibration summary, accounting, logs, and GPU audit output have been reviewed.

The next target-grid calibration order is:

1. `local34_diag_v1_k10000_1k`
2. `full100k_v1_k50000` preflight

Do not use a two-point build as the serious Grid B calibration. A two-point build is only a smoke test; it cannot reveal which age-34 diagonal regimes are easy, hard, slow, memory-heavy, or certification-problematic.

Run the local age-34 diagonal representative calibration first:

```bash
RUN_LABEL=local34_diag_v1_k10000_1k_representative \
CAL_CONFIG=examples/local34_diag_v1_k10000_1k.yaml \
CAL_FULL_PLAN=1 \
CAL_SHARDS=8 \
CAL_BUILD_SAMPLE_BASE_POINTS=40 \
CAL_BUILD_SAMPLE_STRATEGY=representative_hard \
RUN_GPU_AUDIT=1 \
bash scripts/o2/submit_representative_calibration.sh
```

This workflow plans all 1,000 Grid B base points and all 20,000 alpha tables, selects a deterministic representative 40-base-point sample, builds only that selected sample, classifies selected points as easy/moderate/hard/problematic, and bundles the current-run logs and artifacts.

Then run the full target preflight, still as calibration only:

```bash
RUN_LABEL=full100k_v1_k50000_preflight \
CAL_CONFIG=examples/full100k_v1_k50000.yaml \
CAL_PLAN_LIMIT_BUNDLES=20 \
CAL_SHARDS=16 \
TINY_BUILD_LIMIT_BASE_POINTS=1 \
RUN_GPU_AUDIT=1 \
bash scripts/o2/submit_resource_calibration.sh
```

These commands estimate runtime, memory, shard balance, GPU behavior, and adaptive storage shape. They do not enable production, and full Grid A production remains disabled until the Grid B representative calibration summary has been reviewed.

Representative selection includes low/median/high planner work proxies, low/high `R`, low/high `N`, `T_b = 0`, `T_b = 20`, intermediate `T_b` values, and planner-predicted full or large-prefix points when available. Selection is deterministic and writes both CSV and JSON manifests.

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

Because `sacct MaxRSS` can be empty or unreliable for small jobs, Tailbin resource calibration also wraps important commands with:

```bash
/usr/bin/time -v
```

The timing output records elapsed wall time, maximum resident set size, exit status, and command metadata under:

```bash
results/o2_resource_calibration/<run_id>/timing/
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

For `local34_diag_v1_k10000_1k`, the representative HDF5 build uses the config value `pgf_backend: batched`, so it is CPU-based. `RUN_GPU_AUDIT=1` is a separate health/correctness check. The HDF5 builder can use the CuPy path only when the build config explicitly sets `pgf_backend: cupy`; do not assume production HDF5 builds are GPU-accelerated from the audit alone.

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

Do not launch a full target build until the smaller calibration and production samples show acceptable certification and resource efficiency.

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
