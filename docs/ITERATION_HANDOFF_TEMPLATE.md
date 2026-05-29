# Tailbin Iteration Handoff Template

This file is the standard template for closing the feedback loop between O2 runs, ChatGPT analysis, and Codex implementation. Use it after each O2 iteration so results, resource data, and next implementation requests stay reproducible and specific.

## What to Send Back From O2

Paste or bundle the following after each run:

1. Git commit hash:

```bash
git rev-parse HEAD
```

2. Job accounting:

```bash
sacct -j <jobids> --format=JobId,JobName,Partition,NCPUS,State,ReqMem,MaxRSS,Elapsed,TimeLimit,ExitCode
```

3. O2 job report if available:

```bash
O2_jobs_report -j <jobid>
```

4. Queue/priority snapshot when relevant:

```bash
squeue -u "$USER"
sprio -l -u "$USER"
sshare -u "$USER" -U
```

5. Logs:

```bash
tail -n 160 logs/*<jobid>* 2>/dev/null
```

6. Result files:

```bash
find logs outputs results -maxdepth 4 -type f | sort
```

7. Latest result bundle:

```bash
latest_bundle="$(ls -t results/tailbin_o2_*.tgz 2>/dev/null | head -n 1)"
echo "$latest_bundle"
tar -tzf "$latest_bundle" | head -n 80
```

8. Any failure messages from stdout/stderr.

## What ChatGPT Should Analyze

ChatGPT should classify each iteration into:

* environment/setup failure;
* SLURM/resource-request failure;
* numerical/certification failure;
* runtime/performance bottleneck;
* GPU/CUDA/CuPy backend failure;
* packaging/reproducibility failure;
* success ready for next benchmark tier.

For each job, ChatGPT should extract:

* job ID;
* job state and exit code;
* partition;
* requested memory;
* MaxRSS;
* requested CPUs;
* elapsed time;
* time limit;
* whether memory request was too high, too low, or reasonable;
* whether wall-time was too high, too low, or reasonable;
* whether CPU/GPU allocation matched actual work;
* whether outputs and result bundles were produced.

## Resource Calibration Rules

* Do not increase memory just because a job failed; first check whether it failed from OUT_OF_MEMORY.
* If MaxRSS is far below requested memory, reduce future memory requests.
* If jobs finish in seconds or minutes, reduce wall-time unless this is only a tiny smoke test.
* Keep easy and hard shards separate.
* Prefer arrays with concurrency limits over many manual one-off submissions.
* Use dependencies for multi-stage workflows so the user does not need to manually wait and submit the next job.
* Keep generated outputs, logs, HDF5 caches, and bundles out of Git.

## What Codex Needs In Each Prompt

Codex prompts should include:

* current git commit hash;
* exact files it may edit;
* exact files it must not edit;
* summary of O2 results;
* job IDs and accounting summary;
* relevant error messages;
* resource conclusions;
* intended next benchmark tier;
* whether the change is docs-only, script-only, or source-code;
* whether tests may be run locally;
* explicit instruction not to commit unless requested.

## Standard Codex Prompt Skeleton

```text
You are working in my local Tailbin repository.

Goal: <one sentence goal>

O2 evidence:
<job ids, states, logs, accounting, result summary>

Resource conclusions:
<memory/cpu/wall-time/GPU conclusions>

Allowed edits: <files or directories>

Forbidden edits: <files or directories>

Rules:

* Do not install packages unless explicitly asked.
* Do not run O2/SLURM/SSH commands locally.
* Do not run long tests.
* Do not use git add, git commit, or git push.
* Keep generated outputs ignored.
* Preserve O2 resource-efficiency rules.

Requested implementation: <numbered tasks>

After finishing, report:

1. Files changed.
2. Summary of changes.
3. Git status.
4. Diff summary.
5. Any assumptions or follow-up commands needed.
```

## Current Known Tailbin O2 State

* Repo O2 location: `/n/data1/hms/sysbio/hormoz/users/javi/tailbin-cache`
* Correct O2 Python stack: `module load gcc/14.2.0`, `module load python/3.13.1`
* Project-local environment: `.venv_o2`
* Old environment `/home/jae37/pythonEnv_3.10.11` is broken and should not be used.
* O2 setup job `41873491` completed successfully.
* CPU smoke job `41873738` completed successfully.
* GPU audit job `41873739` failed because `cuda/11.7` was not an available module.
* Collector job `41873740` completed and produced a result bundle.
* Next likely task is CUDA module discovery and making the GPU audit script configurable instead of hard-coding `cuda/11.7`.
