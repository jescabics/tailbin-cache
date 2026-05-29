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

9. For resource calibration runs:

```bash
latest_cal="$(ls -td results/o2_resource_calibration/* 2>/dev/null | head -n 1)"
echo "$latest_cal"
cat "$latest_cal/summary.md"
python -m json.tool "$latest_cal/summary.json" | head -n 160
find "$latest_cal/timing" -type f | sort
```

10. For representative Grid B calibration runs:

```bash
latest_rep="$(ls -td results/o2_representative_calibration/* 2>/dev/null | head -n 1)"
echo "$latest_rep"
cat "$latest_rep/summary.md"
python -m json.tool "$latest_rep/summary.json" | head -n 200
head -n 40 "$latest_rep/sample/selected_base_points.csv"
head -n 40 "$latest_rep/classification/selected_base_point_classification.csv"
tail -n 40 "$latest_rep/progress/build_representative_sample.progress.jsonl"
```

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

* target grid name;
* `Kmax`;
* number of base parameter points;
* number of alpha tables;
* whether the age constraint is `<=` or exact equality;
* whether `T` / `T_b` are integer-grid, linearly sampled, or paired diagonal values;
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
* for resource calibration, `/usr/bin/time -v` elapsed time and maximum resident set size for each timed command.
* for representative calibration, whether `RUN_GPU_BUILD=1` was used, whether the actual build backend was `cupy`, and how many progress events/base points completed.

## Resource Calibration Rules

* Do not increase memory just because a job failed; first check whether it failed from OUT_OF_MEMORY.
* If MaxRSS is far below requested memory, reduce future memory requests.
* If jobs finish in seconds or minutes, reduce wall-time unless this is only a tiny smoke test.
* Keep easy and hard shards separate.
* Prefer arrays with concurrency limits over many manual one-off submissions.
* Use dependencies for multi-stage workflows so the user does not need to manually wait and submit the next job.
* Keep generated outputs, logs, HDF5 caches, and bundles out of Git.
* Treat smoke/audit as a functional check and resource calibration as the resource-decision step before production pilot.
* Prefer `/usr/bin/time -v` timing files when `sacct MaxRSS` is missing or unreliable.
* Do not treat a two-point build as serious calibration. Representative calibration should plan the full target grid, select deterministic samples across easy/median/hard/boundary regimes, and build only the selected stage.
* `RUN_GPU_AUDIT=1` only audits GPU health/correctness. `RUN_GPU_BUILD=1` is required for the representative HDF5 build itself to use `pgf_backend: cupy`.
* Run representative calibration in stages: `easy_smoke`, then `stratified_probe`, then larger `representative_sample` only after summaries show real GPU progress.
* Shard planning should remain disabled by default for representative calibration unless shard balance is the specific question.

## What Codex Needs In Each Prompt

Codex prompts should include:

* current git commit hash;
* target grid name;
* `Kmax`;
* number of base parameter points;
* number of alpha tables;
* age constraint mode, such as `T + T_b <= max_age` or `T + T_b = age_exact`;
* whether `T` / `T_b` are integer-grid, linearly sampled, or paired diagonal values;
* representative sample size and selection strategy when applicable;
* representative build stage, such as `easy_smoke`, `stratified_probe`, or `representative_sample`;
* whether `RUN_GPU_BUILD` and `RUN_GPU_AUDIT` were enabled;
* exact files it may edit;
* exact files it must not edit;
* summary of O2 results;
* job IDs and accounting summary;
* relevant error messages;
* resource conclusions;
* intended next benchmark tier;
* whether the change is docs-only, script-only, or source-code;
* whether tests may be run locally;
* which lightweight checks should be run or skipped;
* explicit instruction not to commit unless requested.

Lightweight local checks are allowed and useful when dependencies already exist, for example:

* `git diff --check`
* `python -m compileall src tests`
* `python -m pytest tests/test_gpu_backend.py -q`
* `python -m pytest tests/test_production_cli.py -q`
* `python -m pytest tests/test_production_grid.py -q`
* `python -m tailbin_cache.cli --help`

Codex should not run O2, SLURM, SSH, package install, heavy planner/build, production HDF5, or full-suite test commands locally unless explicitly requested. O2/SLURM commands are run by the user on O2, then the resulting logs/accounting/bundles are sent back for analysis. Each iteration should report which checks were run and which were skipped.

Codex may commit and push only when the user explicitly authorizes commit/push in the current prompt. Before committing or pushing, Codex should run feasible lightweight local checks, at minimum `git diff --check` and `git status`. For docs/O2 script-only changes, committing and pushing after checks pass is acceptable when explicitly authorized. For `src/` or `tests/` changes, Codex should ask before pushing unless the prompt explicitly says to push source/test changes.

## Standard Codex Prompt Skeleton

```text
You are working in my local Tailbin repository.

Goal: <one sentence goal>

O2 evidence:
<job ids, states, logs, accounting, result summary>

Target grid:
<name, Kmax, base point count, alpha table count, age constraint mode, T/T_b sampling>

Resource conclusions:
<memory/cpu/wall-time/GPU conclusions>

Allowed edits: <files or directories>

Forbidden edits: <files or directories>

Rules:

* Do not install packages unless explicitly asked.
* Do not run O2/SLURM/SSH commands locally.
* Do not run long tests or the full test suite.
* Do not use git add, git commit, or git push unless explicitly authorized in this prompt.
* Keep generated outputs ignored.
* Preserve O2 resource-efficiency rules.

Requested implementation: <numbered tasks>

After finishing, report:

1. Files changed.
2. Summary of changes.
3. Git status.
4. Diff summary.
5. Any assumptions or follow-up commands needed.
6. Lightweight checks run or skipped.
```

## Current Known Tailbin O2 State

* Repo O2 location: `/n/data1/hms/sysbio/hormoz/users/javi/tailbin-cache`
* Correct O2 Python stack: `module load gcc/14.2.0`, `module load python/3.13.1`
* Project-local environment: `.venv_o2`
* Old environment `/home/jae37/pythonEnv_3.10.11` is broken and should not be used.
* Commit `095935416df56f7e40a3357efcb0635d76da9313` completed the first O2 smoke/audit loop successfully.
* CPU smoke job `41876802` completed successfully.
* GPU audit job `41876804` completed successfully on a Tesla V100S with about `6.8x` GPU speedup and max relative CPU-vs-GPU error about `8.4e-08`.
* Collector job `41876805` completed successfully and produced a result bundle.
* The previous GPU failure was fixed by changing complex CuPy scatter-add into separate real/imaginary `float64` scatter-adds.
* Collector bundles should include current-run logs and GPU monitor logs by default, selected by CPU/GPU/collector job IDs. Historical `logs/` files are included only with `INCLUDE_ALL_LOGS=1`, and historical GPU monitor logs should not be swept into normal smoke/audit bundles.
* Next likely task is moving from smoke/audit into a small calibrated benchmark tier while keeping result bundles current-run focused.
