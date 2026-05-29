# Tailbin Workflow

This repository uses a split workflow for planning, implementation, cluster execution, and result review.

## Roles

* **ChatGPT** handles mathematical planning, numerical-method design, benchmark interpretation, and high-level debugging strategy.
* **Codex** edits the local repository according to explicit implementation instructions.
* **GitHub** stores the reviewed source of truth.
* **O2** runs computationally intensive planner, benchmark, CPU, GPU, and production jobs from the GitHub-tracked repository.
* **Result bundles** from O2 come back to ChatGPT for interpretation and next-step planning.

## Operating Loop

1. Discuss the mathematical or numerical goal with ChatGPT.
2. Convert the agreed plan into specific implementation instructions for Codex.
3. Let Codex edit the local repository.
4. Review the local diff.
5. Commit and push reviewed changes to GitHub.
6. Pull the updated repository on O2.
7. Run the relevant O2 jobs.
8. Bundle logs, configs, summaries, and result metadata.
9. Bring the result bundle back to ChatGPT for analysis.
10. Repeat.

After smoke/audit is green, run O2 resource calibration before production. Smoke/audit proves the environment and CPU/GPU paths work; resource calibration measures runtime, memory, GPU behavior, and shard-planning shape so the first production pilot uses reviewed resource requests. The next cluster step is staged representative resource calibration for `local34_diag_v1_k10000_1k`: CPU plan/selection, then `RUN_GPU_BUILD=1` representative HDF5 build, then collection. Start with `easy_smoke` before any 8-point or 40-point sample, and keep full `full100k_v1_k50000` production disabled until Grid B calibration is reviewed.

## Boundaries

* ChatGPT does not receive passwords, SSH keys, Duo codes, cluster tokens, or private credentials.
* Codex should not make broad architectural changes unless explicitly instructed.
* O2 jobs should be reproducible from committed code and committed configuration files.
* Large generated artifacts, HDF5 caches, logs, and benchmark outputs should not be committed unless explicitly requested.
* Result bundles should be kept separate from source code and shared back for analysis.

## Current Numerical Goal

Develop a production-grade tail-bin CDF/cache builder for Gaussian-copula inference using the finite-depth PGF target, with certified numerical accuracy, robust hard-regime handling, and scalable O2 execution.

## Current Optimization Direction

The current preferred direction is:

1. Keep the exact finite-depth PGF/Cauchy-FFT target.
2. Use two-sided Chernoff/CGF tail certificates to reduce false full-table rows.
3. Use minimal-node FFT starts with refinement-based certification.
4. Instrument hard rows carefully.
5. Benchmark CPU and GPU backends on O2.
6. Use O2 result bundles to decide whether GPU acceleration, stronger certificates, or a new hard-row representation is needed.
7. Use `/usr/bin/time -v` resource-calibration summaries when `sacct MaxRSS` is missing or unreliable for small jobs.

## Commit Policy

Codex may edit files when asked. Codex may commit and push only when the user explicitly authorizes commit/push in the current prompt. Before committing or pushing, Codex should run feasible lightweight local checks, at minimum `git diff --check` and `git status`. For small docs/O2 script-only changes, Codex may commit and push after checks pass when explicitly authorized. For `src/` or `tests/` changes, Codex should ask before pushing unless the prompt explicitly says to push source/test changes.

## Local Validation Policy

Codex may run lightweight local checks when dependencies already exist and the user has not forbidden them, such as `git diff --check`, targeted `pytest` files, `python -m compileall src tests`, and `python -m tailbin_cache.cli --help`. Codex should not run package installs, full test suites, heavy planners/builds, production HDF5 generation, O2/SLURM/SSH commands, or git add/commit/push unless explicitly requested.

## Related Documentation

* [O2 Runbook](O2_RUNBOOK.md): O2-specific execution, resource-request, monitoring, and SLURM guidance.
* [O2 Resource Calibration](O2_RESOURCE_CALIBRATION.md): Measurement-driven workflow for choosing memory, CPU, GPU, and wall-time requests.
* [Iteration Handoff Template](ITERATION_HANDOFF_TEMPLATE.md): Standard information to send back from O2 and standard structure for ChatGPT-to-Codex implementation prompts.
* [Production Target Grids](PRODUCTION_TARGET_GRID.md): Named scientific target grids, parser semantics, exact axes, counts, and O2 calibration commands.
