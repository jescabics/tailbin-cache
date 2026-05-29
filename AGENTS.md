# Tailbin Repository Instructions

This is the Tailbin cache project.

Before making nontrivial changes, read:

* `docs/WORKFLOW.md`
* `docs/O2_RUNBOOK.md`
* `docs/O2_RESOURCE_CALIBRATION.md`
* `docs/ITERATION_HANDOFF_TEMPLATE.md`

Project roles:

* ChatGPT handles mathematical planning, numerical-method design, benchmark interpretation, and high-level debugging strategy.
* Codex edits the local repository only according to explicit implementation instructions.
* O2 runs computational jobs.

Repository rules:

* Codex may commit and push only when the user explicitly authorizes commit/push in the current prompt.
* Before committing or pushing, run feasible lightweight local checks, at minimum `git diff --check` and `git status`.
* Do not run O2, SLURM, or SSH commands locally.
* Lightweight local validation is allowed when dependencies already exist and the user has not forbidden it. Useful checks include:
  * `git diff --check`
  * `python -m compileall src tests`
  * `python -m pytest tests/test_gpu_backend.py -q`
  * `python -m pytest tests/test_production_cli.py -q`
  * `python -m pytest tests/test_production_grid.py -q`
  * `python -m tailbin_cache.cli --help`
* Do not install packages unless explicitly instructed.
* Do not run the full test suite unless explicitly requested.
* Do not run long tests, heavy planners/builds, production HDF5 cache generation, or heavy compute locally unless explicitly instructed.
* Do not create HDF5 production outputs locally unless explicitly instructed.
* Do not commit generated logs, outputs, results, HDF5 files, virtual environments, or bundles.
* Do not use `git add`, `git commit`, or `git push` unless explicitly authorized in the current prompt.
* For small docs/O2 script-only changes, Codex may commit and push after lightweight checks pass when explicitly authorized.
* For `src/` or `tests/` changes, Codex should report what changed and either ask before pushing or push only if the prompt explicitly says to push source/test changes.
* O2 resource requests must be measurement-driven.
* O2 scripts must use SLURM for heavy work and avoid login-node compute.
* After smoke/audit is green, use resource calibration before production pilot decisions.
* Prefer `/usr/bin/time -v` calibration output when `sacct MaxRSS` is missing or unreliable.

Current O2 environment facts:

* Existing O2 Python stack is `gcc/14.2.0` plus `python/3.13.1` plus repo-local `.venv_o2`.
* Old `/home/jae37/pythonEnv_3.10.11` must not be used.
* Current known GPU module is `cuda/12.8`, not `cuda/11.7`.
