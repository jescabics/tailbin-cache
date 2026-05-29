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

* Do not commit or push unless explicitly instructed.
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
* Do not use `git add`, `git commit`, or `git push` unless explicitly requested.
* O2 resource requests must be measurement-driven.
* O2 scripts must use SLURM for heavy work and avoid login-node compute.

Current O2 environment facts:

* Existing O2 Python stack is `gcc/14.2.0` plus `python/3.13.1` plus repo-local `.venv_o2`.
* Old `/home/jae37/pythonEnv_3.10.11` must not be used.
* Current known GPU module is `cuda/12.8`, not `cuda/11.7`.
