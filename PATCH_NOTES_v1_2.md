# v1.2 stable-fallback patch notes

This release fixes the small-prefix complex-PGF NaN edge case found at:

- R = 0.01
- T = 34
- N = 1e6
- depth = 120
- alpha = 0.05
- theta_f = 0, 100, 200, 300, 400, 500, 600, 1000

## What changed

- Added a stable PGF fallback ladder for non-finite complex Cauchy nodes.
- First fallback uses a finer batched RK4 integration for the same finite-depth PGF target.
- Optional final fallback uses scalar `solve_ivp` for small grids.
- The embedded-refinement accelerator now falls back to independent base/refined builds if the fast embedded pass is non-finite.
- Non-finite PGF/coefficient outputs are caught before FFT tables are stored.
- Metadata records the backend actually used, fallback status, fallback steps, and non-finite counts.
- Added a regression test for the discovered R=0.01,T=34,N=1e6,alpha=0.05 edge.
- Removed old robustness-result artifacts from the packaged release.

## Validation performed

Exact edge-case one-alpha HDF5 build:

```text
n_tables_written      = 8
n_tables_certified    = 8
n_refinement_failures = 0
elapsed_seconds       ~= 15.6
```

Regression suite:

```text
4 passed
```

Rows rescued by the fallback remain subject to the same production acceptance rule:

```text
certified = true
n_refinement_failures = 0
n_tables_certified = n_tables_written
```


## Additional alpha-monotone planner fix

- Higher-alpha rows now always run their native moment preflight before accepting a lower-alpha monotone prefix cap.
- The builder chooses the cheaper certified representation: native constant if available, shorter native prefix if available, or inherited monotone prefix when it avoids a full/long table.
- This prevents lower-alpha prefix certificates from creating unnecessary higher-alpha coefficient builds.
