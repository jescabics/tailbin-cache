# Tail-Bin Lookup Cache Builder

Standalone production-oriented builder for reusable lookup caches of cumulative read-frequency tail-bin count distributions.

For each grid point

```text
(R, T, theta_f, N, depth, alpha)
```

it builds a lookup table

```text
z(k) = Phi^{-1}( P(Y_alpha <= k) ),   k = 0,...,Kmax
```

where `Y_alpha` is the read-sampled cumulative tail-bin mutation count above cutoff `alpha`.
The default model uses diploid latent frequency scaling, so a clone of size `m` has latent allele frequency

```text
m / (2N)
```

and the grid enforces the biological age constraint

```text
T + Tb <= 100,    theta_f = u * Tb.
```

The package is self-contained: the finite-depth PGF backend is vendored under `src/readsampled_cdf/`. It does not depend on the old exploratory packages artifacts.


## Current release notes

This tarball is version 1.3.0. It includes the v1.2 stable complex-PGF fallback and a v1.3 cost-aware alpha-monotone preflight selection fix.

## Production workflow

```bash
pip install -e .

tailbin-cache init-config --production --output grid.yaml

tailbin-cache estimate --config grid.yaml

tailbin-cache plan --config grid.yaml --output-dir plan_out

tailbin-cache plan-shards --config grid.yaml --output-dir shard_plan --n-shards 20

# Launch one process per shard, for example shard 0:
tailbin-cache build-hdf5 \
  --config grid.yaml \
  --output cache_shard_00.h5 \
  --n-shards 20 \
  --shard-index 0
```

Inspect a table value:

```bash
tailbin-cache inspect-hdf5 cache_shard_00.h5 --index 0 --k 50000
```

## Accuracy policy

The builder is designed around explicit z-space error control because the Gaussian copula consumes

```text
z = Phi^{-1}(F).
```

Each nonconstant table is built by damped Cauchy/FFT coefficient extraction from the finite-depth PGF. The builder compares a base table to a refined table and records:

- maximum z-space disagreement,
- maximum CDF-space disagreement,
- analytic Cauchy alias indicators,
- storage/quantization error,
- monotonicity checks,
- finite-z checks,
- moment-certified saturated tails when available.

A table is marked `certified=true` only when it satisfies the configured tolerances. This is an auditable numerical certificate, not formal interval arithmetic over every floating-point operation.

Recommended starting production tolerance:

```yaml
build:
  target_max_abs_z_error: 0.002
  target_max_abs_cdf_error: 1.0e-6
  clip_eps: 1.0e-12
```

## Refinement ladder

The production builder has a refine-until-certified ladder. If a nonconstant table does not pass the first base/refined certificate, it is rebuilt with stronger numerical settings:

- more Fourier/Cauchy nodes,
- larger alias-exclusion eta,
- more frequency-volume bins,
- more time-integration steps.

If all refinement levels fail or a declared node/time budget is exceeded, the table is not silently used. It is marked as a refinement failure / deferred hard table. A production cache should be accepted for inference only when all required tables are certified.

Key config block:

```yaml
refinement:
  enabled: true
  max_levels: 5
  node_factor_growth: 1.6
  alias_eta_growth: 4.0
  n_bins_increment: 4
  steps_growth: 1.25
  max_steps_growth: 1.35
  max_seconds_per_bundle: 1800
  fail_on_uncertified: false
```

Optional hard-table guard:

```yaml
build:
  max_refined_nodes: 262144
```

If this is set, any group requiring more refined Fourier nodes is deferred quickly instead of launching a very long build. Rerun deferred hard tables later with a higher cap or longer budget.


## Stable complex-PGF fallback

The default complex-PGF path is the fast theta-bundled batched RK4 backend.  Some small-prefix Cauchy circles can still make that low-step RK4 solve produce non-finite complex nodes.  In this release those cases are handled automatically:

1. try the normal fast batched RK4 backend;
2. if any complex PGF node is non-finite, retry the same finite-depth target with a finer batched RK4 fallback;
3. for small Fourier grids, optionally retry with scalar `solve_ivp` if the finer RK4 fallback still fails;
4. certify using the same base/refined z-space and CDF-space checks before writing anything.

Relevant optional config knobs:

```yaml
build:
  stable_pgf_fallback: true
  stable_rk4_fallback_step_multiplier: 10.0
  stable_rk4_fallback_max_steps: 2000
  solve_ivp_fallback_node_cap: 4096
```

The fallback is not a surrogate distribution.  It evaluates the same finite-depth PGF target with a more stable numerical integration setting, then passes through the usual certificate gate.

## Kmax

`Kmax=50000` is supported. Lookup is cheap once the HDF5 cache exists. Build cost depends on the largest non-saturated prefix, not just global `Kmax`. If moment bounds prove that a table saturates by `k=2000`, the suffix through 50,000 is represented compactly. If the distribution has real transition mass near 50,000, the table is genuinely hard and may require a full coefficient build.

## Public CLI

```text
init-config
estimate
plan
plan-shards
build-hdf5
inspect-hdf5
```

## Acceptance rule

For production inference, require:

```text
n_refinement_failures = 0
n_tables_certified = n_tables_written
all required expected tables are present/certified
```
