# Tailbin Production Target Grid

This document records what the repository currently specifies about smoke, calibration, and production-scale Tailbin cache grids. It distinguishes the literal YAML intent from what the current CLI actually reads.

## Short Answer

A production-like grid is present in `examples/production_template.yaml`, and the imported README describes it as the starting production workflow. It is the closest thing in the repository to a production target.

However, it should still be treated as a production template, not yet as a scientifically approved final target. Several scientific choices remain open, and some flat example YAMLs use `theta_f_values`, while `src/tailbin_cache/grid.py` currently reads `theta_values` or `Tb_values`.

## Grid Semantics

`grid_from_dict()` accepts either a nested `grid:` block or top-level grid fields.

Recognized axes are:

* `R_values`, or `R_min` / `R_max` / `R_count`, using linear spacing.
* `T_values`, or `T_min` / `T_max` / `T_count`, using linear spacing.
* `theta_values`, or `Tb_values` converted by `theta_f = u * Tb`.
* `N_values`, or `N_min` / `N_max` / `N_count`, using log spacing.
* `depth_values`, or single `depth`.
* `alphas`, or `n_alpha` / `alpha_min` / `alpha_max`, using log spacing.

The age constraint is:

```text
T + theta_f / u <= max_age
```

with default `u = 20` and `max_age = 100`.

Important current mismatch: `theta_f_values` appears in several example configs, but the CLI code does not currently read that key. Those configs are therefore interpreted with the default `theta_values` unless the key is changed or the parser is updated.

## Currently Defined Grids

| Config | Intended tier | Kmax | Expanded parameter points | Dense tables after alpha | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `examples/smoke.yaml` | generated tiny smoke | 64 | 1 | 1 | Uses nested `grid:` and is read as written. |
| `examples/kmax2000_cpu_smoke.yaml` | O2 CPU smoke | 2000 | 12 intended; 32 read by current CLI | 24 intended; 64 read by current CLI | Uses `theta_f_values`; current CLI falls back to default `theta_values`. |
| `examples/o2_resource_calibration.yaml` | O2 resource calibration | 5000 | 144 intended; 288 read by current CLI | 432 intended; 864 read by current CLI | Uses `theta_f_values`; current CLI falls back to default `theta_values`. |
| `examples/kmax20_o2_gpu_probe.yaml` | GPU probe/audit config | 20000 | 13 intended; 8 read by current CLI | 260 intended; 160 read by current CLI | The audit script is currently self-contained, but this config has the same schema mismatch. |
| `examples/production_template.yaml` | production template | 50000 | 77,040 | 1,540,800 | Uses nested `grid:` and `theta_values`, so it is read as written. |
| `examples/o2_production_pilot.yaml` | provisional production-pilot subset | 50000 | 144 | 720 | Added as a bounded, explicitly provisional subset of `production_template.yaml`. |

## Parameter Axes

### Generated Smoke: `examples/smoke.yaml`

* `R`: `[0.9]`
* `T`: `[1]`
* `theta_f`: `[0]`
* `N`: `[10000]`
* `depth`: `[100]`
* `alpha`: `[0.05]`
* `Kmax`: `64`

This is a minimal import/CLI smoke config, not a resource calibration or production target.

### CPU Smoke: `examples/kmax2000_cpu_smoke.yaml`

Literal YAML intent:

* `R`: `[0.01, 0.745]`
* `T`: `[1, 10]`
* `theta_f`: `[0, 100, 600]`
* `N`: `[10000]`
* `depth`: `[90]`
* `alpha`: `[0.05, 0.1]`
* `Kmax`: `2000`

This implies 12 expanded points and 24 dense alpha tables.

Current CLI interpretation uses default `theta_values` instead of `theta_f_values`, giving 32 expanded points and 64 dense alpha tables after age filtering.

### O2 Resource Calibration: `examples/o2_resource_calibration.yaml`

Literal YAML intent:

* `R`: `[0.01, 0.745, 0.95]`
* `T`: `[1, 10, 34]`
* `theta_f`: `[0, 100, 600, 1000]`
* `N`: `[10000, 1000000]`
* `depth`: `[90, 120]`
* `alpha`: `[0.05, 0.1, 0.5]`
* `Kmax`: `5000`

This implies 144 expanded points and 432 dense alpha tables.

Current CLI interpretation uses default `theta_values` instead of `theta_f_values`, giving 288 expanded points and 864 dense alpha tables after age filtering.

### Production Template: `examples/production_template.yaml`

* `R`: 30 linearly spaced values from `0.01` to `0.99`
* `T`: 50 linearly spaced values from `1` to `100`
* `theta_f`: `[0, 100, 200, 300, 400, 500, 600, 1000, 2000]`
* `N`: 8 log-spaced values from `10000` to `100000000`
* `depth`: `[120]`
* `alpha`: 20 log-spaced cutoffs from `0.05` to `1.0`
* `Kmax`: `50000`

The `theta_f = 2000` axis is entirely removed by the age constraint because it implies `Tb = 100`, and the minimum `T` is 1. Other theta values are partly filtered at larger `T`.

After age filtering:

* valid `(T, theta_f)` pairs: 321
* expanded parameter points: `30 * 8 * 321 = 77,040`
* dense alpha tables: `77,040 * 20 = 1,540,800`
* dense int16 values: `1,540,800 * 50,001`
* dense raw storage estimate: about `154.08 GB` before compression

## Closest Config To Intended Production

`examples/production_template.yaml` is closest to intended production because:

* it is emitted by `tailbin-cache init-config --production`;
* it uses the parser-supported nested `grid:` schema;
* it uses `theta_values`, not `theta_f_values`;
* it matches the imported README production workflow;
* it uses production-scale `Kmax = 50000`;
* it includes the refinement ladder and production-oriented fallback settings.

It should still be reviewed before being treated as final source of truth.

## Scientifically Unspecified

These choices still require user/scientific decision:

* Whether the final production axes should be exactly the imported template axes.
* Whether `R` and `T` should be linearly spaced, transformed, or concentrated in hard regimes.
* Whether `N` should remain 8 log-spaced values from `1e4` to `1e8`.
* Whether `theta_f` is the right axis, or whether the project should specify `Tb_values` and derive `theta_f = u * Tb`.
* Whether the flat example configs should be fixed to `theta_values` or the parser should accept `theta_f_values`.
* Whether production should use one depth value, multiple depth values, or empirical read-depth strata.
* Whether alpha cutoffs should be log-spaced from `0.05` to `1.0`, threshold-deduplicated by `ceil(alpha * depth)`, or matched to downstream inference bins.
* Whether `Kmax = 50000` is final for all regimes.
* Whether production tolerances should remain `target_max_abs_z_error = 0.01` and `target_max_abs_cdf_error = 1e-5`, or move toward the stricter imported README suggestion of `0.002` and `1e-6`.
* Whether CPU, GPU, or mixed backend should be used for production shards after calibration.
* What acceptance policy to use for hard/deferred rows before inference consumes the cache.

## Recommended Next Pilot Grid

Because the final production target is not fully approved, `examples/o2_production_pilot.yaml` is explicitly provisional. It samples the production template at low, middle, and high values while keeping the same `Kmax = 50000`:

* `R`: `[0.01, 0.745, 0.99]`
* `T`: `[1, 10, 34, 70, 100]`
* `theta_f`: `[0, 100, 600, 1000, 2000]`
* `N`: `[10000, 1000000, 100000000]`
* `depth`: `[120]`
* `alpha`: `[0.05, 0.1, 0.25, 0.5, 1.0]`
* `Kmax`: `50000`

After age filtering, this provisional pilot implies:

* expanded parameter points: 144
* dense alpha tables: 720

Recommended O2 calibration command:

```bash
RUN_LABEL=production_pilot_preflight \
CAL_CONFIG=examples/o2_production_pilot.yaml \
CAL_PLAN_LIMIT_BUNDLES=20 \
CAL_SHARDS=8 \
TINY_BUILD_LIMIT_BASE_POINTS=1 \
RUN_GPU_AUDIT=1 \
bash scripts/o2/submit_resource_calibration.sh
```

This is still resource calibration, not production. Review `summary.md`, `summary.json`, current-run logs, GPU audit output, and accounting before creating a true production-pilot array.
