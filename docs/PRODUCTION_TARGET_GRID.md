# Tailbin Scientific Target Grids

This document defines the next machine-readable Tailbin target grids. These grids are for resource calibration and preflight planning only until their O2 summaries are reviewed.

Do not launch full production from this iteration.

## Parser Semantics

The grid parser now treats founder/background time axes explicitly:

* `Tb_values`, or `Tb_min` / `Tb_max` / `Tb_count`, means background time `T_b`.
* `theta_f_values`, or `theta_f_min` / `theta_f_max` / `theta_f_count`, means founder load `theta_f`.
* `theta_values` is still accepted for backward compatibility and is interpreted as `theta_f`.
* If `T_b` is provided, `theta_f = u * T_b`.
* If `theta_f` is provided, `T_b = theta_f / u` for age constraints.
* Specify only one of `Tb_values`, `theta_f_values`, or `theta_values` in a config.

Age constraints are explicit:

* `age_constraint_mode: upper_bound` uses `T + T_b <= max_age`.
* `age_constraint_mode: exact` uses paired diagonal points with `T = age_exact - T_b`.
* Exact diagonal grids are not built as a Cartesian product followed by floating-point equality filtering.

Shared scientific constants for the current targets:

* `lambda = 1`
* `u = 20`
* `theta_f = 20 * T_b`
* `depth = 90`
* `Kmax` is inclusive, so each dense table stores `k = 0, ..., Kmax`.
* alpha cutoffs are 20 log-spaced values from `0.05` to `1.0` inclusive.

## Target Summary

| Target grid | Purpose | Constraint | Kmax | Base parameter points | Dense alpha tables | Dense raw storage |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `local34_diag_v1_k10000_1k` | next serious bounded resource/pipeline pilot | `T + T_b = 34` exact diagonal | 10,000 | 1,000 | 20,000 | 400,040,000 bytes, about 400 MB / 381.5 MiB |
| `full100k_v1_k50000` | near-term full scientific production target | `T + T_b <= 100` | 50,000 | 505,000 | 10,100,000 | 1,010,020,200,000 bytes, about 1.01 TB / 940.6 GiB |

The full target is larger than the earlier rough 100,000-point target because the scientific target now fixes `R_count = 10`, `N_count = 10`, and all integer `T` / `T_b` pairs under the age constraint. This gives:

```text
10 R values * 10 N values * 5050 valid T/T_b pairs * 1 depth = 505,000 base points
505,000 base points * 20 alpha cutoffs = 10,100,000 dense tables
```

Adaptive/prefix storage may be much smaller than dense raw storage, but dense raw storage is the conservative upper-bound reference.

## Target A: full100k_v1_k50000

Config:

```text
examples/full100k_v1_k50000.yaml
```

Purpose: near-term full scientific production target.

Parameters:

* `Kmax = 50000`
* `depth = 90`
* `lambda = 1`
* `u = 20`
* `R`: 10 linearly spaced values from `0.01` to `0.99` inclusive
* `N`: 10 log-spaced values from `1e4` to `1e8` inclusive
* `T`: integer values from `1` to `100` inclusive
* `T_b`: integer values from `0` to `100` inclusive
* `theta_f = 20 * T_b`
* constraint: `T + T_b <= 100`
* alpha cutoffs: 20 log-spaced values from `0.05` to `1.0` inclusive

Expected count:

```text
valid T/T_b pairs = 100 + 99 + ... + 1 = 5050
base parameter points = 10 * 10 * 5050 * 1 = 505,000
dense alpha tables = 505,000 * 20 = 10,100,000
values per dense table = 50,001
raw dense bytes = 10,100,000 * 50,001 * 2 = 1,010,020,200,000
```

### Exact Axis Values

`R`:

```text
[0.01, 0.11888888888888889, 0.22777777777777777, 0.33666666666666667, 0.44555555555555554, 0.5544444444444444, 0.6633333333333333, 0.7722222222222221, 0.8811111111111111, 0.99]
```

`N`:

```text
[10000, 27825.594022071243, 77426.36826811278, 215443.46900318822, 599484.2503189409, 1668100.537200059, 4641588.833612777, 12915496.650148827, 35938136.63804626, 100000000]
```

`T`:

```text
[1, 2, 3, ..., 100]
```

`T_b`:

```text
[0, 1, 2, ..., 100], filtered by T + T_b <= 100
```

`theta_f`:

```text
[0, 20, 40, ..., 2000], filtered by T + T_b <= 100
```

`alpha`:

```text
[0.05, 0.058538995686138975, 0.06853628031883595, 0.08024090035856697, 0.09394443439884113, 0.10998825680021057, 0.1287720418070694, 0.15076371999678687, 0.17651113509036337, 0.20665569151220547, 0.24194833267898122, 0.283268248059268, 0.33164477502323264, 0.38828304108831085, 0.45459398534539097, 0.532229506941571, 0.62312361621777, 0.729540613634067, 0.8541314966877565, 1.0]
```

## Target B: local34_diag_v1_k10000_1k

Config:

```text
examples/local34_diag_v1_k10000_1k.yaml
```

Purpose: local, bounded pilot grid around patient age 34, intended as the next serious resource/pipeline pilot.

Parameters:

* `Kmax = 10000`
* `depth = 90`
* `lambda = 1`
* `u = 20`
* `R`: 10 linearly spaced values from `0.01` to `1.0` inclusive
* `N`: 10 log-spaced values from `1e4` to `1e8` inclusive
* `T_b`: 10 linearly spaced values from `0` to `20` inclusive
* `T = 34 - T_b`
* `theta_f = 20 * T_b`
* constraint: `T + T_b = 34` exactly
* alpha cutoffs: 20 log-spaced values from `0.05` to `1.0` inclusive

Important: this grid samples only the `T_b <= 20` segment of the age-34 diagonal. It does not use `T + T_b <= 34`.

Expected count:

```text
diagonal T/T_b pairs = 10
base parameter points = 10 * 10 * 10 * 1 = 1,000
dense alpha tables = 1,000 * 20 = 20,000
values per dense table = 10,001
raw dense bytes = 20,000 * 10,001 * 2 = 400,040,000
```

### Exact Axis Values

`R`:

```text
[0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.0]
```

`N`:

```text
[10000, 27825.594022071243, 77426.36826811278, 215443.46900318822, 599484.2503189409, 1668100.537200059, 4641588.833612777, 12915496.650148827, 35938136.63804626, 100000000]
```

Paired age diagonal:

```text
T_b = 0.0000000000, T = 34.0000000000, theta_f = 0.0000000000
T_b = 2.2222222222, T = 31.7777777778, theta_f = 44.4444444444
T_b = 4.4444444444, T = 29.5555555556, theta_f = 88.8888888889
T_b = 6.6666666667, T = 27.3333333333, theta_f = 133.3333333333
T_b = 8.8888888889, T = 25.1111111111, theta_f = 177.7777777778
T_b = 11.1111111111, T = 22.8888888889, theta_f = 222.2222222222
T_b = 13.3333333333, T = 20.6666666667, theta_f = 266.6666666667
T_b = 15.5555555556, T = 18.4444444444, theta_f = 311.1111111111
T_b = 17.7777777778, T = 16.2222222222, theta_f = 355.5555555556
T_b = 20.0000000000, T = 14.0000000000, theta_f = 400.0000000000
```

`alpha`:

```text
[0.05, 0.058538995686138975, 0.06853628031883595, 0.08024090035856697, 0.09394443439884113, 0.10998825680021057, 0.1287720418070694, 0.15076371999678687, 0.17651113509036337, 0.20665569151220547, 0.24194833267898122, 0.283268248059268, 0.33164477502323264, 0.38828304108831085, 0.45459398534539097, 0.532229506941571, 0.62312361621777, 0.729540613634067, 0.8541314966877565, 1.0]
```

## Next O2 Resource Calibration Commands

Run resource calibration only. Do not run full production.

First run the local age-34 diagonal representative calibration. This replaces the older two-point smoke-style build; the workflow plans all 1,000 Grid B base points, selects a deterministic representative 40-base-point sample, and builds only that selected sample.

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

Representative selection includes predicted easiest, median, and hardest work-proxy points; low/high `R`; low/high `N`; `T_b = 0`; `T_b = 20`; intermediate `T_b` values; and planner-predicted full or large-prefix points when available. The selected base points are written to `sample/selected_base_points.csv` and `sample/selected_base_points.json`.

Then run the full target preflight:

```bash
RUN_LABEL=full100k_v1_k50000_preflight \
CAL_CONFIG=examples/full100k_v1_k50000.yaml \
CAL_PLAN_LIMIT_BUNDLES=20 \
CAL_SHARDS=16 \
TINY_BUILD_LIMIT_BASE_POINTS=1 \
RUN_GPU_AUDIT=1 \
bash scripts/o2/submit_resource_calibration.sh
```

Review both calibration summaries before creating any production submission workflow. Full Grid A production remains disabled until the Grid B representative calibration is reviewed.

The representative Grid B HDF5 build uses `pgf_backend: batched` from `examples/local34_diag_v1_k10000_1k.yaml`, so it is CPU-based. `RUN_GPU_AUDIT=1` remains a separate GPU health/correctness check.

## Relationship To Older Templates

`examples/production_template.yaml` remains an imported production template. It is no longer the named scientific target for the next O2 calibration steps.

`examples/o2_production_pilot.yaml` remains provisional. Prefer `examples/local34_diag_v1_k10000_1k.yaml` for the next bounded pilot and `examples/full100k_v1_k50000.yaml` for full-target preflight.
