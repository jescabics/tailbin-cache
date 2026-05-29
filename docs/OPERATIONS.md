# Operations guide

## Minimal smoke build

```bash
pip install -e .
tailbin-cache init-config --output smoke.yaml
tailbin-cache build-hdf5 --config smoke.yaml --output smoke.h5 --limit-base-points 1
tailbin-cache inspect-hdf5 smoke.h5 --index 0 --k 25
```

## Production planning

```bash
tailbin-cache init-config --production --output grid.yaml
tailbin-cache estimate --config grid.yaml
tailbin-cache plan --config grid.yaml --output-dir plan_out
tailbin-cache plan-shards --config grid.yaml --output-dir shard_plan --n-shards 20
```

## Parallel build

Run these as 20 independent jobs/processes:

```bash
tailbin-cache build-hdf5 --config grid.yaml --output cache_shard_00.h5 --n-shards 20 --shard-index 0
tailbin-cache build-hdf5 --config grid.yaml --output cache_shard_01.h5 --n-shards 20 --shard-index 1
# ... through shard-index 19
```

Each shard is self-contained.  Keep the config and shard metadata with the output files for reproducibility.

## Production acceptance check

After every build, inspect the JSON summary next to the HDF5 file.  A shard is ready for inference only when:

```text
n_refinement_failures = 0
n_tables_certified = n_tables_written
certified_fraction_written = 1.0
```

If a table uses the stable fallback, its metadata records `pgf_backend_used = batched_rk4_fallback` or `solve_ivp_fallback`.  These rows are acceptable when `certified = true`; the fallback is only an integration-stability retry for the same finite-depth PGF target.
