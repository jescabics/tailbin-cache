# Production Status

This is the cleaned standalone cache-builder release with automatic refinement ladder and stable complex-PGF fallback and cost-aware alpha-monotone preflight selection.

Key properties:

- self-contained package with vendored finite-depth PGF backend under `src/readsampled_cdf/`;
- diploid latent frequency scaling `m/(2N)` by default;
- age pruning `T + theta_f/u <= 100` by default;
- no saddlepoint production path;
- HDF5 lookup-cache output;
- adaptive moment-certified constants/prefix tables;
- automatic refine-until-certified ladder for nonconstant tables;
- stable fallback for non-finite complex PGF nodes in small-prefix Cauchy/FFT tables;
- explicit hard-table deferral when configured node/time budgets are exceeded.

A cache shard is production-acceptable only if every required table is present and certified. If `n_refinement_failures > 0`, rerun those hard tables with a higher budget/cap or handle them separately before using the cache for final inference.
