# v1.3 cost-aware alpha-monotone patch notes

This release keeps the v1.2 stable complex-PGF fallback and adds a second production fix found during benchmark probing.

## What changed

- Higher-alpha rows now run their native moment preflight before accepting a lower-alpha monotone right-tail prefix cap.
- For each theta, the builder chooses the cheapest certified representation:
  - native constant if the higher-alpha moment bound certifies saturation,
  - the shorter of native prefix and inherited monotone prefix when both are available,
  - inherited monotone prefix when the native row would otherwise be full/hard.
- This prevents a lower-alpha prefix certificate from forcing many unnecessary higher-alpha coefficient builds.

## Regression symptom fixed

A representative base point from the Colab-style grid, `R=0.01, T=23, N=1e6, depth=120`, could spend several minutes building higher-alpha prefixes even though most higher alpha rows were moment-certified constants. After this patch it completes in about 4 seconds of package-reported build time and writes all 160 rows certified.

## Acceptance rule unchanged

A cache is acceptable for inference only when every required row is present and certified, with `n_refinement_failures = 0`. The patch changes only how certified work is selected; it does not introduce a surrogate distribution.
