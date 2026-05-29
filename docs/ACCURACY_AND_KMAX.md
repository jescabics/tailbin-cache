# Accuracy and Kmax notes

## What accuracy is controlled?

The stored object is

\[
z(k)=\Phi^{-1}\{P(Y\le k)\}.
\]

The primary tolerance is therefore a z-space tolerance, because this is the quantity consumed by a Gaussian copula.  A table is marked certified only when the builder's recorded error indicators pass the configured budgets, for example:

```yaml
build:
  target_max_abs_z_error: 0.01
  target_max_abs_cdf_error: 1.0e-5
  clip_eps: 1.0e-12
```

For coefficient tables, the builder records base-vs-refined Cauchy/FFT disagreement, CDF disagreement, storage quantization error, monotonicity, finiteness, and alias/tail indicators.  For saturated suffixes, it records the one-sided moment bound used to certify the clipped region.

This is a numerical certificate, not interval arithmetic.  It is designed to make the error budget explicit and auditable.  Moment/tail saturation certificates are analytic inequalities conditional on the computed model moments.

## Does larger Kmax make generation slower?

Lookup after a table is built is effectively O(1).  Generation is different.

For genuinely full coefficient tables, increasing Kmax increases the number of Fourier/Cauchy nodes.  The refined node count is chosen as a power of two large enough to resolve the requested prefix.  Work is roughly proportional to the number of PGF nodes, with an FFT overhead.  So full tables become substantially harder as Kmax grows.

For constant or right-saturated prefix tables, global Kmax may matter little.  If a tail certificate proves saturation at k=300, then a cache declared to Kmax=50,000 only stores the prefix through k=299 and treats all larger k as clipped by certificate.

That is why planning matters: the real cost is driven by the largest non-saturated prefix, not just the global Kmax.
