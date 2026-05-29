# Method summary

For a fixed parameter point and cutoff alpha, the target is the finite-depth read-sampled cumulative tail-bin count

\[
Y_\alpha=\#\{\text{mutations read above cutoff }\alpha\}.
\]

The cache stores

\[
z(k)=\Phi^{-1}(F(k)),\qquad F(k)=P(Y_\alpha\le k),\qquad k=0,\ldots,K_{\max}.
\]

## Coefficient extraction

Let

\[
G(z)=E[z^{Y_\alpha}]
\]

be the finite-depth PGF computed by the vendored model backend.  The probability masses are coefficients

\[
p_k=[z^k]G(z).
\]

The builder uses damped Cauchy/FFT extraction on a circle \(|z|=r<1\):

\[
p_k\approx r^{-k}\frac{1}{M}\sum_{m=0}^{M-1}G(r e^{2\pi i m/M})e^{-2\pi i mk/M}.
\]

A refined node count is used for the final table.  A base grid is recovered from embedded refined nodes to produce a two-resolution audit.


## Complex-PGF stability fallback

The first PGF evaluation attempt uses the fast theta-bundled batched RK4 backend.  If any complex Cauchy node is non-finite, the builder does not form FFT coefficients from those values.  It retries the same PGF target with a finer fixed-step RK4 backend, and for small grids can then fall back to scalar `solve_ivp`.  Successful fallback rows are still certified by the same base/refined table comparison, alias indicators, monotonicity checks, finite-z checks, and storage error budget.

## Tail certificates

For a nonnegative integer count, the builder uses one-sided bounds to avoid unnecessary coefficient work:

\[
P(Y\ge K)\le \frac{E[Y]}{K},
\]

and

\[
P(Y\ge K)\le \frac{E[(Y)_2]}{K(K-1)}.
\]

If a certificate proves the CDF is clipped beyond a prefix, the cache stores only the prefix.

## Alpha monotonicity

For cumulative tail bins, if \(\alpha_1\le \alpha_2\), then

\[
Y_{\alpha_2}\le Y_{\alpha_1}\quad\text{almost surely}.
\]

Therefore a right-tail certificate for a lower alpha propagates to higher alphas.  Also, read sampling depends on alpha only through the threshold `ceil(alpha * depth)`, so repeated thresholds are exact aliases.

## Error budget

The primary error metric is Gaussianized error:

\[
|\Delta z|=|\Phi^{-1}(F_1)-\Phi^{-1}(F_2)|.
\]

The builder compares base and refined tables, adds storage quantization error, checks monotonicity and finiteness, and records all indicators in HDF5 metadata.  Tables that fail the configured thresholds are marked uncertified.
