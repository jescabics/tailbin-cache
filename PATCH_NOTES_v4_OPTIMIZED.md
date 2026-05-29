# v4 optimized exact/GPU-ready branch

This branch keeps the finite-depth PGF + damped Cauchy/FFT target.  It is not a
saddlepoint, negative-binomial, normal, or interpolation approximation.

Changes:

- Default exact FFT start changed to minimal certified settings:
  `base_node_factor=1.0`, `refine_node_factor=2.0`.
- Two-sided real-CGF Chernoff tail certificates retained and enabled by default.
  These only certify tail saturation/constant rows; they are not used as CDF
  approximations.
- Optional `pgf_backend: cupy` added for O2 GPU benchmarking.  This evaluates
  complex contour-node PGFs on CUDA via CuPy and preserves theta bundling.
- O2 SLURM scripts and GPU audit example added.
- Existing CPU tests pass; CPU smoke HDF5 build certified 16/16 tables with no
  refinement failures.

Caveat:

The CuPy backend must be validated on an O2 GPU node.  Local CI here has no GPU,
so the package includes a CPU-vs-GPU audit script that should be the first O2
run before production.
