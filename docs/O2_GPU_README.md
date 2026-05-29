# O2 GPU/cluster notes for tailbin-cache optimized exact builder

This package keeps the finite-depth PGF + damped Cauchy/FFT coefficient target.
The optimized production path is:

1. two-sided real-CGF Chernoff tail certificates;
2. minimal-node exact FFT start (`base_node_factor=1.0`, `refine_node_factor=2.0`);
3. stable CPU fallback for small non-finite complex-PGF bundles;
4. optional CuPy GPU backend for the expensive contour-node PGF evaluation.

## CPU production command

```bash
module load gcc/9.2.0
python -m pip install --user -e .
tailbin-cache plan --config config.yaml --output-dir plan_out
tailbin-cache plan-shards --config config.yaml --output-dir shards --n-shards 100
sbatch scripts/o2_cpu_array.sbatch
```

## GPU benchmark command

Use this only after installing CuPy matching the loaded CUDA version.

```bash
module load gcc/9.2.0 cuda/11.7
python -m pip install --user -e .
python -m pip install --user cupy-cuda11x
```

Set in YAML:

```yaml
build:
  pgf_backend: cupy
  batch_size: 4096
```

For correctness development, request double-precision GPUs first:

```bash
sbatch scripts/o2_gpu_probe.sbatch
```

Do not set `CUDA_VISIBLE_DEVICES`; Slurm sets it.

## Important

The GPU backend is optional and should be validated on O2 with CPU-vs-GPU audits
before production.  The package will still run exactly on CPU with
`pgf_backend: batched`.
