#!/usr/bin/env python
"""CPU-vs-CuPy backend audit for one small contour-node batch.

Run on O2 GPU node after installing CuPy.  This does not build a full cache; it
checks that pgf_backend='cupy' returns finite PGF values matching the CPU
batched backend for a representative contour batch.
"""
import json, time
import numpy as np
from tailbin_cache.grid import ParameterPoint
from tailbin_cache.model import make_model_params
from tailbin_cache.builder import BuildConfig, ErrorBudget
from tailbin_cache.theta_bundle import _make_fcfg
from tailbin_cache.batched_backend import BatchedComplexPGFGridEngine, BatchedPGFConfig
from tailbin_cache.gpu_backend import CuPyComplexPGFGridEngine, CuPyPGFConfig

p = ParameterPoint(R=0.745, T=10.0, theta_f=0.0, N=10000.0, depth=90, u=20.0, ploidy_factor=2.0, lam=1.0, condition_on_survival=True)
alpha = 0.05
build = BuildConfig(Kmax=20000, n_bins=24, min_steps=32, max_steps=120, steps_per_time=1.4, batch_size=512)
budget = ErrorBudget()
M = 512
radius = np.exp(-30.0 / M)
z = radius * np.exp(2j * np.pi * np.arange(M) / M)
theta = np.array([0, 100, 600, 1000], dtype=np.float64)
params = make_model_params(p, alpha)
fcfg = _make_fcfg(build, budget)

cpu = BatchedComplexPGFGridEngine(params, fcfg, batch_config=BatchedPGFConfig(batch_size=128))
gpu = CuPyComplexPGFGridEngine(params, fcfg, gpu_config=CuPyPGFConfig(batch_size=512))

# Warmup
cpu.pgf_many_theta(z[:8], theta[:2])
gpu.pgf_many_theta(z[:8], theta[:2])

t0 = time.perf_counter(); vc = cpu.pgf_many_theta(z, theta); tc = time.perf_counter() - t0
t0 = time.perf_counter(); vg = gpu.pgf_many_theta(z, theta); tg = time.perf_counter() - t0
mask = np.isfinite(vc) & np.isfinite(vg)
abs_err = np.max(np.abs(vc[mask] - vg[mask])) if np.any(mask) else float('nan')
rel_err = np.max(np.abs(vc[mask] - vg[mask]) / np.maximum(1.0, np.abs(vc[mask]))) if np.any(mask) else float('nan')
print(json.dumps({
    'M': M,
    'theta_values': theta.tolist(),
    'cpu_seconds': tc,
    'gpu_seconds': tg,
    'speedup_cpu_over_gpu': tc / tg if tg > 0 else None,
    'cpu_finite_fraction': float(np.isfinite(vc).mean()),
    'gpu_finite_fraction': float(np.isfinite(vg).mean()),
    'max_abs_err_cpu_vs_gpu': float(abs_err),
    'max_rel_err_cpu_vs_gpu': float(rel_err),
    'n_steps': int(cpu.n_steps),
}, indent=2, sort_keys=True))
