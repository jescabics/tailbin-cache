import numpy as np
import pytest

from tailbin_cache.builder import BuildConfig, ErrorBudget
from tailbin_cache.grid import ParameterPoint
from tailbin_cache.model import make_model_params
from tailbin_cache.theta_bundle import _make_fcfg
from tailbin_cache.gpu_backend import CuPyComplexPGFGridEngine, CuPyPGFConfig


def test_cupy_convolution_scatter_uses_supported_dtypes():
    try:
        import cupy as cp
    except Exception:
        pytest.skip("CuPy is not installed")

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device available")
    except Exception as exc:
        pytest.skip(f"CUDA device query failed: {exc}")

    p = ParameterPoint(
        R=0.745,
        T=1.0,
        theta_f=0.0,
        N=10000.0,
        depth=20,
        u=20.0,
        ploidy_factor=2.0,
        lam=1.0,
        condition_on_survival=True,
    )
    build = BuildConfig(Kmax=200, n_bins=8, min_steps=2, max_steps=4, steps_per_time=1.0)
    params = make_model_params(p, alpha=0.05)
    fcfg = _make_fcfg(build, ErrorBudget())
    gpu = CuPyComplexPGFGridEngine(params, fcfg, gpu_config=CuPyPGFConfig(batch_size=2))

    q_np = np.vstack([gpu.Q0_np, gpu.Q0_np * (1.0 + 0.05j)]).astype(np.complex128)
    q = cp.asarray(q_np)
    conv = gpu._conv(q)

    expected = np.zeros_like(q_np)
    for i, j, k, w in zip(gpu.ii_np, gpu.jj_np, gpu.kk_np, gpu.ww_np):
        expected[:, k] += w * q_np[:, i] * q_np[:, j]

    assert conv.dtype == cp.complex128
    np.testing.assert_allclose(cp.asnumpy(conv), expected, rtol=1e-12, atol=1e-12)
