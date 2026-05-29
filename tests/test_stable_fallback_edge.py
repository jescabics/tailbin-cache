from dataclasses import replace

import numpy as np

from tailbin_cache.builder import BuildConfig, ErrorBudget
from tailbin_cache.grid import ParameterPoint
from tailbin_cache.theta_bundle import ThetaBundleSpec, generate_theta_bundle_embedded_refinement


def test_small_prefix_edge_uses_stable_rk4_fallback():
    """Regression for the R=0.01,T=34,N=1e6,alpha=0.05 complex-node NaN edge.

    The normal low-step batched complex RK4 backend is intentionally tried first;
    it produces non-finite nodes on this damped Cauchy circle.  The production
    path must recover with the stable RK4 fallback and return finite base/refined
    CDF tables, not silently drop the bundle.
    """
    build = BuildConfig(
        Kmax=17,
        base_node_factor=2.0,
        refine_node_factor=4.0,
        base_alias_eta=30.0,
        refine_alias_eta=36.0,
        n_bins=24,
        steps_per_time=1.4,
        max_steps=120,
        batch_size=128,
        stable_pgf_fallback=True,
        stable_rk4_fallback_step_multiplier=10.0,
        stable_rk4_fallback_max_steps=2000,
    )
    budget = ErrorBudget(target_max_abs_z_error=0.01, target_max_abs_cdf_error=1.0e-5)
    bp = ParameterPoint(R=0.01, T=34.0, theta_f=0.0, N=1_000_000.0, depth=120)
    spec = ThetaBundleSpec(base_point=bp, theta_values=[100.0], alpha=0.05, alpha_index=0)

    base_cdf, ref_cdf, base_meta, ref_meta = generate_theta_bundle_embedded_refinement(spec, build, budget)

    assert base_cdf is not None
    assert ref_cdf is not None
    assert np.isfinite(base_cdf).all()
    assert np.isfinite(ref_cdf).all()
    assert ref_meta["pgf_backend_used"] == "batched_rk4_fallback"
    assert ref_meta["fallback_status"] == "rk4_fallback_succeeded"
    assert ref_meta["fast_backend_nonfinite_count"] > 0
    assert ref_meta["rk4_fallback_nonfinite_count"] == 0
