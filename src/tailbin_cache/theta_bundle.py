from __future__ import annotations

"""Theta-bundled dense z-cache builder.

Founder load theta_f enters the finite-depth PGF only in the final Poisson
thinning factor, not in the branching/read-sampled ODE itself.  Therefore, for
one (R,T,N,depth,alpha) and many theta_f values, we can integrate the complex
finite-depth state Q(z) once per Fourier node and then evaluate all theta_f
PGFs by cheap dot products.  This can reduce table-generation CPU by roughly
the number of founder-load values.
"""

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import hashlib, json, math, time

import numpy as np

from readsampled_cdf.fourier_reference import FourierCDFConfig
from .batched_backend import BatchedComplexPGFGridEngine, BatchedPGFConfig, StableSolveIVPThetaPGFGridEngine
from .gpu_backend import CuPyComplexPGFGridEngine, CuPyPGFConfig
from .builder import BuildConfig, ErrorBudget, cdf_to_z, dequantize_z, next_pow2, quantize_z
from .grid import ParameterPoint
from .model import make_model_params


@dataclass(frozen=True)
class ThetaBundleSpec:
    base_point: ParameterPoint  # theta_f is ignored; theta_values supply loads
    theta_values: Sequence[float]
    alpha: float
    alpha_index: int = 0

    def integration_point(self) -> ParameterPoint:
        return replace(self.base_point, theta_f=0.0)

    def key_prefix(self) -> str:
        p = self.integration_point()
        return f"{p.key()}_thetaBUNDLE_a{self.alpha_index:02d}_alpha{self.alpha:.8g}".replace('.', 'p')

    def stable_id(self) -> str:
        payload = {
            'base_point': self.integration_point().to_dict(),
            'theta_values': [float(x) for x in self.theta_values],
            'alpha': float(self.alpha),
            'alpha_index': int(self.alpha_index),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()[:20]


def _make_fcfg(build: BuildConfig, budget: ErrorBudget) -> FourierCDFConfig:
    return FourierCDFConfig(
        n_bins=int(build.n_bins),
        steps_per_time=float(build.steps_per_time),
        min_steps=int(build.min_steps),
        max_steps=int(build.max_steps),
        clip_eps=float(budget.clip_eps),
        ode_rtol=float(build.ode_rtol),
        ode_atol=float(build.ode_atol),
        use_solve_ivp=False,
    )


def _make_stable_fcfg(build: BuildConfig, budget: ErrorBudget) -> FourierCDFConfig:
    return FourierCDFConfig(
        n_bins=int(build.n_bins),
        steps_per_time=float(build.steps_per_time),
        min_steps=int(build.min_steps),
        max_steps=int(build.max_steps),
        clip_eps=float(budget.clip_eps),
        ode_rtol=float(build.ode_rtol),
        ode_atol=float(build.ode_atol),
        use_solve_ivp=True,
    )


def _stable_fallback_allowed(build: BuildConfig, node_count: int) -> bool:
    if not bool(getattr(build, "stable_pgf_fallback", True)):
        return False
    cap = getattr(build, "solve_ivp_fallback_node_cap", 4096)
    return cap is None or int(node_count) <= int(cap)


def _nonfinite_count(x: np.ndarray) -> int:
    return int(np.asarray(x).size - np.isfinite(x).sum())


class ThetaBundleCauchyGenerator:
    def __init__(self, spec: ThetaBundleSpec, build: BuildConfig, budget: ErrorBudget, *, refined: bool):
        if str(build.pgf_backend) not in ('batched', 'cupy'):
            raise ValueError('theta bundle requires pgf_backend=batched or cupy')
        self.spec = spec
        self.build = build
        self.budget = budget
        self.refined = bool(refined)
        self.Kmax = int(build.Kmax)
        factor = float(build.refine_node_factor if refined else build.base_node_factor)
        self.node_count = next_pow2(int(math.ceil(factor * (self.Kmax + 1))))
        eta = float(build.refine_alias_eta if refined else build.base_alias_eta)
        self.alias_eta = eta
        self.radius = float(math.exp(-eta / float(self.node_count)))
        self.theta_values = np.asarray([float(x) for x in spec.theta_values], dtype=np.float64)
        if str(build.pgf_backend) == 'cupy':
            self.engine = CuPyComplexPGFGridEngine(
                make_model_params(spec.integration_point(), spec.alpha),
                _make_fcfg(build, budget),
                gpu_config=CuPyPGFConfig(batch_size=int(build.batch_size)),
            )
        else:
            self.engine = BatchedComplexPGFGridEngine(
                make_model_params(spec.integration_point(), spec.alpha),
                _make_fcfg(build, budget),
                batch_config=BatchedPGFConfig(batch_size=int(build.batch_size)),
            )

    def _pgf_values_with_fallback(self, z_nodes: np.ndarray, meta: Dict[str, Any]) -> Optional[np.ndarray]:
        vals = self.engine.pgf_many_theta(z_nodes, self.theta_values)
        if np.isfinite(vals).all():
            meta["pgf_backend_used"] = str(self.build.pgf_backend)
            return vals
        meta["fast_backend_nonfinite_count"] = _nonfinite_count(vals)
        meta["fallback_reason"] = "nonfinite_complex_pgf_nodes"
        current_steps = int(getattr(self.engine, "n_steps", max(1, int(math.ceil(float(self.spec.base_point.T) * float(self.build.steps_per_time))))))
        rk4_cap = int(getattr(self.build, "stable_rk4_fallback_max_steps", 2000))
        rk4_steps = min(rk4_cap, max(current_steps + 1, int(math.ceil(current_steps * float(getattr(self.build, "stable_rk4_fallback_step_multiplier", 10.0))))))
        if str(self.build.pgf_backend) in ("batched", "cupy") and rk4_steps > current_steps:
            rk4_build = replace(self.build, min_steps=int(rk4_steps), max_steps=int(rk4_steps), steps_per_time=max(float(self.build.steps_per_time), float(rk4_steps) / max(float(self.spec.base_point.T), np.finfo(float).tiny)))
            rk4 = BatchedComplexPGFGridEngine(
                make_model_params(self.spec.integration_point(), self.spec.alpha),
                _make_fcfg(rk4_build, self.budget),
                batch_config=BatchedPGFConfig(batch_size=int(self.build.batch_size)),
            )
            vals2 = rk4.pgf_many_theta(z_nodes, self.theta_values)
            meta["rk4_fallback_steps"] = int(rk4.n_steps)
            meta["rk4_fallback_nonfinite_count"] = _nonfinite_count(vals2)
            if np.isfinite(vals2).all():
                meta["pgf_backend_used"] = "batched_rk4_fallback"
                meta["fallback_status"] = "rk4_fallback_succeeded"
                return vals2
        if not _stable_fallback_allowed(self.build, self.node_count):
            meta["pgf_backend_used"] = str(self.build.pgf_backend)
            meta["fallback_status"] = "not_attempted_cap_or_disabled"
            return None
        stable = StableSolveIVPThetaPGFGridEngine(
            make_model_params(self.spec.integration_point(), self.spec.alpha),
            _make_stable_fcfg(self.build, self.budget),
        )
        vals = stable.pgf_many_theta(z_nodes, self.theta_values)
        meta["pgf_backend_used"] = "solve_ivp_fallback"
        meta["fallback_status"] = "solve_ivp_attempted"
        meta["fallback_nonfinite_count"] = _nonfinite_count(vals)
        if not np.isfinite(vals).all():
            return None
        return vals

    def generate(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        t0 = time.perf_counter()
        M = int(self.node_count)
        use_sym = bool(getattr(self.build, "use_conjugate_symmetry", True)) and (M % 2 == 0)
        if use_sym:
            n_unique_full = M // 2 + 1
            mgrid = np.arange(n_unique_full, dtype=np.float64)
        else:
            n_unique_full = M
            mgrid = np.arange(M, dtype=np.float64)
        n_eval = int(n_unique_full)
        partial = False
        if self.build.node_budget is not None:
            n_eval = min(int(self.build.node_budget), int(n_unique_full))
            partial = n_eval < int(n_unique_full)
            mgrid = mgrid[:n_eval]
        z_nodes = self.radius * np.exp(2j * np.pi * mgrid / float(M))
        meta_probe: Dict[str, Any] = {}
        vals_unique = self._pgf_values_with_fallback(z_nodes, meta_probe)
        elapsed = float(time.perf_counter() - t0)
        mean_node = float(elapsed / max(1, n_eval))
        meta: Dict[str, Any] = {
            'method': 'theta_bundled_damped_cauchy_fft_coefficients',
            'refined': bool(self.refined),
            'Kmax': self.Kmax,
            'node_count': M,
            'nodes_evaluated': int(n_eval),
            'unique_nodes_required': int(n_unique_full),
            'full_circle_node_count': int(M),
            'used_conjugate_symmetry': bool(use_sym),
            'partial_build': bool(partial),
            'radius': float(self.radius),
            'alias_eta': float(self.alias_eta),
            'analytic_alias_cdf_bound': float(math.exp(-float(self.alias_eta))),
            'seconds': elapsed,
            'mean_seconds_per_node_per_theta_bundle': mean_node,
            'estimated_full_seconds': float(mean_node * n_unique_full),
            'n_theta': int(self.theta_values.size),
            'theta_values': [float(x) for x in self.theta_values],
            'pgf_backend': str(self.build.pgf_backend),
            'batch_size': int(self.build.batch_size),
        }
        meta.update(meta_probe)
        if vals_unique is None:
            meta['status'] = 'nonfinite_complex_pgf_nodes'
            return None, meta
        if partial:
            return None, meta
        if use_sym:
            vals = np.empty((self.theta_values.size, M), dtype=np.complex128)
            vals[:, : n_unique_full] = vals_unique
            for m in range(1, M // 2):
                vals[:, M - m] = np.conj(vals_unique[:, m])
        else:
            vals = vals_unique
        fft_vals = np.fft.fft(vals, axis=1) / float(M)
        n = np.arange(self.Kmax + 1, dtype=np.float64)
        coeff = np.real(fft_vals[:, : self.Kmax + 1] * (self.radius ** (-n))[None, :])
        if not np.isfinite(coeff).all():
            meta['status'] = 'nonfinite_cauchy_coefficients'
            meta['nonfinite_coefficient_count'] = _nonfinite_count(coeff)
            return None, meta
        negative_mass = np.sum(np.minimum(coeff, 0.0), axis=1)
        coeff = np.maximum(coeff, 0.0)
        cdf = np.cumsum(coeff, axis=1)
        cdf = np.clip(cdf, 0.0, 1.0)
        meta.update({
            'negative_mass_clipped_by_theta': [float(x) for x in negative_mass],
            'prefix_mass_at_Kmax_by_theta': [float(x) for x in cdf[:, -1]],
            'monotone_cdf_by_theta': [bool(np.all(np.diff(cdf[h]) >= -1e-12)) for h in range(cdf.shape[0])],
        })
        return cdf, meta


def generate_theta_bundle_embedded_refinement(spec: ThetaBundleSpec, build: BuildConfig, budget: ErrorBudget) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any], Dict[str, Any]]:
    """Generate refined and embedded-base CDFs with one PGF node pass.

    embedded-refinement speed win: the usual two-resolution audit builds a base table and a
    refined table separately.  When the refined Fourier grid is an integer
    multiple of the base grid, the base grid nodes are a subset of the refined
    nodes if we use the refined Cauchy radius for both.  We can therefore:

      1. evaluate the refined grid once,
      2. compute the refined coefficients from all refined nodes,
      3. compute the coarse/base coefficients from the subset of refined nodes.

    This removes the separate base PGF integration.  The base alias bound is
    recomputed as r_ref**M_base.  This remains an auditable two-grid numerical
    check for the same finite PGF target; it is not a new distributional
    approximation.  If the node counts are not nested or a node_budget is set,
    callers should fall back to two separate generator calls.
    """
    Kmax = int(build.Kmax)
    M_base = next_pow2(int(math.ceil(float(build.base_node_factor) * (Kmax + 1))))
    M_ref = next_pow2(int(math.ceil(float(build.refine_node_factor) * (Kmax + 1))))
    if M_ref < M_base or M_ref % M_base != 0 or build.node_budget is not None:
        base_cdf, base_meta = ThetaBundleCauchyGenerator(spec, build, budget, refined=False).generate()
        ref_cdf, ref_meta = ThetaBundleCauchyGenerator(spec, build, budget, refined=True).generate()
        return base_cdf, ref_cdf, base_meta, ref_meta
    t0 = time.perf_counter()
    M = int(M_ref)
    eta_ref = float(build.refine_alias_eta)
    radius = float(math.exp(-eta_ref / float(M)))
    use_sym = bool(getattr(build, "use_conjugate_symmetry", True)) and (M % 2 == 0)
    if use_sym:
        n_unique_full = M // 2 + 1
        mgrid = np.arange(n_unique_full, dtype=np.float64)
    else:
        n_unique_full = M
        mgrid = np.arange(M, dtype=np.float64)
    z_nodes = radius * np.exp(2j * np.pi * mgrid / float(M))
    engine = BatchedComplexPGFGridEngine(
        make_model_params(spec.integration_point(), spec.alpha),
        _make_fcfg(build, budget),
        batch_config=BatchedPGFConfig(batch_size=int(build.batch_size)),
    )
    vals_unique = engine.pgf_many_theta(z_nodes, np.asarray(spec.theta_values, dtype=np.float64))
    elapsed = float(time.perf_counter() - t0)
    if not np.isfinite(vals_unique).all():
        # The embedded path is an acceleration only.  If the fast RK backend is
        # unstable on this complex circle, fall back to independent base/refined
        # generators, which can switch to the stable solve_ivp backend for small
        # tables.  This keeps certification semantics unchanged.
        base_cdf, base_meta = ThetaBundleCauchyGenerator(spec, build, budget, refined=False).generate()
        ref_cdf, ref_meta = ThetaBundleCauchyGenerator(spec, build, budget, refined=True).generate()
        base_meta['embedded_refinement_fallback'] = 'separate_base_refined_due_to_nonfinite_fast_nodes'
        ref_meta['embedded_refinement_fallback'] = 'separate_base_refined_due_to_nonfinite_fast_nodes'
        base_meta['embedded_fast_nonfinite_count'] = _nonfinite_count(vals_unique)
        ref_meta['embedded_fast_nonfinite_count'] = _nonfinite_count(vals_unique)
        return base_cdf, ref_cdf, base_meta, ref_meta
    if use_sym:
        vals_ref = np.empty((len(spec.theta_values), M), dtype=np.complex128)
        vals_ref[:, :n_unique_full] = vals_unique
        for m in range(1, M // 2):
            vals_ref[:, M - m] = np.conj(vals_unique[:, m])
    else:
        vals_ref = vals_unique
    def coeffs_from_vals(vals: np.ndarray, M_use: int, radius_use: float) -> Tuple[np.ndarray, List[float], List[float], List[bool]]:
        fft_vals = np.fft.fft(vals, axis=1) / float(M_use)
        n = np.arange(Kmax + 1, dtype=np.float64)
        coeff = np.real(fft_vals[:, : Kmax + 1] * (radius_use ** (-n))[None, :])
        neg = np.sum(np.minimum(coeff, 0.0), axis=1)
        coeff = np.maximum(coeff, 0.0)
        cdf = np.cumsum(coeff, axis=1)
        cdf = np.clip(cdf, 0.0, 1.0)
        return cdf, [float(x) for x in neg], [float(x) for x in cdf[:, -1]], [bool(np.all(np.diff(cdf[h]) >= -1e-12)) for h in range(cdf.shape[0])]
    ref_cdf, ref_neg, ref_mass, ref_mono = coeffs_from_vals(vals_ref, M_ref, radius)
    stride = M_ref // M_base
    vals_base = vals_ref[:, ::stride]
    # vals_base is exactly the M_base-node Cauchy grid on the same radius.
    base_cdf, base_neg, base_mass, base_mono = coeffs_from_vals(vals_base, M_base, radius)
    if (not np.isfinite(ref_cdf).all()) or (not np.isfinite(base_cdf).all()):
        base_cdf2, base_meta2 = ThetaBundleCauchyGenerator(spec, build, budget, refined=False).generate()
        ref_cdf2, ref_meta2 = ThetaBundleCauchyGenerator(spec, build, budget, refined=True).generate()
        base_meta2['embedded_refinement_fallback'] = 'separate_base_refined_due_to_nonfinite_coefficients'
        ref_meta2['embedded_refinement_fallback'] = 'separate_base_refined_due_to_nonfinite_coefficients'
        return base_cdf2, ref_cdf2, base_meta2, ref_meta2
    common = {
        'method': 'theta_bundled_embedded_damped_cauchy_fft_coefficients',
        'Kmax': Kmax,
        'nodes_evaluated': int(n_unique_full),
        'unique_nodes_required': int(n_unique_full),
        'used_conjugate_symmetry': bool(use_sym),
        'partial_build': False,
        'radius': float(radius),
        'seconds': elapsed,
        'mean_seconds_per_node_per_theta_bundle': float(elapsed / max(1, n_unique_full)),
        'estimated_full_seconds': float(elapsed),
        'n_theta': int(len(spec.theta_values)),
        'theta_values': [float(x) for x in spec.theta_values],
        'pgf_backend': str(build.pgf_backend),
        'batch_size': int(build.batch_size),
        'embedded_refinement': True,
        'base_node_count_from_refined_subset': int(M_base),
        'refined_node_count': int(M_ref),
    }
    base_meta = dict(common)
    base_meta.update({
        'refined': False,
        'node_count': int(M_base),
        'full_circle_node_count': int(M_base),
        'alias_eta': float(-math.log(max(radius ** M_base, np.finfo(float).tiny))),
        'analytic_alias_cdf_bound': float(radius ** M_base),
        'negative_mass_clipped_by_theta': base_neg,
        'prefix_mass_at_Kmax_by_theta': base_mass,
        'monotone_cdf_by_theta': base_mono,
        'notes': 'Embedded coarse table: uses every stride-th refined node at the refined radius; no extra PGF integration.',
    })
    ref_meta = dict(common)
    ref_meta.update({
        'refined': True,
        'node_count': int(M_ref),
        'full_circle_node_count': int(M_ref),
        'alias_eta': float(eta_ref),
        'analytic_alias_cdf_bound': float(math.exp(-eta_ref)),
        'negative_mass_clipped_by_theta': ref_neg,
        'prefix_mass_at_Kmax_by_theta': ref_mass,
        'monotone_cdf_by_theta': ref_mono,
    })
    return base_cdf, ref_cdf, base_meta, ref_meta


class DenseThetaBundleBuilder:
    """Build dense z-cache files for many theta_f values reusing one ODE pass."""

    def __init__(self, spec: ThetaBundleSpec, build: BuildConfig, budget: ErrorBudget):
        self.spec = spec
        self.build = build
        self.budget = budget

    def build_bundle(self, output_dir: str | Path, *, table_index_prefix: str = '') -> Dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        if bool(getattr(self.build, "use_embedded_refinement", True)):
            base_cdf, ref_cdf, base_meta, ref_meta = generate_theta_bundle_embedded_refinement(self.spec, self.build, self.budget)
        else:
            base_cdf, base_meta = ThetaBundleCauchyGenerator(self.spec, self.build, self.budget, refined=False).generate()
            ref_cdf, ref_meta = ThetaBundleCauchyGenerator(self.spec, self.build, self.budget, refined=True).generate()
        bundle_meta: Dict[str, Any] = {
            'format': 'tailbin_theta_bundle_summary_v1_0',
            'bundle_id': self.spec.stable_id(),
            'bundle_key': self.spec.key_prefix(),
            'spec': {
                'base_point': self.spec.integration_point().to_dict(),
                'theta_values': [float(x) for x in self.spec.theta_values],
                'alpha': float(self.spec.alpha),
                'alpha_index': int(self.spec.alpha_index),
            },
            'build_config': asdict(self.build),
            'error_budget': asdict(self.budget),
            'base_meta': base_meta,
            'refined_meta': ref_meta,
            'tables': [],
        }
        if base_cdf is None or ref_cdf is None:
            bundle_meta.update({'certified': False, 'status': 'partial_timing_probe', 'seconds': float(time.perf_counter()-t0)})
            (out / f'{table_index_prefix}{self.spec.key_prefix()}_bundle.json').write_text(json.dumps(bundle_meta, indent=2, sort_keys=True))
            return bundle_meta
        theta_values = [float(x) for x in self.spec.theta_values]
        all_cert = True
        for h, theta in enumerate(theta_values):
            point = replace(self.spec.integration_point(), theta_f=float(theta))
            # Reuse TableSpec-like file name without importing to avoid cycles.
            key = f"{point.key()}_a{self.spec.alpha_index:02d}_alpha{self.spec.alpha:.8g}".replace('.', 'p')
            fname = out / f'{table_index_prefix}{key}.npz'
            base_z = cdf_to_z(base_cdf[h], self.budget)
            ref_z = cdf_to_z(ref_cdf[h], self.budget)
            abs_z_ref = np.abs(ref_z - base_z)
            abs_cdf_ref = np.abs(ref_cdf[h] - base_cdf[h])
            max_abs_z_ref = float(np.nanmax(abs_z_ref))
            max_abs_cdf_ref = float(np.nanmax(abs_cdf_ref))
            argmax_z = int(np.nanargmax(abs_z_ref))
            if self.budget.storage == 'int16':
                stored, storage_meta = quantize_z(ref_z, self.budget.z_clip)
                lookup_z = dequantize_z(stored, float(storage_meta['quant_scale']))
            elif self.budget.storage == 'float32':
                stored = ref_z.astype(np.float32)
                storage_meta = {'storage': 'float32', 'max_abs_z_storage_error_bound': float(np.finfo(np.float32).eps * max(1.0, self.budget.z_clip))}
                lookup_z = np.asarray(stored, dtype=np.float64)
            else:
                raise ValueError('storage must be int16 or float32')
            max_abs_z_storage = float(np.nanmax(np.abs(lookup_z - ref_z)))
            total_z = float(max_abs_z_ref + max_abs_z_storage)
            total_cdf = float(max_abs_cdf_ref + float(ref_meta['analytic_alias_cdf_bound']) + float(base_meta['analytic_alias_cdf_bound']))
            monotone = bool(np.all(np.diff(ref_cdf[h]) >= -1e-12))
            finite_z = bool(np.all(np.isfinite(ref_z)))
            certified = bool(monotone and finite_z and total_z <= float(self.budget.target_max_abs_z_error) and total_cdf <= float(self.budget.target_max_abs_cdf_error))
            all_cert = all_cert and certified
            meta = {
                'format': 'tailbin_dense_z_cache_v1_0',
                'theta_bundle_id': self.spec.stable_id(),
                'table_key': key,
                'certified': certified,
                'status': 'certified' if certified else 'generated_not_certified',
                'spec': {'point': point.to_dict(), 'alpha': float(self.spec.alpha), 'alpha_index': int(self.spec.alpha_index)},
                'Kmax': int(self.build.Kmax),
                'build_config': asdict(self.build),
                'error_budget': asdict(self.budget),
                'base_meta': base_meta,
                'refined_meta': ref_meta,
                'max_abs_z_refinement': max_abs_z_ref,
                'max_abs_cdf_refinement': max_abs_cdf_ref,
                'max_abs_z_storage': max_abs_z_storage,
                'total_z_error_indicator': total_z,
                'total_cdf_error_indicator': total_cdf,
                'argmax_z_refinement_k': argmax_z,
                'monotone_cdf': monotone,
                'finite_z': finite_z,
                'prefix_mass_at_Kmax': float(ref_cdf[h, -1]),
                'storage_meta': storage_meta,
                'guarantee_notes': [
                    'Theta-bundled build reuses the same finite-depth Q(z) integration across theta_f values.',
                    'Analytic Cauchy alias bounds apply per theta table for the finite PGF target.',
                    'Refinement disagreement is a numerical error indicator, not interval arithmetic.',
                ],
            }
            save = np.savez_compressed if bool(self.build.compressed_npz) else np.savez
            save(fname, z_values=stored, cdf_values=ref_cdf[h].astype(np.float32), metadata=json.dumps(meta, sort_keys=True))
            bundle_meta['tables'].append({
                'theta_f': theta,
                'path': str(fname),
                'certified': certified,
                'total_z_error_indicator': total_z,
                'total_cdf_error_indicator': total_cdf,
                'seconds_amortized_note': 'see bundle seconds; Q integration shared across theta values',
            })
        bundle_meta.update({'certified': bool(all_cert), 'status': 'certified' if all_cert else 'generated_not_certified', 'seconds': float(time.perf_counter()-t0)})
        (out / f'{table_index_prefix}{self.spec.key_prefix()}_bundle.json').write_text(json.dumps(bundle_meta, indent=2, sort_keys=True))
        return bundle_meta
