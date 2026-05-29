from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import hashlib
import json
import math
import time

import numpy as np
from scipy.stats import norm

from readsampled_cdf.fourier_reference import ComplexPGFGridEngine, FourierCDFConfig
from .batched_backend import BatchedComplexPGFGridEngine, BatchedPGFConfig, StableSolveIVPThetaPGFGridEngine
from .gpu_backend import CuPyComplexPGFGridEngine, CuPyPGFConfig

from .grid import ParameterPoint
from .model import make_model_params


def next_pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


@dataclass(frozen=True)
class ErrorBudget:
    """Numerical certification targets in the space used by the Gaussian copula."""

    target_max_abs_z_error: float = 1.0e-2
    target_max_abs_cdf_error: float = 1.0e-5
    clip_eps: float = 1.0e-12
    z_clip: float = 7.05
    storage: str = "int16"  # int16 or float32


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for one dense CDF/z table build."""

    Kmax: int = 50_000
    base_node_factor: float = 1.0
    refine_node_factor: float = 2.0
    base_alias_eta: float = 30.0
    refine_alias_eta: float = 36.0
    n_bins: int = 24
    steps_per_time: float = 1.4
    min_steps: int = 32
    max_steps: int = 120
    ode_rtol: float = 1.0e-7
    ode_atol: float = 1.0e-9
    use_solve_ivp: bool = False
    node_budget: Optional[int] = None  # for timing probes only; certified=False if partial
    pgf_backend: str = "batched"  # batched, cupy, or scalar
    batch_size: int = 128
    compressed_npz: bool = True
    use_conjugate_symmetry: bool = True  # evaluate only half the Cauchy circle and reconstruct the rest
    use_embedded_refinement: bool = True  # derive coarse refinement check from refined nodes when possible
    max_refined_nodes: Optional[int] = None  # defer groups whose refined FFT node count exceeds this cap
    stable_pgf_fallback: bool = True  # retry non-finite complex-PGF tables with a safer backend
    stable_rk4_fallback_step_multiplier: float = 10.0  # first fallback: finer fixed-step RK4
    stable_rk4_fallback_max_steps: int = 2000
    solve_ivp_fallback_node_cap: Optional[int] = 4096  # final fallback: slow solve_ivp for small FFT grids
    chernoff_tail_enabled: bool = True  # experimental: two-sided real-CGF Chernoff tail certificates
    chernoff_prefix_threshold: int = 2048  # only run Chernoff if moment prefix is this large or full


@dataclass(frozen=True)
class TableSpec:
    point: ParameterPoint
    alpha: float
    alpha_index: int = 0

    def key(self) -> str:
        return f"{self.point.key()}_a{self.alpha_index:02d}_alpha{self.alpha:.8g}".replace(".", "p")

    def stable_id(self) -> str:
        payload = {"point": self.point.to_dict(), "alpha": float(self.alpha), "alpha_index": int(self.alpha_index)}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:20]


def cdf_to_z(cdf: np.ndarray, budget: ErrorBudget) -> np.ndarray:
    c = np.clip(np.asarray(cdf, dtype=np.float64), float(budget.clip_eps), 1.0 - float(budget.clip_eps))
    z = norm.ppf(c)
    return np.clip(z, -float(budget.z_clip), float(budget.z_clip))


def quantize_z(z: np.ndarray, z_clip: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    scale = 32767.0 / float(z_clip)
    q = np.rint(np.clip(z, -float(z_clip), float(z_clip)) * scale).astype(np.int16)
    return q, {
        "storage": "int16",
        "z_clip": float(z_clip),
        "quant_scale": float(scale),
        "max_abs_z_storage_error_bound": float(0.5 / scale),
    }


def dequantize_z(q: np.ndarray, quant_scale: float) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) / float(quant_scale)


class CauchyTableGenerator:
    """Damped Cauchy/FFT coefficient extraction for one finite-depth PGF target.

    For G(z)=sum p_n z^n and nodes z_m=r exp(2 pi i m/M), FFT gives
        p_n + sum_{l>=1} p_{n+lM} r^{lM}.
    Because p_n >= 0 and sum p_n <= 1, the nonnegative alias contamination is
    bounded by r^M = exp(-eta).  This is a genuine analytic Cauchy alias bound
    for the finite PGF target.  The remaining errors are from finite-grid/ODE
    approximation and floating point; those are audited by two-resolution builds.
    """

    def __init__(self, spec: TableSpec, build: BuildConfig, budget: ErrorBudget, *, refined: bool):
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
        fcfg = FourierCDFConfig(
            n_bins=int(build.n_bins),
            steps_per_time=float(build.steps_per_time),
            min_steps=int(build.min_steps),
            max_steps=int(build.max_steps),
            clip_eps=float(budget.clip_eps),
            ode_rtol=float(build.ode_rtol),
            ode_atol=float(build.ode_atol),
            use_solve_ivp=bool(build.use_solve_ivp),
        )
        if str(build.pgf_backend) == "batched":
            self.engine = BatchedComplexPGFGridEngine(make_model_params(spec.point, spec.alpha), fcfg, batch_config=BatchedPGFConfig(batch_size=int(build.batch_size)))
        elif str(build.pgf_backend) == "cupy":
            self.engine = CuPyComplexPGFGridEngine(make_model_params(spec.point, spec.alpha), fcfg, gpu_config=CuPyPGFConfig(batch_size=int(build.batch_size)))
        elif str(build.pgf_backend) == "scalar":
            self.engine = ComplexPGFGridEngine(make_model_params(spec.point, spec.alpha), fcfg)
        else:
            raise ValueError("pgf_backend must be batched, cupy, or scalar")

    def _stable_fallback_allowed(self, node_count: int) -> bool:
        if not bool(getattr(self.build, "stable_pgf_fallback", True)):
            return False
        cap = getattr(self.build, "solve_ivp_fallback_node_cap", 4096)
        return cap is None or int(node_count) <= int(cap)

    def _pgf_values_with_fallback(self, z_nodes: np.ndarray, meta: Dict[str, Any]) -> Optional[np.ndarray]:
        if hasattr(self.engine, "pgf_many"):
            vals = self.engine.pgf_many(z_nodes)
        else:
            vals = np.empty(z_nodes.size, dtype=np.complex128)
            for m, z in enumerate(z_nodes):
                vals[m] = self.engine.pgf(z)
        if np.isfinite(vals).all():
            meta["pgf_backend_used"] = str(self.build.pgf_backend)
            return vals
        n_bad = int(vals.size - np.isfinite(vals).sum())
        meta["fast_backend_nonfinite_count"] = n_bad
        meta["fallback_reason"] = "nonfinite_complex_pgf_nodes"
        current_steps = int(getattr(self.engine, "n_steps", max(1, int(math.ceil(float(self.spec.point.T) * float(self.build.steps_per_time))))))
        rk4_cap = int(getattr(self.build, "stable_rk4_fallback_max_steps", 2000))
        rk4_steps = min(rk4_cap, max(current_steps + 1, int(math.ceil(current_steps * float(getattr(self.build, "stable_rk4_fallback_step_multiplier", 10.0))))))
        if str(self.build.pgf_backend) == "batched" and rk4_steps > current_steps:
            rk4_build = replace(self.build, min_steps=int(rk4_steps), max_steps=int(rk4_steps), steps_per_time=max(float(self.build.steps_per_time), float(rk4_steps) / max(float(self.spec.point.T), np.finfo(float).tiny)))
            rk4_cfg = FourierCDFConfig(
                n_bins=int(rk4_build.n_bins), steps_per_time=float(rk4_build.steps_per_time),
                min_steps=int(rk4_build.min_steps), max_steps=int(rk4_build.max_steps),
                clip_eps=float(self.budget.clip_eps), ode_rtol=float(rk4_build.ode_rtol),
                ode_atol=float(rk4_build.ode_atol), use_solve_ivp=False,
            )
            rk4 = BatchedComplexPGFGridEngine(make_model_params(self.spec.point, self.spec.alpha), rk4_cfg, batch_config=BatchedPGFConfig(batch_size=int(self.build.batch_size)))
            vals2 = rk4.pgf_many(z_nodes)
            meta["rk4_fallback_steps"] = int(rk4.n_steps)
            meta["rk4_fallback_nonfinite_count"] = int(vals2.size - np.isfinite(vals2).sum())
            if np.isfinite(vals2).all():
                meta["pgf_backend_used"] = "batched_rk4_fallback"
                meta["fallback_status"] = "rk4_fallback_succeeded"
                return vals2
        if not self._stable_fallback_allowed(self.node_count):
            meta["pgf_backend_used"] = str(self.build.pgf_backend)
            meta["fallback_status"] = "not_attempted_cap_or_disabled"
            return None
        stable_cfg = FourierCDFConfig(
            n_bins=int(self.build.n_bins),
            steps_per_time=float(self.build.steps_per_time),
            min_steps=int(self.build.min_steps),
            max_steps=int(self.build.max_steps),
            clip_eps=float(self.budget.clip_eps),
            ode_rtol=float(self.build.ode_rtol),
            ode_atol=float(self.build.ode_atol),
            use_solve_ivp=True,
        )
        stable = StableSolveIVPThetaPGFGridEngine(make_model_params(self.spec.point, self.spec.alpha), stable_cfg)
        vals = stable.pgf_many(z_nodes)
        meta["pgf_backend_used"] = "solve_ivp_fallback"
        meta["fallback_status"] = "solve_ivp_attempted"
        meta["fallback_nonfinite_count"] = int(vals.size - np.isfinite(vals).sum())
        if not np.isfinite(vals).all():
            return None
        return vals

    def generate(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        t0 = time.perf_counter()
        M = int(self.node_count)
        use_sym = bool(getattr(self.build, "use_conjugate_symmetry", True)) and (M % 2 == 0)
        if use_sym:
            # Real PMF coefficients imply G(conj(z)) = conj(G(z)).  On the
            # Cauchy circle z_m = r exp(2 pi i m/M), nodes m and M-m are
            # conjugate pairs.  We therefore evaluate only m=0..M/2 and fill
            # the remaining half exactly by conjugation.  This is a mathematical
            # identity for the same finite PGF target, not an approximation.
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
            "method": "damped_cauchy_fft_coefficients",
            "pgf_backend": str(self.build.pgf_backend),
            "batch_size": int(self.build.batch_size),
            "refined": bool(self.refined),
            "Kmax": self.Kmax,
            "node_count": M,
            "nodes_evaluated": int(n_eval),
            "unique_nodes_required": int(n_unique_full),
            "full_circle_node_count": int(M),
            "used_conjugate_symmetry": bool(use_sym),
            "partial_build": bool(partial),
            "radius": float(self.radius),
            "alias_eta": float(self.alias_eta),
            "analytic_alias_cdf_bound": float(math.exp(-float(self.alias_eta))),
            "seconds": elapsed,
            "mean_seconds_per_node": mean_node,
            "estimated_full_seconds": float(mean_node * n_unique_full) if np.isfinite(mean_node) else math.nan,
        }
        meta.update(meta_probe)
        if vals_unique is None:
            meta["status"] = "nonfinite_complex_pgf_nodes"
            return None, meta
        if partial:
            return None, meta
        if use_sym:
            vals = np.empty(M, dtype=np.complex128)
            vals[: n_unique_full] = vals_unique
            # m=1..M/2-1 have conjugate partners at M-m.  m=0 and m=M/2
            # are self-conjugate locations on the real axis and are already set.
            for m in range(1, M // 2):
                vals[M - m] = np.conj(vals_unique[m])
        else:
            vals = vals_unique
        fft_vals = np.fft.fft(vals) / float(M)
        n = np.arange(self.Kmax + 1, dtype=np.float64)
        coeff = np.real(fft_vals[: self.Kmax + 1] * (self.radius ** (-n)))
        if not np.isfinite(coeff).all():
            meta["status"] = "nonfinite_cauchy_coefficients"
            meta["nonfinite_coefficient_count"] = int(coeff.size - np.isfinite(coeff).sum())
            return None, meta
        negative_mass = float(np.sum(np.minimum(coeff, 0.0)))
        coeff = np.maximum(coeff, 0.0)
        cdf = np.cumsum(coeff)
        cdf = np.clip(cdf, 0.0, 1.0)
        meta.update({
            "negative_mass_clipped": negative_mass,
            "prefix_mass_at_Kmax": float(cdf[-1]) if cdf.size else 0.0,
            "monotone_cdf": bool(np.all(np.diff(cdf) >= -1e-12)),
        })
        return cdf, meta


class DenseCacheBuilder:
    """Build one certified dense z-cache table for one parameter-alpha point."""

    def __init__(self, spec: TableSpec, build: BuildConfig, budget: ErrorBudget):
        self.spec = spec
        self.build = build
        self.budget = budget

    def build_table(self, output_path: str | Path) -> Dict[str, Any]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        base_cdf, base_meta = CauchyTableGenerator(self.spec, self.build, self.budget, refined=False).generate()
        ref_cdf, ref_meta = CauchyTableGenerator(self.spec, self.build, self.budget, refined=True).generate()
        if base_cdf is None or ref_cdf is None:
            meta = self._metadata(base_meta, ref_meta, certified=False, status="partial_timing_probe", seconds=time.perf_counter()-t0)
            with output_path.with_suffix(".json").open("w") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
            return meta
        base_z = cdf_to_z(base_cdf, self.budget)
        ref_z = cdf_to_z(ref_cdf, self.budget)
        abs_z_ref = np.abs(ref_z - base_z)
        abs_cdf_ref = np.abs(ref_cdf - base_cdf)
        if (not np.isfinite(abs_z_ref).any()) or (not np.isfinite(abs_cdf_ref).any()):
            meta = self._metadata(
                base_meta, ref_meta, certified=False, status="nonfinite_table_generated",
                seconds=time.perf_counter() - t0,
                extra={"reason": "base/refined table comparison produced no finite values"},
            )
            with output_path.with_suffix(".json").open("w") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
            return meta
        max_abs_z_ref = float(np.nanmax(abs_z_ref))
        max_abs_cdf_ref = float(np.nanmax(abs_cdf_ref))
        argmax_z = int(np.nanargmax(abs_z_ref))

        if self.budget.storage == "int16":
            stored, storage_meta = quantize_z(ref_z, self.budget.z_clip)
            lookup_z = dequantize_z(stored, float(storage_meta["quant_scale"]))
        elif self.budget.storage == "float32":
            stored = ref_z.astype(np.float32)
            storage_meta = {
                "storage": "float32",
                "max_abs_z_storage_error_bound": float(np.finfo(np.float32).eps * max(1.0, self.budget.z_clip)),
            }
            lookup_z = np.asarray(stored, dtype=np.float64)
        else:
            raise ValueError("storage must be int16 or float32")
        max_abs_z_storage = float(np.nanmax(np.abs(lookup_z - ref_z)))
        total_z_error_indicator = float(max_abs_z_ref + max_abs_z_storage)
        total_cdf_error_indicator = float(max_abs_cdf_ref + float(ref_meta["analytic_alias_cdf_bound"]) + float(base_meta["analytic_alias_cdf_bound"]))
        monotone = bool(np.all(np.diff(ref_cdf) >= -1e-12))
        finite_z = bool(np.all(np.isfinite(ref_z)))
        certified = bool(
            monotone
            and finite_z
            and total_z_error_indicator <= float(self.budget.target_max_abs_z_error)
            and total_cdf_error_indicator <= float(self.budget.target_max_abs_cdf_error)
        )
        meta = self._metadata(
            base_meta, ref_meta, certified=certified,
            status="certified" if certified else "generated_not_certified",
            seconds=time.perf_counter() - t0,
            extra={
                "max_abs_z_refinement": max_abs_z_ref,
                "max_abs_cdf_refinement": max_abs_cdf_ref,
                "max_abs_z_storage": max_abs_z_storage,
                "total_z_error_indicator": total_z_error_indicator,
                "total_cdf_error_indicator": total_cdf_error_indicator,
                "argmax_z_refinement_k": argmax_z,
                "monotone_cdf": monotone,
                "finite_z": finite_z,
                "prefix_mass_at_Kmax": float(ref_cdf[-1]),
                "storage_meta": storage_meta,
            },
        )
        save = np.savez_compressed if bool(self.build.compressed_npz) else np.savez
        save(
            output_path,
            z_values=stored,
            cdf_values=ref_cdf.astype(np.float32),
            metadata=json.dumps(meta, sort_keys=True),
        )
        return meta

    def _metadata(self, base_meta: Dict[str, Any], ref_meta: Dict[str, Any], *, certified: bool, status: str, seconds: float, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d = {
            "format": "tailbin_dense_z_cache_v1_0",
            "table_id": self.spec.stable_id(),
            "table_key": self.spec.key(),
            "certified": bool(certified),
            "status": str(status),
            "seconds": float(seconds),
            "spec": {
                "point": self.spec.point.to_dict(),
                "alpha": float(self.spec.alpha),
                "alpha_index": int(self.spec.alpha_index),
            },
            "Kmax": int(self.build.Kmax),
            "build_config": asdict(self.build),
            "error_budget": asdict(self.budget),
            "base_meta": base_meta,
            "refined_meta": ref_meta,
            "guarantee_notes": [
                "Analytic Cauchy alias bounds are rigorous for the finite PGF target when coefficients are nonnegative and sum to <=1.",
                "Storage quantization bound is deterministic for stored z values.",
                "Refinement disagreement is an auditable numerical error indicator for finite-grid/ODE/FFT discretization, not a formal interval-arithmetic proof.",
                "For theorem-level floating-point guarantees, replace the backend with interval arithmetic or a verified ODE solver.",
            ],
        }
        if extra:
            d.update(extra)
        return d


class DenseZCache:
    """Read one dense or compact z-cache table.

    Earlier dense tables store a dense `z_values` array.  compact tables may store compact
    moment-saturated tables:

    * representation=constant: one z value applies to every k in 0..Kmax.
    * representation=right_saturated_prefix: z_prefix stores 0..prefix_kmax;
      all k>=saturation_start_k return +z_clip.
    * representation=full: a full dense or prefix array with prefix_kmax=Kmax.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = np.load(self.path, allow_pickle=False)
        self.metadata = json.loads(str(data["metadata"]))
        self.representation = str(self.metadata.get("representation", "dense"))
        self._cdf = data["cdf_values"] if "cdf_values" in data.files else (data["cdf_prefix"] if "cdf_prefix" in data.files else None)
        sm = self.metadata.get("storage_meta", {})
        if "z_prefix" in data.files:
            raw = data["z_prefix"]
        else:
            raw = data["z_values"]
        if self.representation == "constant":
            self._z = np.asarray(raw, dtype=np.float32)
        elif sm.get("storage") == "int16":
            self._z = dequantize_z(raw, float(sm["quant_scale"])).astype(np.float32)
        else:
            self._z = np.asarray(raw, dtype=np.float32)

    @property
    def Kmax(self) -> int:
        return int(self.metadata["Kmax"])

    @property
    def certified(self) -> bool:
        return bool(self.metadata.get("certified", False))

    @property
    def prefix_kmax(self) -> int:
        if "prefix_kmax" in self.metadata:
            return int(self.metadata["prefix_kmax"])
        return self.Kmax

    def _check_k(self, k: int) -> int:
        k = int(k)
        if k < 0 or k > self.Kmax:
            raise IndexError(f"k={k} outside 0..{self.Kmax}")
        return k

    def z(self, k: int) -> float:
        k = self._check_k(k)
        if self.representation == "constant":
            return float(self._z[0])
        if self.representation == "right_saturated_prefix" and k > self.prefix_kmax:
            return float(self.metadata.get("error_budget", {}).get("z_clip", 7.05))
        return float(self._z[k])

    def cdf(self, k: int) -> float:
        k = self._check_k(k)
        if self.representation == "constant":
            sm = self.metadata.get("storage_meta", {})
            if "constant_cdf" in sm:
                return float(sm["constant_cdf"])
            return float(norm.cdf(self.z(k)))
        if self.representation == "right_saturated_prefix" and k > self.prefix_kmax:
            return float(1.0 - float(self.metadata.get("error_budget", {}).get("clip_eps", 1e-12)))
        if self._cdf is not None:
            return float(self._cdf[k])
        return float(norm.cdf(self.z(k)))

    def z_many(self, ks: Sequence[int]) -> np.ndarray:
        idx = np.asarray(ks, dtype=int)
        if idx.size and (idx.min() < 0 or idx.max() > self.Kmax):
            raise IndexError(f"ks outside 0..{self.Kmax}")
        if self.representation == "constant":
            return np.full(idx.shape, float(self._z[0]), dtype=np.float32)
        if self.representation == "right_saturated_prefix":
            out = np.empty(idx.shape, dtype=np.float32)
            mask = idx <= self.prefix_kmax
            out[mask] = self._z[idx[mask]]
            out[~mask] = float(self.metadata.get("error_budget", {}).get("z_clip", 7.05))
            return out
        return np.asarray(self._z[idx], dtype=np.float32)
