from __future__ import annotations

"""Adaptive theta-bundled cache builder with moment-certified saturation.

Theta bundling made the expensive finite-depth complex PGF integration reusable across
founder loads.  adaptive saturation adds an even larger lever: do not build a dense Cauchy/FFT
coefficient table out to the global Kmax when moment bounds already certify
that the clipped Gaussianized CDF is saturated.

For nonnegative integer Y and K>=1,
    P(Y >= K) <= E[Y] / K                                      (Markov)
and for K>=2,
    P(Y >= K) <= E[(Y)_2] / (K (K-1))                           (factorial Markov)
where E[(Y)_2] = Var(Y) + E[Y]^2 - E[Y].

If either bound is <= clip_eps, then for all k >= K-1 the clipped z(k) is
exactly +z_clip.  Thus only coefficients 0..K-2 need to be extracted.

For high-mean left-tail cases, Cantelli gives
    P(Y <= x) <= Var(Y) / (Var(Y) + (E[Y]-x)^2),  x < E[Y].
If this bound is <= clip_eps at x=Kmax+1/2, the entire table is clipped to
-z_clip and no Fourier table is needed.

These certificates are rigorous for the finite-depth target, assuming the
moment calculation is accurate at the selected finite grid/ODE resolution.  The
moments are computed by the same finite-depth CGF engine and logged in metadata.
"""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json, math, time

import numpy as np

from readsampled_cdf.fast_frequency import FastFrequencyCGFEngine, FastFrequencyConfig
from .builder import BuildConfig, ErrorBudget, cdf_to_z, dequantize_z, next_pow2, quantize_z
from .grid import ParameterPoint
from .model import make_model_params
from .theta_bundle import ThetaBundleSpec, ThetaBundleCauchyGenerator, generate_theta_bundle_embedded_refinement


def _tail_bound_right(mu: float, fac2: float, K: int) -> float:
    if K <= 0:
        return 1.0
    vals = []
    if np.isfinite(mu) and mu >= 0.0:
        vals.append(float(mu) / float(K))
    if K >= 2 and np.isfinite(fac2) and fac2 >= 0.0:
        vals.append(float(fac2) / float(K * (K - 1)))
    if not vals:
        return math.inf
    return max(0.0, min(vals))


def certified_right_saturation_start(mu: float, fac2: float, Kmax: int, eps: float) -> Optional[Tuple[int, float, str]]:
    """Return (start_k, bound, method), where z(k>=start_k)=+clip is certified."""
    if not (np.isfinite(mu) and mu >= 0.0):
        return None
    # Try closed-form candidates from Markov/factorial Markov.  Very large
    # finite means may overflow when divided by eps; those cases cannot certify
    # right saturation inside Kmax anyway, so we skip the candidate safely.
    best: Optional[Tuple[int, float, str]] = None
    K_markov = Kmax + 2
    if eps > 0.0 and np.isfinite(mu):
        ratio = mu / eps
        if np.isfinite(ratio) and ratio <= float(Kmax + 2):
            K_markov = int(math.floor(ratio)) + 1
    for K, method in [(K_markov, "markov")]:
        if 1 <= K <= Kmax + 1:
            b = _tail_bound_right(mu, fac2, K)
            if b <= eps:
                best = (K - 1, b, method)
    # Factorial candidate: solve K(K-1) > fac2/eps.
    if np.isfinite(fac2) and fac2 >= 0.0 and eps > 0.0:
        ratio2 = fac2 / eps
        if not np.isfinite(ratio2) or ratio2 > float((Kmax + 2) * (Kmax + 2)):
            K_fac = Kmax + 2
        else:
            K_fac = int(math.ceil((1.0 + math.sqrt(1.0 + 4.0 * ratio2)) / 2.0))
        for K in [max(2, K_fac - 2), max(2, K_fac - 1), max(2, K_fac), max(2, K_fac + 1), max(2, K_fac + 2)]:
            if 1 <= K <= Kmax + 1:
                b = _tail_bound_right(mu, fac2, K)
                if b <= eps:
                    cand = (K - 1, b, "second_factorial_moment" if K >= 2 else "markov")
                    if best is None or cand[0] < best[0]:
                        best = cand
    return best


def certified_full_left(mu: float, var: float, Kmax: int, eps: float) -> Optional[Tuple[float, str]]:
    """Return (bound, method) if P(Y<=Kmax) <= eps is certified."""
    x = float(Kmax) + 0.5
    if not (np.isfinite(mu) and np.isfinite(var) and var >= 0.0 and mu > x):
        return None
    d = mu - x
    b = float(var) / max(float(var) + d * d, np.finfo(float).tiny)
    if b <= eps:
        return b, "cantelli_left_tail"
    return None


def _make_fast_config(build: BuildConfig) -> FastFrequencyConfig:
    return FastFrequencyConfig(
        n_bins=int(build.n_bins),
        steps_per_time=float(build.steps_per_time),
        min_steps=int(build.min_steps),
        max_steps=int(build.max_steps),
    )




def _chernoff_tail_certificates(engine: FastFrequencyCGFEngine, theta_values: Sequence[float], build: BuildConfig, budget: ErrorBudget) -> Tuple[List[Optional[Tuple[int, float, str]]], List[Optional[Tuple[float, str]]]]:
    """Two-sided real-CGF Chernoff certificates.

    Right tail: for s>0, P(Y >= K) <= exp(K(s)-sK).  If this is <= clip_eps,
    then all k >= K-1 are +z_clip.

    Left tail: for s<0, P(Y <= Kmax) <= exp(K(s)-s(Kmax+1)).  If this is
    <= clip_eps, then all requested k=0..Kmax are -z_clip.

    These are conservative finite-depth CGF bounds at the configured grid/ODE
    resolution; they are not saddlepoint approximations.  The fixed s-grid is
    intentionally small to keep preflight cheap and robust.
    """
    eps = float(budget.clip_eps)
    logeps = math.log(max(eps, np.finfo(float).tiny))
    Kmax = int(build.Kmax)
    pos_grid = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.75)
    neg_grid = (-0.001, -0.003, -0.01, -0.03, -0.1, -0.3, -0.75)
    theta_list = [float(t) for t in theta_values]
    right: List[Optional[Tuple[int, float, str]]] = [None for _ in theta_list]
    left: List[Optional[Tuple[float, str]]] = [None for _ in theta_list]

    for ss in pos_grid:
        s = float(ss)
        for j, theta in enumerate(theta_list):
            try:
                K0, _mu, _var, _meta = engine.K012_theta(s, theta)
            except Exception:
                continue
            if not np.isfinite(K0):
                continue
            # Need K0 - s*K <= logeps.  Saturation starts at k=K-1.
            K = int(math.floor((float(K0) - logeps) / s) + 1)
            if 1 <= K <= Kmax + 1:
                log_bound = float(K0) - s * float(K)
                if log_bound <= logeps:
                    cand = (K - 1, float(math.exp(max(log_bound, -745.0))), "chernoff_right_real_cgf")
                    if right[j] is None or cand[0] < right[j][0]:
                        right[j] = cand

    # Kmax+1 rather than Kmax is a harmless integer continuity guard.
    x = float(Kmax + 1)
    for ss in neg_grid:
        s = float(ss)
        for j, theta in enumerate(theta_list):
            try:
                K0, _mu, _var, _meta = engine.K012_theta(s, theta)
            except Exception:
                continue
            if not np.isfinite(K0):
                continue
            log_bound = float(K0) - s * x
            if log_bound <= logeps:
                cand = (float(math.exp(max(log_bound, -745.0))), "chernoff_left_real_cgf")
                if left[j] is None or cand[0] < left[j][0]:
                    left[j] = cand
    return right, left

def moment_preflight(spec: ThetaBundleSpec, build: BuildConfig, budget: ErrorBudget) -> List[Dict[str, Any]]:
    """Compute moment-based saturation decisions for every theta in a bundle."""
    engine = FastFrequencyCGFEngine(
        make_model_params(spec.integration_point(), spec.alpha),
        config=_make_fast_config(build),
    )
    out: List[Dict[str, Any]] = []
    eps = float(budget.clip_eps)
    theta_list = [float(x) for x in spec.theta_values]
    raw_rows: List[Dict[str, Any]] = []
    needs_chernoff = False
    threshold = int(getattr(build, "chernoff_prefix_threshold", 2048))
    for theta in theta_list:
        K0, mu, var, meta = engine.K012_theta(0.0, theta)
        fac2 = float(var + mu * mu - mu) if np.isfinite(mu) and np.isfinite(var) else math.nan
        if np.isfinite(fac2) and fac2 < 0.0 and fac2 > -1e-8 * max(1.0, abs(mu), abs(var)):
            fac2 = 0.0
        left = certified_full_left(mu, var, int(build.Kmax), eps)
        right = certified_right_saturation_start(mu, fac2, int(build.Kmax), eps)
        if left is not None:
            kind = "constant_left"
            prefix_kmax = -1
            saturation_start_k = 0
            bound = float(left[0])
            cert_method = str(left[1])
        elif right is not None:
            start_k, bound, cert_method = right
            if start_k <= 0:
                kind = "constant_right"
                prefix_kmax = -1
                saturation_start_k = 0
            else:
                kind = "right_saturated_prefix"
                prefix_kmax = min(int(build.Kmax), int(start_k) - 1)
                saturation_start_k = int(start_k)
        else:
            kind = "full"
            prefix_kmax = int(build.Kmax)
            saturation_start_k = None
            bound = None
            cert_method = None
        row = {
            "theta_f": theta,
            "mean": float(mu) if np.isfinite(mu) else math.nan,
            "variance": float(var) if np.isfinite(var) else math.nan,
            "second_factorial_moment": float(fac2) if np.isfinite(fac2) else None,
            "moment_meta": meta,
            "saturation_kind": kind,
            "prefix_kmax": int(prefix_kmax),
            "saturation_start_k": saturation_start_k,
            "tail_bound": bound,
            "tail_certificate_method": cert_method,
        }
        raw_rows.append(row)
        if bool(getattr(build, "chernoff_tail_enabled", True)) and (kind == "full" or int(prefix_kmax) >= threshold):
            needs_chernoff = True
    if needs_chernoff:
        cright, cleft = _chernoff_tail_certificates(engine, theta_list, build, budget)
        for row, rcert, lcert in zip(raw_rows, cright, cleft):
            old_kind = str(row["saturation_kind"])
            old_prefix = int(row["prefix_kmax"])
            if lcert is not None:
                row.update({
                    "saturation_kind": "constant_left",
                    "prefix_kmax": -1,
                    "saturation_start_k": 0,
                    "tail_bound": float(lcert[0]),
                    "tail_certificate_method": str(lcert[1]),
                    "chernoff_repaired_from": old_kind,
                })
            elif rcert is not None:
                start_k, bound, method = rcert
                cand_kind = "constant_right" if int(start_k) <= 0 else "right_saturated_prefix"
                cand_prefix = -1 if int(start_k) <= 0 else min(int(build.Kmax), int(start_k) - 1)
                if old_kind == "full" or (old_prefix >= 0 and cand_prefix >= 0 and cand_prefix < old_prefix) or (old_prefix >= 0 and cand_prefix < 0):
                    row.update({
                        "saturation_kind": cand_kind,
                        "prefix_kmax": int(cand_prefix),
                        "saturation_start_k": 0 if cand_kind == "constant_right" else int(start_k),
                        "tail_bound": float(bound),
                        "tail_certificate_method": str(method),
                        "chernoff_repaired_from": old_kind,
                    })
    out.extend(raw_rows)
    return out


def _save_constant_table(path: Path, *, spec: ThetaBundleSpec, point: ParameterPoint, build: BuildConfig, budget: ErrorBudget, z_value: float, cdf_value: float, pre: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    meta = {
        "format": "tailbin_dense_z_cache_v1_0",
        "representation": "constant",
        "certified": True,
        "status": "certified_moment_saturated_constant",
        "spec": {"point": point.to_dict(), "alpha": float(spec.alpha), "alpha_index": int(spec.alpha_index)},
        "Kmax": int(build.Kmax),
        "prefix_kmax": -1,
        "seconds": float(seconds),
        "build_config": asdict(build),
        "error_budget": asdict(budget),
        "preflight": pre,
        "storage_meta": {"storage": "constant", "constant_z": float(z_value), "constant_cdf": float(cdf_value)},
        "total_z_error_indicator": 0.0,
        "total_cdf_error_indicator": float(pre.get("tail_bound") or 0.0),
        "guarantee_notes": [
            "This compact table is saturated by a one-sided moment certificate; no Fourier coefficients were needed.",
            "For all k in 0..Kmax, clipped z(k) is the stored constant under the configured clip_eps.",
        ],
    }
    np.savez_compressed(path, z_values=np.array([z_value], dtype=np.float32), metadata=json.dumps(meta, sort_keys=True))
    return meta


def _save_prefix_table(path: Path, *, spec: ThetaBundleSpec, point: ParameterPoint, build: BuildConfig, budget: ErrorBudget, prefix_cdf: np.ndarray, base_prefix_cdf: np.ndarray, base_meta: Dict[str, Any], ref_meta: Dict[str, Any], pre: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    ref_z = cdf_to_z(prefix_cdf, budget)
    base_z = cdf_to_z(base_prefix_cdf, budget)
    max_abs_z_ref = float(np.nanmax(np.abs(ref_z - base_z))) if ref_z.size else 0.0
    max_abs_cdf_ref = float(np.nanmax(np.abs(prefix_cdf - base_prefix_cdf))) if prefix_cdf.size else 0.0
    if budget.storage == "int16":
        stored, storage_meta = quantize_z(ref_z, budget.z_clip)
        lookup_z = dequantize_z(stored, float(storage_meta["quant_scale"]))
    elif budget.storage == "float32":
        stored = ref_z.astype(np.float32)
        storage_meta = {"storage": "float32", "max_abs_z_storage_error_bound": float(np.finfo(np.float32).eps * max(1.0, budget.z_clip))}
        lookup_z = stored.astype(np.float64)
    else:
        raise ValueError("storage must be int16 or float32")
    max_abs_z_storage = float(np.nanmax(np.abs(lookup_z - ref_z))) if ref_z.size else 0.0
    total_z = float(max_abs_z_ref + max_abs_z_storage)
    total_cdf = float(max_abs_cdf_ref + float(ref_meta.get("analytic_alias_cdf_bound", 0.0)) + float(base_meta.get("analytic_alias_cdf_bound", 0.0)) + float(pre.get("tail_bound") or 0.0))
    monotone = bool(prefix_cdf.size == 0 or np.all(np.diff(prefix_cdf) >= -1e-12))
    finite_z = bool(np.all(np.isfinite(ref_z)))
    certified = bool(monotone and finite_z and total_z <= float(budget.target_max_abs_z_error) and total_cdf <= max(float(budget.target_max_abs_cdf_error), float(budget.clip_eps) * 2.0))
    representation = "right_saturated_prefix" if pre["saturation_kind"] == "right_saturated_prefix" else "full"
    meta = {
        "format": "tailbin_dense_z_cache_v1_0",
        "representation": representation,
        "certified": certified,
        "status": "certified" if certified else "generated_not_certified",
        "spec": {"point": point.to_dict(), "alpha": float(spec.alpha), "alpha_index": int(spec.alpha_index)},
        "Kmax": int(build.Kmax),
        "prefix_kmax": int(pre["prefix_kmax"]),
        "saturation_start_k": pre.get("saturation_start_k"),
        "seconds": float(seconds),
        "build_config": asdict(build),
        "error_budget": asdict(budget),
        "preflight": pre,
        "base_meta": base_meta,
        "refined_meta": ref_meta,
        "storage_meta": storage_meta,
        "max_abs_z_refinement": max_abs_z_ref,
        "max_abs_cdf_refinement": max_abs_cdf_ref,
        "max_abs_z_storage": max_abs_z_storage,
        "total_z_error_indicator": total_z,
        "total_cdf_error_indicator": total_cdf,
        "monotone_cdf": monotone,
        "finite_z": finite_z,
        "guarantee_notes": [
            "Prefix coefficients are generated by damped Cauchy/FFT extraction.",
            "The right-saturated suffix is certified by a moment tail bound, so clipped z(k) is +z_clip for k>=saturation_start_k.",
            "Refinement disagreement audits finite-grid/ODE/FFT discretization; it is not interval arithmetic.",
        ],
    }
    arrays = {"z_prefix": stored, "metadata": json.dumps(meta, sort_keys=True)}
    if prefix_cdf.size:
        arrays["cdf_prefix"] = prefix_cdf.astype(np.float32)
    np.savez_compressed(path, **arrays)
    return meta


class AdaptiveThetaBundleBuilder:
    """Build theta-bundled tables using moment-certified saturation first."""

    def __init__(self, spec: ThetaBundleSpec, build: BuildConfig, budget: ErrorBudget):
        self.spec = spec
        self.build = build
        self.budget = budget

    def build_bundle_with_preflight(self, output_dir: str | Path, preflight: List[Dict[str, Any]], *, table_index_prefix: str = "") -> Dict[str, Any]:
        """Build a bundle from an externally supplied preflight plan.

        The alpha-monotone builder uses this to pass exact alpha-monotone right-tail prefix
        certificates from a lower cutoff to higher cutoffs.  This avoids
        recomputing moment preflight and, more importantly, prevents a higher
        cutoff from paying for a longer prefix than one already certified by
        stochastic dominance.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        tables: List[Dict[str, Any]] = []
        # Save fully saturated constant tables immediately.
        remaining: List[Dict[str, Any]] = []
        for pre in preflight:
            theta = float(pre["theta_f"])
            point = replace(self.spec.integration_point(), theta_f=theta)
            key = f"{point.key()}_a{self.spec.alpha_index:02d}_alpha{self.spec.alpha:.8g}".replace('.', 'p')
            fname = out / f"{table_index_prefix}{key}.npz"
            if pre["saturation_kind"] == "constant_right":
                meta = _save_constant_table(fname, spec=self.spec, point=point, build=self.build, budget=self.budget, z_value=float(self.budget.z_clip), cdf_value=1.0 - float(self.budget.clip_eps), pre=pre, seconds=0.0)
                tables.append({"theta_f": theta, "path": str(fname), "certified": True, "representation": "constant_right", "prefix_kmax": -1})
            elif pre["saturation_kind"] == "constant_left":
                meta = _save_constant_table(fname, spec=self.spec, point=point, build=self.build, budget=self.budget, z_value=-float(self.budget.z_clip), cdf_value=float(self.budget.clip_eps), pre=pre, seconds=0.0)
                tables.append({"theta_f": theta, "path": str(fname), "certified": True, "representation": "constant_left", "prefix_kmax": -1})
            else:
                remaining.append(pre)
        # adaptive bucketing: bucket remaining theta values by the actual FFT node
        # counts required for their certified prefix.  adaptive put every remaining
        # theta value in one group at the largest prefix; if one founder load
        # needed a huge prefix, all smaller-prefix founder loads paid that huge
        # base/refined FFT cost.  Here, theta values sharing the same base/refined
        # node counts are grouped together, while smaller prefixes stay cheap.
        # This is exact with respect to the same Cauchy/FFT target: each table is
        # still generated to at least its certified prefix length, and suffixes are
        # still certified by the moment bound.
        bucket_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        one_max_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        grouping_strategy = "none"
        if remaining:
            k_eff_max = max(max(0, int(pre["prefix_kmax"])) for pre in remaining)
            one_base = next_pow2(int(math.ceil(float(self.build.base_node_factor) * (k_eff_max + 1))))
            one_ref = next_pow2(int(math.ceil(float(self.build.refine_node_factor) * (k_eff_max + 1))))
            one_max_groups[(one_base, one_ref)] = list(remaining)
            for pre in remaining:
                k_eff_i = max(0, int(pre["prefix_kmax"]))
                base_nodes_i = next_pow2(int(math.ceil(float(self.build.base_node_factor) * (k_eff_i + 1))))
                ref_nodes_i = next_pow2(int(math.ceil(float(self.build.refine_node_factor) * (k_eff_i + 1))))
                bucket_groups.setdefault((base_nodes_i, ref_nodes_i), []).append(pre)
            one_work = one_base + one_ref
            bucket_work = sum(k[0] + k[1] for k in bucket_groups)
            if bucket_work < one_work:
                groups = bucket_groups
                grouping_strategy = "prefix_node_bucketed"
            else:
                groups = one_max_groups
                grouping_strategy = "one_max_prefix"
        else:
            groups = {}
        group_metas: List[Dict[str, Any]] = []
        for (base_nodes, ref_nodes), group in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            k_eff = max(int(pre["prefix_kmax"]) for pre in group)
            theta_values = [float(p["theta_f"]) for p in group]
            cap = getattr(self.build, "max_refined_nodes", None)
            if cap is not None and int(ref_nodes) > int(cap):
                # Strict production safety: do not launch an unbounded hard FFT group.
                # The caller can rerun these deferred groups with a higher cap or a
                # longer/harder build queue.  No uncertified arrays are written.
                group_metas.append({
                    "prefix_kmax": int(k_eff),
                    "theta_values": theta_values,
                    "seconds": 0.0,
                    "base_nodes": int(base_nodes),
                    "refined_nodes": int(ref_nodes),
                    "status": "deferred_refined_node_cap_exceeded",
                    "max_refined_nodes": int(cap),
                    "n_theta": len(theta_values),
                })
                for pre in group:
                    tables.append({
                        "theta_f": float(pre["theta_f"]),
                        "path": None,
                        "certified": False,
                        "representation": "deferred_hard_table",
                        "prefix_kmax": int(pre["prefix_kmax"]),
                        "total_z_error_indicator": None,
                        "total_cdf_error_indicator": None,
                        "status": "deferred_refined_node_cap_exceeded",
                        "required_refined_nodes": int(ref_nodes),
                    })
                continue
            sub_spec = ThetaBundleSpec(base_point=self.spec.integration_point(), theta_values=theta_values, alpha=self.spec.alpha, alpha_index=self.spec.alpha_index)
            sub_build = replace(self.build, Kmax=int(k_eff))
            gt0 = time.perf_counter()
            if bool(getattr(self.build, "use_embedded_refinement", True)):
                base_cdf, ref_cdf, base_meta, ref_meta = generate_theta_bundle_embedded_refinement(sub_spec, sub_build, self.budget)
            else:
                base_cdf, base_meta = ThetaBundleCauchyGenerator(sub_spec, sub_build, self.budget, refined=False).generate()
                ref_cdf, ref_meta = ThetaBundleCauchyGenerator(sub_spec, sub_build, self.budget, refined=True).generate()
            gsec = float(time.perf_counter() - gt0)
            group_meta = {"prefix_kmax": int(k_eff), "theta_values": theta_values, "seconds": gsec, "base_meta": base_meta, "refined_meta": ref_meta, "n_theta": len(theta_values)}
            group_metas.append(group_meta)
            if base_cdf is None or ref_cdf is None:
                # partial timing probe: write json only at bundle level
                continue
            for h, pre in enumerate(group):
                theta = float(pre["theta_f"])
                point = replace(self.spec.integration_point(), theta_f=theta)
                key = f"{point.key()}_a{self.spec.alpha_index:02d}_alpha{self.spec.alpha:.8g}".replace('.', 'p')
                fname = out / f"{table_index_prefix}{key}.npz"
                kk = int(pre["prefix_kmax"]) + 1
                meta = _save_prefix_table(fname, spec=self.spec, point=point, build=self.build, budget=self.budget, prefix_cdf=ref_cdf[h, :kk], base_prefix_cdf=base_cdf[h, :kk], base_meta=base_meta, ref_meta=ref_meta, pre=pre, seconds=gsec / max(1, len(group)))
                tables.append({"theta_f": theta, "path": str(fname), "certified": bool(meta.get("certified")), "representation": meta.get("representation"), "prefix_kmax": int(pre["prefix_kmax"]), "total_z_error_indicator": meta.get("total_z_error_indicator"), "total_cdf_error_indicator": meta.get("total_cdf_error_indicator")})
        all_cert = bool(tables and all(t.get("certified") for t in tables) and len(tables) == len(preflight))
        bundle_meta = {
            "format": "tailbin_adaptive_theta_bundle_summary_v1_0",
            "bundle_id": self.spec.stable_id(),
            "bundle_key": self.spec.key_prefix(),
            "certified": all_cert,
            "status": "certified" if all_cert else "generated_not_certified_or_partial",
            "seconds": float(time.perf_counter() - t0),
            "spec": {"base_point": self.spec.integration_point().to_dict(), "theta_values": [float(x) for x in self.spec.theta_values], "alpha": float(self.spec.alpha), "alpha_index": int(self.spec.alpha_index)},
            "build_config": asdict(self.build),
            "error_budget": asdict(self.budget),
            "preflight": preflight,
            "groups": group_metas,
            "grouping_strategy": grouping_strategy,
            "tables": tables,
            "n_tables_written": len(tables),
            "n_tables_certified": int(sum(1 for t in tables if t.get("certified"))),
            "n_constant_tables": int(sum(1 for t in tables if str(t.get("representation", "")).startswith("constant"))),
            "n_prefix_tables": int(sum(1 for t in tables if t.get("representation") == "right_saturated_prefix")),
            "n_full_tables": int(sum(1 for t in tables if t.get("representation") == "full")),
            "notes": "adaptive builder chooses the cheaper of one-max-prefix grouping and prefix-node bucketing after moment preflight.",
        }
        (out / f"{table_index_prefix}{self.spec.key_prefix()}_adaptive_bundle.json").write_text(json.dumps(bundle_meta, indent=2, sort_keys=True))
        return bundle_meta


    def build_bundle(self, output_dir: str | Path, *, table_index_prefix: str = "") -> Dict[str, Any]:
        preflight = moment_preflight(self.spec, self.build, self.budget)
        return self.build_bundle_with_preflight(output_dir, preflight, table_index_prefix=table_index_prefix)
