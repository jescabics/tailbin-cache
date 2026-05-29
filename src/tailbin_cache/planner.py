from __future__ import annotations

"""Adaptive-cache planning utilities.

The planner is intentionally cheap compared with full coefficient-table builds:
it only computes the finite-depth moments needed for saturation preflight and
then summarizes how many tables would be constant, right-saturated prefixes, or
full unresolved tables.  It also estimates the benefit of adaptive prefix-node
bucketing versus the one-max-prefix-per-theta-bundle strategy.
"""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv, json, math, time

from .adaptive_bundle import moment_preflight
from .builder import BuildConfig, ErrorBudget, next_pow2
from .grid import CacheGrid, ParameterPoint
from .theta_bundle import ThetaBundleSpec


def _storage_bytes_for_representation(storage: str, prefix_kmax: int, Kmax: int, representation: str) -> int:
    if representation.startswith("constant"):
        return 2 if storage == "int16" else 4
    n = max(0, int(prefix_kmax) + 1)
    per = 2 if storage == "int16" else 4
    return int(n * per)


def _node_counts_for_prefix(build: BuildConfig, prefix_kmax: int) -> tuple[int, int]:
    k = max(0, int(prefix_kmax))
    base = next_pow2(int(math.ceil(float(build.base_node_factor) * (k + 1))))
    ref = next_pow2(int(math.ceil(float(build.refine_node_factor) * (k + 1))))
    return base, ref


def unique_base_points(grid: CacheGrid) -> List[ParameterPoint]:
    base_points: List[ParameterPoint] = []
    seen = set()
    for p in grid.parameter_points():
        key = (float(p.R), float(p.T), float(p.N), int(p.depth), float(p.u), float(p.ploidy_factor), float(p.lam), bool(p.condition_on_survival))
        if key not in seen:
            seen.add(key)
            base_points.append(ParameterPoint(R=p.R, T=p.T, theta_f=0.0, N=p.N, depth=p.depth, u=p.u, ploidy_factor=p.ploidy_factor, lam=p.lam, condition_on_survival=p.condition_on_survival))
    return base_points


def plan_adaptive_cache(grid: CacheGrid, build: BuildConfig, budget: ErrorBudget, *, limit_bundles: Optional[int] = None, output_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    bundle_rows: List[Dict[str, Any]] = []
    bundles = []
    idx = 0
    for bp in unique_base_points(grid):
        theta_values = grid.valid_theta_values_for_T(bp.T)
        if not theta_values:
            continue
        for ai, alpha in enumerate(grid.alphas):
            bundles.append((idx, ThetaBundleSpec(base_point=bp, theta_values=theta_values, alpha=float(alpha), alpha_index=int(ai))))
            idx += 1
    n_bundles_expected = len(bundles)
    if limit_bundles is not None:
        bundles = bundles[: int(limit_bundles)]

    for bundle_idx, spec in bundles:
        b0 = time.perf_counter()
        pre = moment_preflight(spec, build, budget)
        # one-group cost proxy: one group at max nonconstant prefix.
        nonconstant = [p for p in pre if not str(p["saturation_kind"]).startswith("constant")]
        if nonconstant:
            k_v03 = max(int(p["prefix_kmax"]) for p in nonconstant)
            v03_base, v03_ref = _node_counts_for_prefix(build, k_v03)
            v03_node_work = v03_base + v03_ref
        else:
            k_v03 = -1; v03_base = v03_ref = v03_node_work = 0
        # adaptive cost proxy: choose cheaper of one max-prefix group and prefix-node buckets.
        buckets: Dict[tuple[int, int], List[Dict[str, Any]]] = {}
        for p in nonconstant:
            buckets.setdefault(_node_counts_for_prefix(build, int(p["prefix_kmax"])), []).append(p)
        bucket_node_work = int(sum(k[0] + k[1] for k in buckets))
        if v03_node_work == 0 or bucket_node_work < v03_node_work:
            v04_node_work = bucket_node_work
            grouping_strategy = "prefix_node_bucketed" if bucket_node_work else "none"
        else:
            v04_node_work = int(v03_node_work)
            grouping_strategy = "one_max_prefix"
        max_prefix = max([int(p["prefix_kmax"]) for p in nonconstant], default=-1)
        bundle_rows.append({
            "bundle_idx": int(bundle_idx),
            "R": float(spec.base_point.R), "T": float(spec.base_point.T), "N": float(spec.base_point.N), "depth": int(spec.base_point.depth),
            "alpha": float(spec.alpha), "alpha_index": int(spec.alpha_index),
            "n_theta": len(theta_values),
            "n_constant": int(sum(1 for p in pre if str(p["saturation_kind"]).startswith("constant"))),
            "n_prefix": int(sum(1 for p in pre if p["saturation_kind"] == "right_saturated_prefix")),
            "n_full": int(sum(1 for p in pre if p["saturation_kind"] == "full")),
            "max_prefix_kmax": int(max_prefix),
            "v03_node_work_proxy": int(v03_node_work),
            "v04_node_work_proxy": int(v04_node_work),
            "v04_vs_v03_work_ratio": float(v04_node_work / v03_node_work) if v03_node_work else 0.0,
            "n_v04_prefix_buckets": int(len(buckets)),
            "grouping_strategy": grouping_strategy,
            "seconds_preflight": float(time.perf_counter() - b0),
        })
        for p in pre:
            rep = str(p["saturation_kind"])
            storage_bytes = _storage_bytes_for_representation(str(budget.storage), int(p["prefix_kmax"]), int(build.Kmax), rep)
            base_nodes, ref_nodes = (0, 0) if rep.startswith("constant") else _node_counts_for_prefix(build, int(p["prefix_kmax"]))
            rows.append({
                "bundle_idx": int(bundle_idx),
                "R": float(spec.base_point.R), "T": float(spec.base_point.T), "N": float(spec.base_point.N), "depth": int(spec.base_point.depth),
                "alpha": float(spec.alpha), "alpha_index": int(spec.alpha_index),
                "theta_f": float(p["theta_f"]),
                "mean": p.get("mean"), "variance": p.get("variance"), "second_factorial_moment": p.get("second_factorial_moment"),
                "representation": rep,
                "prefix_kmax": int(p["prefix_kmax"]),
                "saturation_start_k": p.get("saturation_start_k"),
                "tail_bound": p.get("tail_bound"),
                "tail_certificate_method": p.get("tail_certificate_method"),
                "base_nodes": int(base_nodes), "refined_nodes": int(ref_nodes),
                "node_work_proxy": int(base_nodes + ref_nodes),
                "storage_bytes_raw": int(storage_bytes),
            })
    elapsed = time.perf_counter() - t0
    n_tables = len(rows)
    n_constant = int(sum(1 for r in rows if str(r["representation"]).startswith("constant")))
    n_prefix = int(sum(1 for r in rows if r["representation"] == "right_saturated_prefix"))
    n_full = int(sum(1 for r in rows if r["representation"] == "full"))
    storage_raw = int(sum(int(r["storage_bytes_raw"]) for r in rows))
    dense_raw = int(n_tables * (int(build.Kmax) + 1) * (2 if budget.storage == "int16" else 4))
    node_work_tablewise = int(sum(int(r["node_work_proxy"]) for r in rows))
    node_work_v03 = int(sum(int(r["v03_node_work_proxy"]) for r in bundle_rows))
    node_work_v04 = int(sum(int(r["v04_node_work_proxy"]) for r in bundle_rows))
    prefix_vals = [int(r["prefix_kmax"]) for r in rows if int(r["prefix_kmax"]) >= 0]
    def q(v: float) -> Optional[float]:
        if not prefix_vals: return None
        import numpy as np
        return float(np.quantile(prefix_vals, v))
    summary = {
        "format": "tailbin_adaptive_cache_plan_v1_0",
        "elapsed_seconds": float(elapsed),
        "grid": grid.to_dict(),
        "build_config": asdict(build),
        "error_budget": asdict(budget),
        "n_bundles_expected": int(n_bundles_expected),
        "n_bundles_planned": int(len(bundles)),
        "n_tables_planned": int(n_tables),
        "n_constant_tables": n_constant,
        "n_prefix_tables": n_prefix,
        "n_full_tables": n_full,
        "fraction_constant": float(n_constant / n_tables) if n_tables else 0.0,
        "fraction_compact_prefix_or_constant": float((n_constant + n_prefix) / n_tables) if n_tables else 0.0,
        "raw_storage_bytes_adaptive": storage_raw,
        "raw_storage_gb_adaptive": float(storage_raw / 1e9),
        "raw_storage_bytes_dense": dense_raw,
        "raw_storage_gb_dense": float(dense_raw / 1e9),
        "adaptive_storage_ratio_vs_dense": float(storage_raw / dense_raw) if dense_raw else 0.0,
        "node_work_proxy_tablewise_no_theta_reuse": node_work_tablewise,
        "node_work_proxy_v03_one_max_prefix": node_work_v03,
        "node_work_proxy_v04_prefix_bucketed": node_work_v04,
        "v04_vs_v03_node_work_ratio": float(node_work_v04 / node_work_v03) if node_work_v03 else 0.0,
        "prefix_kmax_quantiles": {"q50": q(0.5), "q75": q(0.75), "q90": q(0.9), "q95": q(0.95), "q99": q(0.99), "max": max(prefix_vals) if prefix_vals else None},
        "notes": [
            "This is a moment-preflight plan, not a coefficient-table build.",
            "Node-work proxies count base+refined FFT nodes after saturation decisions; actual runtime also depends on ODE/grid cost.",
            "the planner chooses the cheaper of one-max-prefix grouping and prefix-node bucketing for each theta bundle.",
        ],
    }
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "plan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        if rows:
            with (out / "plan_tables.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        if bundle_rows:
            with (out / "plan_bundles.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(bundle_rows[0].keys())); w.writeheader(); w.writerows(bundle_rows)
    summary["rows_preview"] = rows[: min(10, len(rows))]
    summary["bundle_rows_preview"] = bundle_rows[: min(10, len(bundle_rows))]
    return summary
