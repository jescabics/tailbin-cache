from __future__ import annotations

"""Difficulty-balanced sharding for adaptive HDF5 builds.

Naive modulo sharding is simple but can leave one CPU with many hard bundles and
another CPU with mostly constant/moment-saturated bundles.  This module computes
an inexpensive moment-preflight cost proxy for every theta bundle and greedily
assigns bundles to shards so the total predicted FFT node work is balanced.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv
import json
import math
import time

from .adaptive_bundle import moment_preflight
from .builder import BuildConfig, ErrorBudget, next_pow2
from .grid import CacheGrid
from .hdf5_adaptive import adaptive_bundle_specs


def _node_counts_for_prefix(build: BuildConfig, prefix_kmax: int) -> tuple[int, int]:
    k = max(0, int(prefix_kmax))
    base = next_pow2(int(math.ceil(float(build.base_node_factor) * (k + 1))))
    ref = next_pow2(int(math.ceil(float(build.refine_node_factor) * (k + 1))))
    return int(base), int(ref)


def bundle_cost_proxy(spec, build: BuildConfig, budget: ErrorBudget) -> Dict[str, Any]:
    pre = moment_preflight(spec, build, budget)
    nonconstant = [p for p in pre if not str(p["saturation_kind"]).startswith("constant")]
    # Match adaptive builder logic: if several theta values require tables,
    # bucket by FFT node count rather than building all at the max prefix.
    buckets: Dict[tuple[int, int], int] = {}
    max_prefix = -1
    for p in nonconstant:
        pk = int(p["prefix_kmax"])
        max_prefix = max(max_prefix, pk)
        buckets[_node_counts_for_prefix(build, pk)] = buckets.get(_node_counts_for_prefix(build, pk), 0) + 1
    node_work = int(sum(base + ref for base, ref in buckets))
    return {
        "n_theta": int(len(spec.theta_values)),
        "n_constant": int(sum(1 for p in pre if str(p["saturation_kind"]).startswith("constant"))),
        "n_prefix": int(sum(1 for p in pre if p["saturation_kind"] == "right_saturated_prefix")),
        "n_full": int(sum(1 for p in pre if p["saturation_kind"] == "full")),
        "max_prefix_kmax": int(max_prefix),
        "n_prefix_buckets": int(len(buckets)),
        "node_work_proxy": int(node_work),
    }


def balanced_shard_plan(
    grid: CacheGrid,
    build: BuildConfig,
    budget: ErrorBudget,
    *,
    n_shards: int,
    limit_bundles: Optional[int] = None,
) -> Dict[str, Any]:
    if int(n_shards) < 1:
        raise ValueError("n_shards must be >= 1")
    t0 = time.perf_counter()
    bundles = adaptive_bundle_specs(grid)
    n_expected = len(bundles)
    if limit_bundles is not None:
        bundles = bundles[: int(limit_bundles)]
    bundle_rows: List[Dict[str, Any]] = []
    for bundle_idx, spec in bundles:
        c = bundle_cost_proxy(spec, build, budget)
        row = {
            "bundle_idx": int(bundle_idx),
            "R": float(spec.base_point.R),
            "T": float(spec.base_point.T),
            "N": float(spec.base_point.N),
            "depth": int(spec.base_point.depth),
            "ploidy_factor": float(spec.base_point.ploidy_factor),
            "alpha": float(spec.alpha),
            "alpha_index": int(spec.alpha_index),
            **c,
        }
        bundle_rows.append(row)
    # Greedy longest-processing-time scheduling by predicted node work.  Use a
    # small positive floor so all-constant bundles are still distributed.
    shard_rows: List[List[Dict[str, Any]]] = [[] for _ in range(int(n_shards))]
    shard_loads = [0.0 for _ in range(int(n_shards))]
    for row in sorted(bundle_rows, key=lambda r: (float(r["node_work_proxy"]), int(r["n_theta"])), reverse=True):
        j = min(range(int(n_shards)), key=lambda x: shard_loads[x])
        shard_rows[j].append(row)
        shard_loads[j] += max(1.0, float(row["node_work_proxy"]))
    rows_out: List[Dict[str, Any]] = []
    for shard_idx, rows in enumerate(shard_rows):
        for r in rows:
            rr = dict(r)
            rr["shard_index"] = int(shard_idx)
            rows_out.append(rr)
    per_shard = []
    for shard_idx, rows in enumerate(shard_rows):
        load = float(sum(max(1.0, float(r["node_work_proxy"])) for r in rows))
        per_shard.append({
            "shard_index": int(shard_idx),
            "n_bundles": int(len(rows)),
            "node_work_proxy": float(load),
            "n_constant_tables": int(sum(int(r["n_constant"]) for r in rows)),
            "n_nonconstant_tables": int(sum(int(r["n_theta"]) - int(r["n_constant"]) for r in rows)),
        })
    loads = [r["node_work_proxy"] for r in per_shard]
    summary = {
        "format": "tailbin_balanced_shard_plan_v1_0",
        "elapsed_seconds": float(time.perf_counter() - t0),
        "grid": grid.to_dict(),
        "build_config": asdict(build),
        "error_budget": asdict(budget),
        "n_shards": int(n_shards),
        "n_bundles_expected_total": int(n_expected),
        "n_bundles_planned": int(len(bundle_rows)),
        "total_node_work_proxy": float(sum(loads)),
        "max_shard_load": float(max(loads) if loads else 0.0),
        "min_shard_load": float(min(loads) if loads else 0.0),
        "load_imbalance_ratio": float(max(loads) / max(min(loads), 1.0)) if loads else 0.0,
        "per_shard": per_shard,
        "bundle_rows": rows_out,
    }
    return summary


def write_balanced_shard_plan(plan: Dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "balanced_shards.summary.json").write_text(json.dumps({k: v for k, v in plan.items() if k != "bundle_rows"}, indent=2, sort_keys=True))
    rows = list(plan.get("bundle_rows", []))
    if rows:
        with (out / "balanced_shards.bundles.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


def bundle_indices_for_shard(plan: Dict[str, Any], shard_index: int) -> set[int]:
    return {int(r["bundle_idx"]) for r in plan.get("bundle_rows", []) if int(r["shard_index"]) == int(shard_index)}


def read_balanced_shard_plan(path_or_dir: str | Path) -> Dict[str, Any]:
    """Read a shard plan produced by plan-shards.

    Accepts either the output directory or the bundles CSV path.  This lets
    large 20-CPU builds compute the expensive moment-preflight plan once and
    then launch many shard builders without each shard recomputing the global
    plan.
    """
    p = Path(path_or_dir)
    if p.is_dir():
        summary_path = p / "balanced_shards.summary.json"
        bundles_path = p / "balanced_shards.bundles.csv"
    else:
        bundles_path = p
        summary_path = p.with_name("balanced_shards.summary.json")
    summary: Dict[str, Any]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {"format": "tailbin_balanced_shard_plan_loaded_from_csv"}
    rows: List[Dict[str, Any]] = []
    if bundles_path.exists():
        with bundles_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rr: Dict[str, Any] = dict(r)
                for key in ["bundle_idx", "depth", "alpha_index", "n_theta", "n_constant", "n_prefix", "n_full", "max_prefix_kmax", "n_prefix_buckets", "node_work_proxy", "shard_index"]:
                    if key in rr and rr[key] not in (None, ""):
                        rr[key] = int(float(rr[key]))
                for key in ["R", "T", "N", "ploidy_factor", "alpha"]:
                    if key in rr and rr[key] not in (None, ""):
                        rr[key] = float(rr[key])
                rows.append(rr)
    summary["bundle_rows"] = rows
    return summary
