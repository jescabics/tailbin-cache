
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
import json
import time

import yaml

from .builder import BuildConfig, DenseCacheBuilder, ErrorBudget, TableSpec
from .refinement import refinement_config_from_dict, RefinementConfig
from .grid import grid_from_dict


def config_from_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def build_config_from_dict(cfg: Dict[str, Any]) -> tuple[BuildConfig, ErrorBudget, int]:
    b = cfg.get("build", cfg)
    eb = ErrorBudget(
        target_max_abs_z_error=float(b.get("target_max_abs_z_error", 1e-2)),
        target_max_abs_cdf_error=float(b.get("target_max_abs_cdf_error", 1e-5)),
        clip_eps=float(b.get("clip_eps", 1e-12)),
        z_clip=float(b.get("z_clip", 7.05)),
        storage=str(b.get("storage", "int16")),
    )
    bc = BuildConfig(
        Kmax=int(b.get("Kmax", 50_000)),
        base_node_factor=float(b.get("base_node_factor", 1.0)),
        refine_node_factor=float(b.get("refine_node_factor", 2.0)),
        base_alias_eta=float(b.get("base_alias_eta", 30.0)),
        refine_alias_eta=float(b.get("refine_alias_eta", 36.0)),
        n_bins=int(b.get("n_bins", 24)),
        steps_per_time=float(b.get("steps_per_time", 1.4)),
        min_steps=int(b.get("min_steps", 32)),
        max_steps=int(b.get("max_steps", 120)),
        ode_rtol=float(b.get("ode_rtol", 1e-7)),
        ode_atol=float(b.get("ode_atol", 1e-9)),
        use_solve_ivp=bool(b.get("use_solve_ivp", False)),
        node_budget=b.get("node_budget", None),
        pgf_backend=str(b.get("pgf_backend", "batched")),
        batch_size=int(b.get("batch_size", 128)),
        compressed_npz=bool(b.get("compressed_npz", True)),
        use_conjugate_symmetry=bool(b.get("use_conjugate_symmetry", True)),
        use_embedded_refinement=bool(b.get("use_embedded_refinement", True)),
        max_refined_nodes=(None if b.get("max_refined_nodes", None) is None else int(b.get("max_refined_nodes"))),
        stable_pgf_fallback=bool(b.get("stable_pgf_fallback", True)),
        stable_rk4_fallback_step_multiplier=float(b.get("stable_rk4_fallback_step_multiplier", 10.0)),
        stable_rk4_fallback_max_steps=int(b.get("stable_rk4_fallback_max_steps", 2000)),
        solve_ivp_fallback_node_cap=(None if b.get("solve_ivp_fallback_node_cap", 4096) is None else int(b.get("solve_ivp_fallback_node_cap", 4096))),
        chernoff_tail_enabled=bool(b.get("chernoff_tail_enabled", True)),
        chernoff_prefix_threshold=int(b.get("chernoff_prefix_threshold", 2048)),
    )
    n_jobs = int(cfg.get("parallel", {}).get("n_jobs", 1))
    return bc, eb, n_jobs


def _build_one(args: Tuple[int, Any, float, int, BuildConfig, ErrorBudget, str, bool]) -> Dict[str, Any]:
    idx, point, alpha, ai, build_cfg, budget, tables_dir_str, resume = args
    tables_dir = Path(tables_dir_str)
    spec = TableSpec(point=point, alpha=alpha, alpha_index=ai)
    fname = tables_dir / f"{idx:06d}_{spec.key()}.npz"
    json_name = fname.with_suffix(".json")
    if resume and fname.exists():
        try:
            from .builder import DenseZCache
            cache = DenseZCache(fname)
            meta = cache.metadata
            skipped = True
        except Exception:
            meta = json.loads(json_name.read_text()) if json_name.exists() else {}
            skipped = True
    elif resume and json_name.exists() and not fname.exists():
        meta = json.loads(json_name.read_text())
        skipped = True
    else:
        meta = DenseCacheBuilder(spec, build_cfg, budget).build_table(fname)
        skipped = False
    return {
        "idx": idx,
        "path": str(fname if fname.exists() else fname.with_suffix(".json")),
        "certified": bool(meta.get("certified", False)),
        "status": meta.get("status"),
        "seconds": float(meta.get("seconds", 0.0)),
        "skipped_existing": bool(skipped),
        "total_z_error_indicator": meta.get("total_z_error_indicator"),
        "total_cdf_error_indicator": meta.get("total_cdf_error_indicator"),
        "base_estimated_full_seconds": (meta.get("base_meta") or {}).get("estimated_full_seconds"),
        "refined_estimated_full_seconds": (meta.get("refined_meta") or {}).get("estimated_full_seconds"),
        "R": float(point.R),
        "T": float(point.T),
        "theta_f": float(point.theta_f),
        "N": float(point.N),
        "depth": int(point.depth),
        "alpha": float(alpha),
        "alpha_index": int(ai),
    }


def build_from_config(config_path: str | Path, output_dir: str | Path, *, limit_tables: Optional[int] = None) -> Dict[str, Any]:
    cfg = config_from_yaml(config_path)
    grid = grid_from_dict(cfg)
    build_cfg, budget, n_jobs = build_config_from_dict(cfg)
    resume = bool(cfg.get("build", {}).get("resume", True))
    out = Path(output_dir)
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    specs = list(grid.table_specs())
    if limit_tables is not None:
        specs = specs[: int(limit_tables)]
    tasks = [(idx, point, alpha, ai, build_cfg, budget, str(tables_dir), resume) for idx, (point, alpha, ai) in enumerate(specs)]
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    if int(n_jobs) <= 1 or len(tasks) <= 1:
        for task in tasks:
            rows.append(_build_one(task))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            futs = [ex.submit(_build_one, task) for task in tasks]
            for fut in as_completed(futs):
                rows.append(fut.result())
        rows.sort(key=lambda r: int(r["idx"]))

    elapsed = time.perf_counter() - t0
    ncert = int(sum(1 for r in rows if r["certified"]))
    summary = {
        "format": "tailbin_cache_build_summary_v1_0",
        "config_path": str(config_path),
        "output_dir": str(out),
        "grid": grid.to_dict(),
        "build_config": build_cfg.__dict__,
        "error_budget": budget.__dict__,
        "parallel": {"n_jobs": int(n_jobs), "resume": bool(resume)},
        "n_tables_expected": int(grid.n_tables),
        "n_tables_attempted": len(rows),
        "n_tables_certified": ncert,
        "certified_fraction_attempted": float(ncert / len(rows)) if rows else 0.0,
        "elapsed_seconds": float(elapsed),
        "mean_seconds_per_table": float(sum(r["seconds"] for r in rows) / len(rows)) if rows else None,
        "wall_seconds_per_attempted_table": float(elapsed / len(rows)) if rows else None,
        "rows": rows,
        "notes": "legacy dense-table builder; production use should prefer build-hdf5.",
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    if rows:
        import csv
        with (out / "tables.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return summary


def _build_theta_one(args: Tuple[int, Any, Sequence[float], BuildConfig, ErrorBudget, str]) -> Dict[str, Any]:
    i, spec, theta_values, build_cfg, budget, tables_dir_str = args
    from .theta_bundle import DenseThetaBundleBuilder
    meta = DenseThetaBundleBuilder(spec, build_cfg, budget).build_bundle(Path(tables_dir_str), table_index_prefix=f"{i:06d}_")
    return {
        "bundle_idx": i,
        "certified": bool(meta.get("certified", False)),
        "status": meta.get("status"),
        "seconds": float(meta.get("seconds", 0.0)),
        "n_theta": len(theta_values),
        "n_tables_written": len(meta.get("tables", [])),
        "n_tables_certified": int(sum(1 for t in meta.get("tables", []) if t.get("certified"))),
        "R": float(spec.base_point.R),
        "T": float(spec.base_point.T),
        "N": float(spec.base_point.N),
        "depth": int(spec.base_point.depth),
        "alpha": float(spec.alpha),
        "alpha_index": int(spec.alpha_index),
    }


def build_theta_bundles_from_config(config_path: str | Path, output_dir: str | Path, *, limit_bundles: Optional[int] = None) -> Dict[str, Any]:
    """Build tables grouped over theta_f values.

    This is the recommended mode when the founder-load grid has multiple values,
    because the finite-depth ODE is independent of theta_f and can be reused.
    """
    cfg = config_from_yaml(config_path)
    grid = grid_from_dict(cfg)
    build_cfg, budget, n_jobs = build_config_from_dict(cfg)
    out = Path(output_dir)
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    from .theta_bundle import DenseThetaBundleBuilder, ThetaBundleSpec
    from .grid import ParameterPoint

    # Unique base points excluding theta_f.
    base_points = []
    seen = set()
    for p in grid.parameter_points():
        key = (float(p.R), float(p.T), float(p.N), int(p.depth), float(p.u), float(p.ploidy_factor), float(p.lam), bool(p.condition_on_survival))
        if key not in seen:
            seen.add(key)
            base_points.append(ParameterPoint(R=p.R, T=p.T, theta_f=0.0, N=p.N, depth=p.depth, u=p.u, ploidy_factor=p.ploidy_factor, lam=p.lam, condition_on_survival=p.condition_on_survival))
    bundles = []
    idx = 0
    for bp in base_points:
        theta_values = grid.valid_theta_values_for_T(bp.T)
        if not theta_values:
            continue
        for ai, alpha in enumerate(grid.alphas):
            bundles.append((idx, ThetaBundleSpec(base_point=bp, theta_values=theta_values, alpha=float(alpha), alpha_index=int(ai))))
            idx += 1
    if limit_bundles is not None:
        bundles = bundles[: int(limit_bundles)]
    t0 = time.perf_counter()

    tasks = [(i, spec, spec.theta_values, build_cfg, budget, str(tables_dir)) for i, spec in bundles]

    rows: List[Dict[str, Any]] = []
    if int(n_jobs) <= 1 or len(bundles) <= 1:
        for task in tasks:
            rows.append(_build_theta_one(task))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            futs = [ex.submit(_build_theta_one, task) for task in tasks]
            for fut in as_completed(futs):
                rows.append(fut.result())
        rows.sort(key=lambda r: int(r["bundle_idx"]))
    elapsed = time.perf_counter() - t0
    n_tables_attempted = int(sum(r["n_tables_written"] for r in rows))
    n_tables_certified = int(sum(r["n_tables_certified"] for r in rows))
    summary = {
        "format": "tailbin_theta_bundle_build_summary_v1_0",
        "config_path": str(config_path),
        "output_dir": str(out),
        "grid": grid.to_dict(),
        "build_config": build_cfg.__dict__,
        "error_budget": budget.__dict__,
        "parallel": {"n_jobs": int(n_jobs)},
        "n_theta_values": len(theta_values),
        "n_bundles_expected": int(len(base_points) * len(grid.alphas)),
        "n_bundles_attempted": len(rows),
        "n_bundles_certified": int(sum(1 for r in rows if r["certified"])),
        "n_tables_attempted": n_tables_attempted,
        "n_tables_certified": n_tables_certified,
        "certified_fraction_attempted": float(n_tables_certified / n_tables_attempted) if n_tables_attempted else 0.0,
        "elapsed_seconds": float(elapsed),
        "mean_seconds_per_bundle": float(sum(r["seconds"] for r in rows) / len(rows)) if rows else None,
        "wall_seconds_per_bundle": float(elapsed / len(rows)) if rows else None,
        "amortized_wall_seconds_per_table": float(elapsed / n_tables_attempted) if n_tables_attempted else None,
        "rows": rows,
        "notes": "Theta-bundled builder reuses ODE integrations across founder loads; tables are still written per theta_f for simple lookup.",
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    if rows:
        import csv
        with (out / "bundles.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return summary


def _build_adaptive_theta_one(args: Tuple[int, Any, Sequence[float], BuildConfig, ErrorBudget, str]) -> Dict[str, Any]:
    i, spec, theta_values, build_cfg, budget, tables_dir_str = args
    from .adaptive_bundle import AdaptiveThetaBundleBuilder
    meta = AdaptiveThetaBundleBuilder(spec, build_cfg, budget).build_bundle(Path(tables_dir_str), table_index_prefix=f"{i:06d}_")
    return {
        "bundle_idx": i,
        "certified": bool(meta.get("certified", False)),
        "status": meta.get("status"),
        "seconds": float(meta.get("seconds", 0.0)),
        "n_theta": len(theta_values),
        "n_tables_written": int(meta.get("n_tables_written", 0)),
        "n_tables_certified": int(meta.get("n_tables_certified", 0)),
        "n_constant_tables": int(meta.get("n_constant_tables", 0)),
        "n_prefix_tables": int(meta.get("n_prefix_tables", 0)),
        "n_full_tables": int(meta.get("n_full_tables", 0)),
        "R": float(spec.base_point.R),
        "T": float(spec.base_point.T),
        "N": float(spec.base_point.N),
        "depth": int(spec.base_point.depth),
        "alpha": float(spec.alpha),
        "alpha_index": int(spec.alpha_index),
    }


def build_adaptive_theta_bundles_from_config(config_path: str | Path, output_dir: str | Path, *, limit_bundles: Optional[int] = None) -> Dict[str, Any]:
    """Build compact/adaptive theta bundles with moment-certified saturation.

    This is the recommended adaptive mode. It first computes finite-depth moments for
    each theta value and uses one-sided moment certificates to skip full Kmax
    coefficient extraction when the clipped CDF is already saturated.
    """
    cfg = config_from_yaml(config_path)
    grid = grid_from_dict(cfg)
    build_cfg, budget, n_jobs = build_config_from_dict(cfg)
    out = Path(output_dir)
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    from .theta_bundle import ThetaBundleSpec
    from .grid import ParameterPoint

    base_points = []
    seen = set()
    for p in grid.parameter_points():
        key = (float(p.R), float(p.T), float(p.N), int(p.depth), float(p.u), float(p.ploidy_factor), float(p.lam), bool(p.condition_on_survival))
        if key not in seen:
            seen.add(key)
            base_points.append(ParameterPoint(R=p.R, T=p.T, theta_f=0.0, N=p.N, depth=p.depth, u=p.u, ploidy_factor=p.ploidy_factor, lam=p.lam, condition_on_survival=p.condition_on_survival))
    bundles = []
    idx = 0
    for bp in base_points:
        theta_values = grid.valid_theta_values_for_T(bp.T)
        if not theta_values:
            continue
        for ai, alpha in enumerate(grid.alphas):
            bundles.append((idx, ThetaBundleSpec(base_point=bp, theta_values=theta_values, alpha=float(alpha), alpha_index=int(ai))))
            idx += 1
    if limit_bundles is not None:
        bundles = bundles[: int(limit_bundles)]
    tasks = [(i, spec, spec.theta_values, build_cfg, budget, str(tables_dir)) for i, spec in bundles]
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    if int(n_jobs) <= 1 or len(tasks) <= 1:
        for task in tasks:
            rows.append(_build_adaptive_theta_one(task))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            futs = [ex.submit(_build_adaptive_theta_one, task) for task in tasks]
            for fut in as_completed(futs):
                rows.append(fut.result())
        rows.sort(key=lambda r: int(r["bundle_idx"]))
    elapsed = time.perf_counter() - t0
    n_tables_attempted = int(sum(r["n_tables_written"] for r in rows))
    n_tables_certified = int(sum(r["n_tables_certified"] for r in rows))
    summary = {
        "format": "tailbin_adaptive_theta_bundle_build_summary_v1_0",
        "config_path": str(config_path),
        "output_dir": str(out),
        "grid": grid.to_dict(),
        "build_config": build_cfg.__dict__,
        "error_budget": budget.__dict__,
        "parallel": {"n_jobs": int(n_jobs)},
        "n_theta_values": len(theta_values),
        "n_bundles_expected": int(len(base_points) * len(grid.alphas)),
        "n_bundles_attempted": len(rows),
        "n_bundles_certified": int(sum(1 for r in rows if r["certified"])),
        "n_tables_attempted": n_tables_attempted,
        "n_tables_certified": n_tables_certified,
        "certified_fraction_attempted": float(n_tables_certified / n_tables_attempted) if n_tables_attempted else 0.0,
        "n_constant_tables": int(sum(r["n_constant_tables"] for r in rows)),
        "n_prefix_tables": int(sum(r["n_prefix_tables"] for r in rows)),
        "n_full_tables": int(sum(r["n_full_tables"] for r in rows)),
        "elapsed_seconds": float(elapsed),
        "mean_seconds_per_bundle": float(sum(r["seconds"] for r in rows) / len(rows)) if rows else None,
        "wall_seconds_per_bundle": float(elapsed / len(rows)) if rows else None,
        "amortized_wall_seconds_per_table": float(elapsed / n_tables_attempted) if n_tables_attempted else None,
        "rows": rows,
        "notes": "adaptive adaptive builder uses moment-certified saturation and compact tables before falling back to coefficient extraction.",
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    if rows:
        import csv
        with (out / "bundles.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return summary
