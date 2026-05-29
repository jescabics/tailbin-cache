from __future__ import annotations

from pathlib import Path
import argparse
import json

from .grid import write_default_config, grid_from_dict
from .runner import config_from_yaml, build_config_from_dict
from .estimate import estimate_storage, estimate_runtime, estimate_theta_bundled_runtime


def _json_print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="tailbin-cache",
        description="Build certified HDF5 lookup caches z(k)=Phi^{-1}(P(Y<=k)) for read-sampled tail-bin counts.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init-config", help="write a smoke or production YAML config")
    sp.add_argument("--output", required=True)
    sp.add_argument("--production", action="store_true", help="write production-style defaults instead of a tiny smoke config")

    sp = sub.add_parser("estimate", help="estimate dense storage and rough runtime from a config")
    sp.add_argument("--config", required=True)
    sp.add_argument("--compression-ratio", type=float, default=2.0)
    sp.add_argument("--seconds-per-bundle", type=float, default=None, help="optional assumed seconds per (R,T,N,depth,alpha) theta-bundle")
    sp.add_argument("--seconds-per-expanded-point", type=float, default=None, help="optional assumed seconds per expanded (R,T,theta,N,depth) point across all alphas")

    sp = sub.add_parser("plan", help="fast adaptive preflight plan: constants, prefixes, full tables, storage")
    sp.add_argument("--config", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--limit-bundles", type=int, default=None)

    sp = sub.add_parser("plan-shards", help="make a difficulty-balanced shard plan for parallel HDF5 builds")
    sp.add_argument("--config", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--n-shards", type=int, default=20)
    sp.add_argument("--limit-bundles", type=int, default=None)

    sp = sub.add_parser("build-hdf5", help="recommended production builder: alpha-monotone adaptive HDF5 cache shard")
    sp.add_argument("--config", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--limit-base-points", type=int, default=None, help="smoke/timing: build only first N base points in this shard")
    sp.add_argument("--n-shards", type=int, default=1)
    sp.add_argument("--shard-index", type=int, default=0)
    sp.add_argument("--compression", default="gzip")
    sp.add_argument("--compression-opts", type=int, default=4)

    sp = sub.add_parser("inspect-hdf5", help="inspect one table row from a generated HDF5 shard")
    sp.add_argument("path")
    sp.add_argument("--index", type=int, default=0)
    sp.add_argument("--k", type=int, default=0)

    args = p.parse_args(argv)

    if args.cmd == "init-config":
        write_default_config(args.output, smoke=not args.production)
        print(f"wrote {args.output}")
        return 0

    if args.cmd == "estimate":
        cfg = config_from_yaml(args.config)
        grid = grid_from_dict(cfg)
        build_cfg, budget, n_jobs = build_config_from_dict(cfg)
        out = {
            "dense_storage": estimate_storage(grid, build_cfg.Kmax, budget.storage, args.compression_ratio),
            "grid": {
                "n_parameter_points_expanded": grid.n_parameter_points,
                "n_alpha_cutoffs": len(grid.alphas),
                "n_dense_tables": grid.n_tables,
                "Kmax": build_cfg.Kmax,
                "ploidy_factor": grid.ploidy_factor,
                "max_age": grid.max_age,
                "age_constraint_enabled": grid.enforce_age_constraint,
            },
        }
        if args.seconds_per_bundle is not None:
            out["theta_bundled_runtime"] = estimate_theta_bundled_runtime(grid, args.seconds_per_bundle, n_jobs=n_jobs)
        if args.seconds_per_expanded_point is not None:
            out["expanded_point_runtime"] = estimate_runtime(grid.n_parameter_points, args.seconds_per_expanded_point, n_jobs=n_jobs)
        _json_print(out)
        return 0

    if args.cmd == "plan":
        cfg = config_from_yaml(args.config)
        grid = grid_from_dict(cfg)
        build_cfg, budget, _n_jobs = build_config_from_dict(cfg)
        from .planner import plan_adaptive_cache
        summary = plan_adaptive_cache(grid, build_cfg, budget, limit_bundles=args.limit_bundles, output_dir=args.output_dir)
        keys = [
            "n_bundles_expected", "n_bundles_planned", "n_tables_planned",
            "n_constant_tables", "n_prefix_tables", "n_full_tables",
            "raw_storage_gb_adaptive", "raw_storage_gb_dense", "adaptive_storage_ratio_vs_dense",
            "v04_vs_v03_node_work_ratio", "elapsed_seconds",
        ]
        _json_print({k: summary[k] for k in keys if k in summary})
        return 0

    if args.cmd == "plan-shards":
        cfg = config_from_yaml(args.config)
        grid = grid_from_dict(cfg)
        build_cfg, budget, _n_jobs = build_config_from_dict(cfg)
        from .shards import balanced_shard_plan, write_balanced_shard_plan
        summary = balanced_shard_plan(grid, build_cfg, budget, n_shards=args.n_shards, limit_bundles=args.limit_bundles)
        write_balanced_shard_plan(summary, args.output_dir)
        keys = ["n_shards", "n_bundles_planned", "total_node_work_proxy", "max_shard_load", "min_shard_load", "load_imbalance_ratio", "elapsed_seconds"]
        _json_print({k: summary[k] for k in keys if k in summary})
        return 0

    if args.cmd == "build-hdf5":
        from .hdf5_alpha_monotone import build_alpha_monotone_hdf5_from_config
        summary = build_alpha_monotone_hdf5_from_config(
            args.config, args.output, limit_base_points=args.limit_base_points,
            n_shards=args.n_shards, shard_index=args.shard_index,
            compression=args.compression, compression_opts=args.compression_opts,
        )
        keys = [
            "n_base_points_attempted", "n_tables_written", "n_tables_certified", "certified_fraction_written",
            "n_constant_tables", "n_alpha_threshold_aliases", "n_alpha_monotone_propagated_constants",
            "n_nonconstant_tables", "n_refinement_failures", "elapsed_seconds", "mean_seconds_per_base_point", "output_mb",
        ]
        _json_print({k: summary[k] for k in keys if k in summary})
        return 0

    if args.cmd == "inspect-hdf5":
        from .hdf5_adaptive import HDF5AdaptiveCache
        c = HDF5AdaptiveCache(args.path)
        try:
            meta = c.metadata(args.index)
            out = {
                "path": str(args.path),
                "index": int(args.index),
                "k": int(args.k),
                "z": c.z(args.index, args.k),
                "cdf": c.cdf(args.index, args.k),
                "metadata": meta,
            }
        finally:
            c.close()
        _json_print(out)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
