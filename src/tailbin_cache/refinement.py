from __future__ import annotations

"""Automatic refinement ladder for production cache generation.

The cache builder never treats a failed numerical check as acceptable.  This
module retries nonconstant tables with increasingly stronger numerical settings
until all tables in the requested bundle certify or a declared budget is hit.

The ladder is deliberately conservative and auditable.  It records every attempt
and annotates the final NPZ/HDF5 table metadata with the level that certified it.
The certificate remains the usual cache-builder certificate: base/refined z-space
agreement, CDF-space agreement, storage error, monotonicity, finite-z checks, and
analytic tail/alias bounds.  This is not interval arithmetic, but it is a strict
refine-until-certified numerical workflow.
"""

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math
import time

import numpy as np

from .builder import BuildConfig, ErrorBudget
from .adaptive_bundle import AdaptiveThetaBundleBuilder
from .theta_bundle import ThetaBundleSpec


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = True
    max_levels: int = 4
    node_factor_growth: float = 1.6
    alias_eta_growth: float = 4.0
    n_bins_increment: int = 4
    steps_growth: float = 1.25
    max_steps_growth: float = 1.35
    max_seconds_per_bundle: float = 1800.0
    fail_on_uncertified: bool = False
    keep_failed_attempts: bool = False


def refinement_config_from_dict(cfg: Dict[str, Any]) -> RefinementConfig:
    r = cfg.get("refinement", {}) or {}
    return RefinementConfig(
        enabled=bool(r.get("enabled", True)),
        max_levels=int(r.get("max_levels", 4)),
        node_factor_growth=float(r.get("node_factor_growth", 1.6)),
        alias_eta_growth=float(r.get("alias_eta_growth", 4.0)),
        n_bins_increment=int(r.get("n_bins_increment", 4)),
        steps_growth=float(r.get("steps_growth", 1.25)),
        max_steps_growth=float(r.get("max_steps_growth", 1.35)),
        max_seconds_per_bundle=float(r.get("max_seconds_per_bundle", 1800.0)),
        fail_on_uncertified=bool(r.get("fail_on_uncertified", False)),
        keep_failed_attempts=bool(r.get("keep_failed_attempts", False)),
    )


def build_config_for_level(base: BuildConfig, ref: RefinementConfig, level: int) -> BuildConfig:
    if level <= 0:
        return base
    g = float(ref.node_factor_growth) ** float(level)
    sg = float(ref.steps_growth) ** float(level)
    mg = float(ref.max_steps_growth) ** float(level)
    return replace(
        base,
        base_node_factor=float(base.base_node_factor) * g,
        refine_node_factor=float(base.refine_node_factor) * g,
        base_alias_eta=float(base.base_alias_eta) + float(ref.alias_eta_growth) * level,
        refine_alias_eta=float(base.refine_alias_eta) + float(ref.alias_eta_growth) * level,
        n_bins=int(base.n_bins) + int(ref.n_bins_increment) * level,
        steps_per_time=float(base.steps_per_time) * sg,
        min_steps=max(int(base.min_steps), int(math.ceil(float(base.min_steps) * sg))),
        max_steps=max(int(base.max_steps), int(math.ceil(float(base.max_steps) * mg))),
    )


def _read_npz_meta(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        raw = z["metadata"]
        if hasattr(raw, "item"):
            raw = raw.item()
        return json.loads(str(raw))


def _annotate_npz(path: Path, annotation: Dict[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: np.asarray(z[name]) for name in z.files if name != "metadata"}
        raw = z["metadata"]
        if hasattr(raw, "item"):
            raw = raw.item()
        meta = json.loads(str(raw))
    meta.setdefault("refinement_ladder", {}).update(annotation)
    meta["refinement_level"] = int(annotation.get("selected_level", 0))
    # Mark explicitly that certification is by the full ladder, not just one attempt.
    if bool(meta.get("certified", False)):
        meta["status"] = str(meta.get("status", "certified")) + "_via_refinement_ladder"
    arrays["metadata"] = json.dumps(meta, sort_keys=True)
    np.savez_compressed(path, **arrays)


def _bundle_attempt_summary(meta: Dict[str, Any], level: int, build: BuildConfig, seconds: float) -> Dict[str, Any]:
    tables = meta.get("tables", []) or []
    return {
        "level": int(level),
        "certified": bool(meta.get("certified", False)),
        "status": str(meta.get("status", "unknown")),
        "seconds": float(seconds),
        "n_tables_written": int(len(tables)),
        "n_tables_certified": int(sum(1 for t in tables if t.get("certified"))),
        "max_total_z_error_indicator": float(max([float(t.get("total_z_error_indicator") or 0.0) for t in tables] or [0.0])),
        "max_total_cdf_error_indicator": float(max([float(t.get("total_cdf_error_indicator") or 0.0) for t in tables] or [0.0])),
        "base_node_factor": float(build.base_node_factor),
        "refine_node_factor": float(build.refine_node_factor),
        "base_alias_eta": float(build.base_alias_eta),
        "refine_alias_eta": float(build.refine_alias_eta),
        "n_bins": int(build.n_bins),
        "steps_per_time": float(build.steps_per_time),
        "min_steps": int(build.min_steps),
        "max_steps": int(build.max_steps),
    }


def build_bundle_with_ladder(
    spec: ThetaBundleSpec,
    base_build: BuildConfig,
    budget: ErrorBudget,
    refinement: RefinementConfig,
    output_dir: str | Path,
    *,
    preflight: Optional[List[Dict[str, Any]]] = None,
    table_index_prefix: str = "",
) -> Dict[str, Any]:
    """Build a theta bundle with automatic numerical refinement.

    The selected attempt's NPZ files are left in output_dir.  Failed attempts are
    removed unless keep_failed_attempts=True.  The return metadata includes the
    full ladder history and the selected level.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    levels = int(refinement.max_levels) if bool(refinement.enabled) else 1
    levels = max(1, levels)
    start_all = time.perf_counter()
    attempts: List[Dict[str, Any]] = []
    selected_meta: Optional[Dict[str, Any]] = None
    selected_level = -1
    last_meta: Optional[Dict[str, Any]] = None

    for level in range(levels):
        if (time.perf_counter() - start_all) > float(refinement.max_seconds_per_bundle):
            attempts.append({"level": int(level), "status": "budget_exhausted_before_attempt", "certified": False, "seconds": 0.0})
            break
        build = build_config_for_level(base_build, refinement, level)
        attempt_dir = out / f"__attempt_level_{level}"
        if attempt_dir.exists():
            import shutil
            shutil.rmtree(attempt_dir)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        builder = AdaptiveThetaBundleBuilder(spec, build, budget)
        if preflight is None:
            meta = builder.build_bundle(attempt_dir, table_index_prefix=table_index_prefix)
        else:
            meta = builder.build_bundle_with_preflight(attempt_dir, preflight, table_index_prefix=table_index_prefix)
        sec = time.perf_counter() - t0
        last_meta = meta
        attempts.append(_bundle_attempt_summary(meta, level, build, sec))
        if bool(meta.get("certified", False)):
            selected_meta = meta
            selected_level = level
            # Move NPZ files from the successful attempt into output_dir.
            for p in attempt_dir.glob("*.npz"):
                _annotate_npz(p, {
                    "selected_level": int(selected_level),
                    "attempts": attempts,
                    "refinement_config": asdict(refinement),
                })
                target = out / p.name
                if target.exists():
                    target.unlink()
                p.rename(target)
            # Keep the bundle JSON with annotations.
            selected_meta = dict(selected_meta)
            selected_meta["refinement_ladder"] = {
                "selected_level": int(selected_level),
                "attempts": attempts,
                "refinement_config": asdict(refinement),
                "total_seconds": float(time.perf_counter() - start_all),
            }
            (out / f"{table_index_prefix}{spec.key_prefix()}_adaptive_bundle_refined.json").write_text(json.dumps(selected_meta, indent=2, sort_keys=True))
            if not bool(refinement.keep_failed_attempts):
                import shutil
                for d in out.glob("__attempt_level_*"):
                    if d.exists():
                        shutil.rmtree(d)
            return selected_meta
        if not bool(refinement.keep_failed_attempts):
            import shutil
            shutil.rmtree(attempt_dir)

    # No level certified.  Keep the last attempt's files if requested; otherwise
    # leave no NPZ files and return a failure summary so the HDF5 builder can log
    # the failed hard table without silently writing uncertified data.
    fail_meta = dict(last_meta or {})
    fail_meta.update({
        "format": "tailbin_refinement_ladder_failure_v1_0",
        "certified": False,
        "status": "refinement_ladder_exhausted_uncertified",
        "seconds": float(time.perf_counter() - start_all),
        "spec": {"base_point": spec.integration_point().to_dict(), "theta_values": [float(x) for x in spec.theta_values], "alpha": float(spec.alpha), "alpha_index": int(spec.alpha_index)},
        "build_config": asdict(base_build),
        "error_budget": asdict(budget),
        "refinement_ladder": {
            "selected_level": None,
            "attempts": attempts,
            "refinement_config": asdict(refinement),
            "total_seconds": float(time.perf_counter() - start_all),
        },
    })
    (out / f"{table_index_prefix}{spec.key_prefix()}_refinement_failed.json").write_text(json.dumps(fail_meta, indent=2, sort_keys=True))
    if bool(refinement.fail_on_uncertified):
        raise RuntimeError(f"Refinement ladder exhausted without certification for {spec.key_prefix()}")
    return fail_meta
