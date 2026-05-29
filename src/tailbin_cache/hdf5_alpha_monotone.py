
from __future__ import annotations

"""Alpha-monotone HDF5 builder.

This module adds a structural optimization that is exact for cumulative read-
frequency tail bins.  For a fixed parameter point and two cutoffs alpha_low <=
alpha_high, the tail count above the higher cutoff is stochastically bounded by
(the same mutations counted above) the lower cutoff:

    Y(alpha_high) <= Y(alpha_low)   almost surely.

Therefore any right-tail saturation certificate for the lower cutoff propagates
upward in alpha.  More generally, if a lower cutoff has certified
P(Y(alpha_low) >= K) <= eps, then every higher cutoff has the same right-tail
saturation from k>=K-1.  production propagates these prefix caps, not just the
constant-right special case K=1.

The builder processes all alphas for a base point in order, keeping these
right-saturation certificates per theta_f.  This is most useful in sparse/low-
mean regions, where many high cutoffs become constants.
"""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv, json, math, time
from datetime import datetime, timezone

import h5py
import numpy as np

from .adaptive_bundle import moment_preflight, AdaptiveThetaBundleBuilder
from .builder import BuildConfig, ErrorBudget
from .grid import CacheGrid, ParameterPoint, grid_from_dict
from .hdf5_adaptive import StreamingHDF5Writer, _constant_meta_from_preflight, _read_npz_payload, _unique_base_points
from .runner import build_config_from_dict, config_from_yaml
from .refinement import refinement_config_from_dict, build_bundle_with_ladder
from .theta_bundle import ThetaBundleSpec
from .alpha_opt import read_count_threshold


def _right_constant_preflight_from_source(theta: float, *, source_alpha: float, source_alpha_index: int, source_bound: float, build: BuildConfig, budget: ErrorBudget) -> Dict[str, Any]:
    return {
        "theta_f": float(theta),
        "mean": None,
        "variance": None,
        "second_factorial_moment": None,
        "moment_meta": {"source": "alpha_monotone_propagation"},
        "saturation_kind": "constant_right",
        "prefix_kmax": -1,
        "saturation_start_k": 0,
        "tail_bound": float(source_bound),
        "tail_certificate_method": "alpha_monotone_right_saturation",
        "alpha_monotone_source": {
            "source_alpha": float(source_alpha),
            "source_alpha_index": int(source_alpha_index),
            "explanation": "Since Y(alpha_high) <= Y(alpha_low), the right-tail certificate P(Y_low>=1)<=eps propagates to this higher cutoff.",
        },
    }


def _right_prefix_preflight_from_source(theta: float, *, source_alpha: float, source_alpha_index: int, saturation_start_k: int, source_bound: float, build: BuildConfig, budget: ErrorBudget) -> Dict[str, Any]:
    start = int(saturation_start_k)
    if start <= 0:
        return _right_constant_preflight_from_source(theta, source_alpha=source_alpha, source_alpha_index=source_alpha_index, source_bound=source_bound, build=build, budget=budget)
    return {
        "theta_f": float(theta),
        "mean": None,
        "variance": None,
        "second_factorial_moment": None,
        "moment_meta": {"source": "alpha_monotone_prefix_propagation"},
        "saturation_kind": "right_saturated_prefix",
        "prefix_kmax": int(start - 1),
        "saturation_start_k": int(start),
        "tail_bound": float(source_bound),
        "tail_certificate_method": "alpha_monotone_right_saturation_prefix",
        "alpha_monotone_source": {
            "source_alpha": float(source_alpha),
            "source_alpha_index": int(source_alpha_index),
            "saturation_start_k": int(start),
            "explanation": "Since Y(alpha_high) <= Y(alpha_low), the right-tail certificate P(Y_low>=K)<=eps propagates to this higher cutoff, so only the prefix k<K-1 needs coefficient extraction.",
        },
    }


def _bundle_key_base(point: ParameterPoint) -> Tuple[float, float, float, int, float, float, float, bool]:
    return (float(point.R), float(point.T), float(point.N), int(point.depth), float(point.u), float(point.ploidy_factor), float(point.lam), bool(point.condition_on_survival))


def _read_base_point_manifest(path: str | Path) -> set[int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"base point manifest not found: {p}")
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            rows = data.get("selected_base_points", [])
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return {int(row["base_idx"]) for row in rows}
    out: set[int] = set()
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "base_idx" not in (reader.fieldnames or []):
            raise ValueError(f"base point manifest must include a base_idx column: {p}")
        for row in reader:
            out.add(int(float(row["base_idx"])))
    return out


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _ProgressWriter:
    def __init__(self, path: Optional[str | Path], *, echo_stdout: bool = True):
        self.path = None if path is None else Path(path)
        self.echo_stdout = bool(echo_stdout)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **payload: Any) -> None:
        event = {"timestamp": _utc_now_iso(), "event_type": str(event_type), **payload}
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True, default=str) + "\n")
                f.flush()
        if self.echo_stdout:
            stage = event.get("stage", "build")
            base = event.get("base_point_ordinal")
            total = event.get("n_base_points_total")
            base_idx = event.get("base_idx")
            alpha = event.get("alpha")
            elapsed = event.get("cumulative_elapsed_seconds")
            pieces = [f"[{stage}]", str(event_type)]
            if base is not None and total is not None:
                pieces.append(f"base {int(base) + 1}/{int(total)}")
            if base_idx is not None:
                pieces.append(f"base_idx={base_idx}")
            if alpha is not None:
                pieces.append(f"alpha={float(alpha):.8g}")
            if elapsed is not None:
                pieces.append(f"elapsed={float(elapsed):.1f}s")
            print(" ".join(pieces), flush=True)


def build_alpha_monotone_hdf5_from_config(
    config_path: str | Path,
    output_path: str | Path,
    *,
    limit_base_points: Optional[int] = None,
    n_shards: int = 1,
    shard_index: int = 0,
    compression: str = "gzip",
    compression_opts: int = 4,
    base_point_manifest: Optional[str | Path] = None,
    pgf_backend: Optional[str] = None,
    require_pgf_backend: Optional[str] = None,
    progress_path: Optional[str | Path] = None,
    progress_stdout: bool = True,
    max_seconds_per_base_point: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
    fail_on_timeout: bool = True,
) -> Dict[str, Any]:
    """Build an adaptive HDF5 cache by sharding over base points and scanning alphas.

    This command is intentionally base-point sharded rather than bundle-sharded:
    keeping all alpha cutoffs for a base point in one process allows exact
    monotone propagation of right-tail saturation certificates.
    """
    cfg = config_from_yaml(config_path)
    grid = grid_from_dict(cfg)
    build_cfg, budget, _n_jobs = build_config_from_dict(cfg)
    requested_backend = str(pgf_backend) if pgf_backend not in (None, "") else str(build_cfg.pgf_backend)
    if pgf_backend not in (None, ""):
        build_cfg = replace(build_cfg, pgf_backend=str(pgf_backend))
    if require_pgf_backend not in (None, "") and str(build_cfg.pgf_backend) != str(require_pgf_backend):
        raise RuntimeError(
            f"requested build requires pgf_backend={require_pgf_backend!r}, "
            f"but effective backend is {build_cfg.pgf_backend!r}"
        )
    refinement_cfg = refinement_config_from_dict(cfg)
    if int(n_shards) < 1:
        raise ValueError("n_shards must be >=1")
    if not (0 <= int(shard_index) < int(n_shards)):
        raise ValueError("shard_index must satisfy 0 <= shard_index < n_shards")
    all_base = _unique_base_points(grid)
    selected_base_indices: Optional[set[int]] = None
    if base_point_manifest is not None:
        selected_base_indices = _read_base_point_manifest(base_point_manifest)
        missing = sorted(i for i in selected_base_indices if i < 0 or i >= len(all_base))
        if missing:
            raise ValueError(f"base point manifest contains out-of-range base_idx values: {missing[:10]}")
    base_points = [
        (i, bp)
        for i, bp in enumerate(all_base)
        if (selected_base_indices is None or int(i) in selected_base_indices)
        and int(i) % int(n_shards) == int(shard_index)
    ]
    if limit_base_points is not None:
        base_points = base_points[: int(limit_base_points)]

    t0 = time.perf_counter()
    deadline = None if timeout_seconds is None else float(t0 + float(timeout_seconds))
    progress = _ProgressWriter(progress_path, echo_stdout=progress_stdout)
    rows: List[Dict[str, Any]] = []

    def output_size() -> int:
        p = Path(output_path)
        return int(p.stat().st_size) if p.exists() else 0

    def timeout_check(stage: str, *, base_started: Optional[float] = None, base_idx: Optional[int] = None) -> None:
        now = time.perf_counter()
        if deadline is not None and now > deadline:
            progress.emit(
                "error",
                stage=stage,
                base_idx=base_idx,
                backend=str(build_cfg.pgf_backend),
                cumulative_elapsed_seconds=float(now - t0),
                output_file_size_bytes=output_size(),
                message=f"overall build timeout exceeded: {float(timeout_seconds)} seconds",
            )
            if fail_on_timeout:
                raise TimeoutError(f"overall build timeout exceeded: {float(timeout_seconds)} seconds")
        if base_started is not None and max_seconds_per_base_point is not None:
            base_elapsed = now - float(base_started)
            if base_elapsed > float(max_seconds_per_base_point):
                progress.emit(
                    "error",
                    stage=stage,
                    base_idx=base_idx,
                    backend=str(build_cfg.pgf_backend),
                    elapsed_seconds_base_point=float(base_elapsed),
                    cumulative_elapsed_seconds=float(now - t0),
                    output_file_size_bytes=output_size(),
                    message=f"base point exceeded max_seconds_per_base_point={float(max_seconds_per_base_point)}",
                )
                if fail_on_timeout:
                    raise TimeoutError(f"base point {base_idx} exceeded {float(max_seconds_per_base_point)} seconds")

    progress.emit(
        "start_build",
        stage="build_hdf5",
        config_path=str(config_path),
        output_path=str(output_path),
        base_point_manifest=None if base_point_manifest is None else str(base_point_manifest),
        requested_backend=requested_backend,
        backend=str(build_cfg.pgf_backend),
        Kmax=int(build_cfg.Kmax),
        n_base_points_total=int(len(base_points)),
        n_alpha_cutoffs=int(len(grid.alphas)),
    )

    try:
        with StreamingHDF5Writer(output_path, compression=compression, compression_opts=int(compression_opts)) as writer:
            for base_ordinal, (base_idx, bp) in enumerate(base_points):
                timeout_check("before_base_point", base_idx=int(base_idx))
                base_t0 = time.perf_counter()
                theta_values_all = [float(x) for x in grid.valid_theta_values_for_T(bp.T)]
                Tb_values = [float(theta) / float(bp.u) if float(bp.u) != 0.0 else math.inf for theta in theta_values_all]
                base_payload = {
                    "stage": "build_hdf5",
                    "base_point_ordinal": int(base_ordinal),
                    "n_base_points_total": int(len(base_points)),
                    "base_idx": int(base_idx),
                    "R": float(bp.R),
                    "N": float(bp.N),
                    "T": float(bp.T),
                    "Tb": float(Tb_values[0]) if len(Tb_values) == 1 else None,
                    "theta_f": float(theta_values_all[0]) if len(theta_values_all) == 1 else None,
                    "theta_values": theta_values_all,
                    "depth": int(bp.depth),
                    "Kmax": int(build_cfg.Kmax),
                    "n_alpha_cutoffs": int(len(grid.alphas)),
                    "backend": str(build_cfg.pgf_backend),
                }
                progress.emit(
                    "start_base_point",
                    **base_payload,
                    cumulative_elapsed_seconds=float(base_t0 - t0),
                    output_file_size_bytes=output_size(),
                )

                right_saturation: Dict[float, Dict[str, Any]] = {}
                threshold_sources: Dict[Tuple[float, int], int] = {}
                n_threshold_aliases = 0
                n_prop_constants = 0
                n_moment_constants = 0
                n_nonconstant = 0
                n_nonconstant_cert = 0
                n_refinement_failures = 0

                alpha_items = sorted([(int(ai), float(a)) for ai, a in enumerate(grid.alphas)], key=lambda x: x[1])
                for ai, alpha in alpha_items:
                    timeout_check("before_alpha", base_started=base_t0, base_idx=int(base_idx))
                    alpha_t0 = time.perf_counter()
                    tables_before = len(writer.rows)
                    cert_before = int(sum(1 for r in writer.rows if bool(r.get("certified", False))))
                    progress.emit(
                        "start_alpha",
                        **base_payload,
                        alpha=float(alpha),
                        alpha_index=int(ai),
                        cumulative_elapsed_seconds=float(alpha_t0 - t0),
                        tables_attempted=int(len(writer.rows)),
                        tables_written=int(len(writer.rows)),
                        tables_certified=int(cert_before),
                    )
                    q_threshold = read_count_threshold(float(alpha), int(bp.depth))
                    unresolved_thetas: List[float] = []
                    for theta in theta_values_all:
                        alias_src = threshold_sources.get((float(theta), int(q_threshold)))
                        if alias_src is not None:
                            src_row = dict(writer.rows[int(alias_src)])
                            point_alias = ParameterPoint(R=bp.R, T=bp.T, theta_f=float(theta), N=bp.N, depth=bp.depth, u=bp.u, ploidy_factor=bp.ploidy_factor, lam=bp.lam, condition_on_survival=bp.condition_on_survival)
                            meta_alias = {
                                "format": "tailbin_dense_z_cache_v1_0",
                                "representation": str(src_row.get("representation", "unknown")),
                                "certified": bool(src_row.get("certified", True)),
                                "status": "certified_alpha_threshold_alias",
                                "spec": {"point": point_alias.to_dict(), "alpha": float(alpha), "alpha_index": int(ai)},
                                "Kmax": int(build_cfg.Kmax),
                                "prefix_kmax": int(src_row.get("prefix_kmax", -1)),
                                "saturation_start_k": int(src_row.get("saturation_start_k", -1)),
                                "seconds": 0.0,
                                "build_config": asdict(build_cfg),
                                "error_budget": asdict(budget),
                                "storage_meta": {"storage": "alias", "source_row_index": int(alias_src)},
                                "total_z_error_indicator": float(src_row.get("total_z_error_indicator", 0.0)),
                                "total_cdf_error_indicator": float(src_row.get("total_cdf_error_indicator", 0.0)),
                                "alpha_threshold_alias": {"read_count_threshold": int(q_threshold), "source_row_index": int(alias_src)},
                                "guarantee_notes": ["Exact alias: ceil(alpha*depth) is identical to a previously built cutoff, so the read-sampled tail-count distribution is identical."],
                            }
                            new_idx = writer.append_alias(source_index=int(alias_src), meta=meta_alias, bundle_idx=int(base_idx) * 100000 + int(ai))
                            threshold_sources[(float(theta), int(q_threshold))] = int(new_idx)
                            n_threshold_aliases += 1
                            continue
                        src = right_saturation.get(float(theta))
                        point = ParameterPoint(R=bp.R, T=bp.T, theta_f=float(theta), N=bp.N, depth=bp.depth, u=bp.u, ploidy_factor=bp.ploidy_factor, lam=bp.lam, condition_on_survival=bp.condition_on_survival)
                        spec_for_meta = ThetaBundleSpec(base_point=bp, theta_values=[theta], alpha=float(alpha), alpha_index=int(ai))
                        if src is not None and int(src.get("saturation_start_k", 0)) <= 0:
                            pre = _right_constant_preflight_from_source(theta, source_alpha=float(src["source_alpha"]), source_alpha_index=int(src["source_alpha_index"]), source_bound=float(src.get("tail_bound", budget.clip_eps)), build=build_cfg, budget=budget)
                            meta, z_value, cdf_value = _constant_meta_from_preflight(spec_for_meta, point, build_cfg, budget, pre)
                            meta["status"] = "certified_alpha_monotone_constant_right"
                            row_idx = writer.append_constant(meta=meta, z_value=z_value, cdf_value=cdf_value, bundle_idx=int(base_idx) * 100000 + int(ai))
                            threshold_sources[(float(theta), int(q_threshold))] = int(row_idx)
                            n_prop_constants += 1
                        else:
                            unresolved_thetas.append(float(theta))

                    if unresolved_thetas:
                        spec_moment = ThetaBundleSpec(base_point=bp, theta_values=unresolved_thetas, alpha=float(alpha), alpha_index=int(ai))
                        native_preflight = moment_preflight(spec_moment, build_cfg, budget)
                        native_by_theta = {float(p["theta_f"]): p for p in native_preflight}
                        preflight: List[Dict[str, Any]] = []
                        for theta in unresolved_thetas:
                            theta = float(theta)
                            native = native_by_theta.get(theta)
                            if native is None:
                                continue
                            src = right_saturation.get(theta)
                            if src is None:
                                preflight.append(native)
                                continue
                            monotone = _right_prefix_preflight_from_source(
                                theta,
                                source_alpha=float(src["source_alpha"]),
                                source_alpha_index=int(src["source_alpha_index"]),
                                saturation_start_k=int(src.get("saturation_start_k", 0)),
                                source_bound=float(src.get("tail_bound", budget.clip_eps)),
                                build=build_cfg,
                                budget=budget,
                            )
                            if native.get("saturation_kind") in {"constant_right", "constant_left"}:
                                preflight.append(native)
                            elif native.get("saturation_kind") == "right_saturated_prefix":
                                if int(monotone.get("prefix_kmax", build_cfg.Kmax)) < int(native.get("prefix_kmax", build_cfg.Kmax)):
                                    preflight.append(monotone)
                                else:
                                    preflight.append(native)
                            else:
                                preflight.append(monotone)

                        spec = ThetaBundleSpec(base_point=bp, theta_values=[float(p["theta_f"]) for p in preflight], alpha=float(alpha), alpha_index=int(ai))
                        remaining_thetas: List[float] = []
                        forced_prefix_preflight: List[Dict[str, Any]] = []
                        for pre in preflight:
                            theta = float(pre["theta_f"])
                            point = ParameterPoint(R=bp.R, T=bp.T, theta_f=theta, N=bp.N, depth=bp.depth, u=bp.u, ploidy_factor=bp.ploidy_factor, lam=bp.lam, condition_on_survival=bp.condition_on_survival)
                            if pre["saturation_kind"] in {"constant_right", "constant_left"}:
                                meta, z_value, cdf_value = _constant_meta_from_preflight(spec, point, build_cfg, budget, pre)
                                row_idx = writer.append_constant(meta=meta, z_value=z_value, cdf_value=cdf_value, bundle_idx=int(base_idx) * 100000 + int(ai))
                                threshold_sources[(float(theta), int(q_threshold))] = int(row_idx)
                                n_moment_constants += 1
                                if pre["saturation_kind"] == "constant_right":
                                    right_saturation[theta] = {"source_alpha": float(alpha), "source_alpha_index": int(ai), "saturation_start_k": 0, "tail_bound": float(pre.get("tail_bound") or 0.0)}
                            elif pre.get("tail_certificate_method") == "alpha_monotone_right_saturation_prefix":
                                forced_prefix_preflight.append(pre)
                            else:
                                remaining_thetas.append(theta)
                        build_groups: List[Tuple[List[float], Optional[List[Dict[str, Any]]]]] = []
                        if remaining_thetas:
                            build_groups.append((remaining_thetas, None))
                        if forced_prefix_preflight:
                            build_groups.append(([float(p["theta_f"]) for p in forced_prefix_preflight], forced_prefix_preflight))
                        for build_thetas, pre_override in build_groups:
                            sub_spec = ThetaBundleSpec(base_point=bp, theta_values=build_thetas, alpha=float(alpha), alpha_index=int(ai))
                            import tempfile
                            with tempfile.TemporaryDirectory(prefix="tailbin_v09_") as tmp:
                                meta = build_bundle_with_ladder(
                                    sub_spec, build_cfg, budget, refinement_cfg, Path(tmp),
                                    preflight=pre_override, table_index_prefix=f"{base_idx:06d}_{ai:02d}_",
                                )
                                if not bool(meta.get("certified", False)) and not list(Path(tmp).glob("*.npz")):
                                    n_refinement_failures += 1
                                for p in sorted(Path(tmp).glob("*.npz")):
                                    arr, cdf, row_meta = _read_npz_payload(p)
                                    row_idx = writer.append_array(arr=arr, cdf=cdf, meta=row_meta, bundle_idx=int(base_idx) * 100000 + int(ai))
                                    n_nonconstant += 1
                                    n_nonconstant_cert += int(bool(row_meta.get("certified", False)))
                                    pre = (row_meta.get("preflight") or {})
                                    theta = float((row_meta.get("spec") or {}).get("point", {}).get("theta_f", math.nan))
                                    if math.isfinite(theta):
                                        threshold_sources[(float(theta), int(q_threshold))] = int(row_idx)
                                    if pre.get("saturation_kind") in {"right_saturated_prefix", "constant_right"}:
                                        start_k = int(pre.get("saturation_start_k", 0 if pre.get("saturation_kind") == "constant_right" else 999999999))
                                        old_src = right_saturation.get(theta)
                                        if old_src is None or start_k < int(old_src.get("saturation_start_k", 999999999)):
                                            right_saturation[theta] = {"source_alpha": float(alpha), "source_alpha_index": int(ai), "saturation_start_k": int(start_k), "tail_bound": float(pre.get("tail_bound") or 0.0)}

                    tables_cert = int(sum(1 for r in writer.rows if bool(r.get("certified", False))))
                    progress.emit(
                        "finish_alpha",
                        **base_payload,
                        alpha=float(alpha),
                        alpha_index=int(ai),
                        elapsed_seconds_alpha=float(time.perf_counter() - alpha_t0),
                        cumulative_elapsed_seconds=float(time.perf_counter() - t0),
                        tables_attempted=int(len(writer.rows)),
                        tables_written=int(len(writer.rows)),
                        tables_certified=int(tables_cert),
                        tables_written_this_alpha=int(len(writer.rows) - tables_before),
                        tables_certified_this_alpha=int(tables_cert - cert_before),
                        output_file_size_bytes=output_size(),
                    )
                    timeout_check("after_alpha", base_started=base_t0, base_idx=int(base_idx))

                row = {
                    "base_idx": int(base_idx),
                    "R": float(bp.R), "T": float(bp.T), "N": float(bp.N), "depth": int(bp.depth),
                    "n_theta": int(len(theta_values_all)),
                    "n_alphas": int(len(grid.alphas)),
                    "n_alpha_threshold_aliases": int(n_threshold_aliases),
                    "n_alpha_monotone_constants": int(n_prop_constants),
                    "n_moment_constants": int(n_moment_constants),
                    "n_nonconstant": int(n_nonconstant),
                    "n_nonconstant_certified": int(n_nonconstant_cert),
                    "n_refinement_failures": int(n_refinement_failures),
                    "seconds": float(time.perf_counter() - base_t0),
                    "pgf_backend": str(build_cfg.pgf_backend),
                }
                rows.append(row)
                progress.emit(
                    "finish_base_point",
                    **base_payload,
                    elapsed_seconds_base_point=float(row["seconds"]),
                    cumulative_elapsed_seconds=float(time.perf_counter() - t0),
                    tables_attempted=int(len(writer.rows)),
                    tables_written=int(len(writer.rows)),
                    tables_certified=int(sum(1 for r in writer.rows if bool(r.get("certified", False)))),
                    output_file_size_bytes=output_size(),
                    n_refinement_failures=int(n_refinement_failures),
                )
                timeout_check("after_base_point", base_started=base_t0, base_idx=int(base_idx))
    except Exception as exc:
        progress.emit(
            "error",
            stage="build_hdf5",
            backend=str(build_cfg.pgf_backend),
            cumulative_elapsed_seconds=float(time.perf_counter() - t0),
            output_file_size_bytes=output_size(),
            message=str(exc),
        )
        raise

    elapsed = float(time.perf_counter() - t0)
    with h5py.File(output_path, "r") as h5:
        n_tables = int(h5.attrs.get("n_tables", 0))
        n_cert = int(h5.attrs.get("n_certified", 0))
        reps = np.asarray(h5["manifest/representation"], dtype=str) if n_tables else np.array([], dtype=str)
        statuses = np.asarray(h5["manifest/status"], dtype=str) if n_tables else np.array([], dtype=str)
        z_err = np.asarray(h5["manifest/total_z_error_indicator"], dtype=float) if n_tables else np.array([], dtype=float)
        cdf_err = np.asarray(h5["manifest/total_cdf_error_indicator"], dtype=float) if n_tables else np.array([], dtype=float)
        n_constant = int(np.sum(reps == "constant")) if n_tables else 0
        n_prop = int(np.sum(statuses == "certified_alpha_monotone_constant_right")) if n_tables else 0
        n_alias = int(np.sum(statuses == "certified_alpha_threshold_alias")) if n_tables else 0
        size_bytes = int(Path(output_path).stat().st_size)
    seconds_per_base_point = [float(r["seconds"]) for r in rows]
    expected_tables_for_attempted = int(
        sum(len(grid.valid_theta_values_for_T(bp.T)) * len(grid.alphas) for _base_idx, bp in base_points)
    )
    summary = {
        "format": "tailbin_alpha_monotone_hdf5_build_summary_v1_0",
        "config_path": str(config_path),
        "output_path": str(output_path),
        "grid": grid.to_dict(),
        "build_config": asdict(build_cfg),
        "requested_pgf_backend": requested_backend,
        "actual_build_backend": str(build_cfg.pgf_backend),
        "progress_path": None if progress_path is None else str(progress_path),
        "timeout_seconds": None if timeout_seconds is None else float(timeout_seconds),
        "max_seconds_per_base_point_threshold": None if max_seconds_per_base_point is None else float(max_seconds_per_base_point),
        "error_budget": asdict(budget),
        "refinement_config": asdict(refinement_cfg),
        "sharding": {"mode": "base_point_modulo", "n_shards": int(n_shards), "shard_index": int(shard_index)},
        "base_point_manifest": None if base_point_manifest is None else str(base_point_manifest),
        "n_base_points_selected_by_manifest": None if selected_base_indices is None else int(len(selected_base_indices)),
        "n_base_points_expected_total": int(len(all_base)),
        "n_base_points_attempted": int(len(base_points)),
        "n_tables_attempted": int(expected_tables_for_attempted),
        "n_tables_written": n_tables,
        "n_tables_certified": n_cert,
        "certified_fraction_written": float(n_cert / n_tables) if n_tables else 0.0,
        "n_constant_tables": n_constant,
        "n_alpha_threshold_aliases": n_alias,
        "n_alpha_monotone_propagated_constants": n_prop,
        "n_nonconstant_tables": int(n_tables - n_constant),
        "n_refinement_failures": int(sum(int(r.get("n_refinement_failures", 0)) for r in rows)),
        "elapsed_seconds": elapsed,
        "mean_seconds_per_base_point": float(sum(seconds_per_base_point) / len(seconds_per_base_point)) if seconds_per_base_point else None,
        "median_seconds_per_base_point": float(np.median(seconds_per_base_point)) if seconds_per_base_point else None,
        "max_seconds_per_base_point": float(max(seconds_per_base_point)) if seconds_per_base_point else None,
        "max_total_z_error_indicator": float(np.nanmax(z_err)) if z_err.size else None,
        "max_total_cdf_error_indicator": float(np.nanmax(cdf_err)) if cdf_err.size else None,
        "output_bytes": size_bytes,
        "output_mb": float(size_bytes / 1e6),
        "base_rows": rows,
        "notes": [
            "The builder scans all alphas for a base point in increasing cutoff order.",
            "If two alphas have the same read-count threshold ceil(alpha*depth), the later one is written as an exact alias to the first table.",
            "If a lower alpha has certified P(Y>=K)<=clip_eps, all higher alphas for the same theta inherit the same right-saturated suffix; only the prefix k<K-1 is built.",
            "Both optimizations are exact for cumulative tail counts, not approximations.",
        ],
    }
    progress.emit(
        "finish_build",
        stage="build_hdf5",
        backend=str(build_cfg.pgf_backend),
        cumulative_elapsed_seconds=float(elapsed),
        tables_attempted=int(expected_tables_for_attempted),
        tables_written=int(n_tables),
        tables_certified=int(n_cert),
        output_file_size_bytes=int(size_bytes),
    )
    Path(output_path).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    if rows:
        with Path(output_path).with_suffix(".basepoints.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return summary
