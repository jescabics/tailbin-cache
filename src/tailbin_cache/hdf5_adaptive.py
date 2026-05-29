from __future__ import annotations

"""Direct adaptive HDF5 cache builder and reader.

An older workflow produced many small NPZ files and then packed them into HDF5.  That is
fine for smoke tests but operationally poor for large grids, especially because
most moment-saturated tables are constants and do not need a per-table file at
all.  This module writes a compact sharded HDF5 cache directly:

* constant clipped tables are represented by manifest rows only;
* prefix/full tables are stored in flat compressed int16/float32 arrays with
  per-row offsets and lengths;
* shard_index/n_shards lets users run 20 independent processes safely, each
  writing its own HDF5 file.

The mathematical target is unchanged from the adaptive builder: moment-certified
saturation where possible and damped Cauchy/FFT coefficient extraction for the
remaining prefix/full rows.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv
import json
import math
import tempfile
import time

import h5py
import numpy as np

from .adaptive_bundle import moment_preflight, AdaptiveThetaBundleBuilder
from .builder import BuildConfig, ErrorBudget, dequantize_z
from .grid import CacheGrid, ParameterPoint, grid_from_dict
from .runner import build_config_from_dict, config_from_yaml
from .theta_bundle import ThetaBundleSpec


def _unique_base_points(grid: CacheGrid) -> List[ParameterPoint]:
    base_points: List[ParameterPoint] = []
    seen = set()
    for p in grid.parameter_points():
        key = (float(p.R), float(p.T), float(p.N), int(p.depth), float(p.u), float(p.ploidy_factor), float(p.lam), bool(p.condition_on_survival))
        if key not in seen:
            seen.add(key)
            base_points.append(ParameterPoint(R=p.R, T=p.T, theta_f=0.0, N=p.N, depth=p.depth, u=p.u, ploidy_factor=p.ploidy_factor, lam=p.lam, condition_on_survival=p.condition_on_survival))
    return base_points


def adaptive_bundle_specs(grid: CacheGrid) -> List[Tuple[int, ThetaBundleSpec]]:
    base_points = _unique_base_points(grid)
    bundles: List[Tuple[int, ThetaBundleSpec]] = []
    idx = 0
    for bp in base_points:
        theta_values = grid.valid_theta_values_for_T(bp.T)
        if not theta_values:
            continue
        for ai, alpha in enumerate(grid.alphas):
            bundles.append((idx, ThetaBundleSpec(base_point=bp, theta_values=theta_values, alpha=float(alpha), alpha_index=int(ai))))
            idx += 1
    return bundles


def _read_npz_payload(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        meta_raw = z["metadata"]
        if hasattr(meta_raw, "item"):
            meta_raw = meta_raw.item()
        meta = json.loads(str(meta_raw))
        if "z_prefix" in z.files:
            arr = np.asarray(z["z_prefix"])
        else:
            arr = np.asarray(z["z_values"])
        cdf = np.asarray(z["cdf_prefix"]) if "cdf_prefix" in z.files else (np.asarray(z["cdf_values"]) if "cdf_values" in z.files else None)
    return arr, cdf, meta


class StreamingHDF5Writer:
    """Append adaptive z-cache rows to a flat-array HDF5 file."""

    def __init__(self, path: str | Path, *, compression: str = "gzip", compression_opts: int = 4):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.path, "w")
        self.h5.attrs["format"] = "tailbin_adaptive_cache_hdf5_v1_0"
        self.h5.attrs["created_seconds_unix"] = float(time.time())
        self.compression = compression
        self.compression_opts = int(compression_opts)
        self.z_i2 = self.h5.create_dataset(
            "z_int16_flat", shape=(0,), maxshape=(None,), chunks=(max(1024, 65536),),
            dtype=np.int16, compression=compression, compression_opts=compression_opts,
        )
        self.z_f4 = self.h5.create_dataset(
            "z_float32_flat", shape=(0,), maxshape=(None,), chunks=(max(1024, 65536),),
            dtype=np.float32, compression=compression, compression_opts=compression_opts,
        )
        self.cdf_f4 = self.h5.create_dataset(
            "cdf_float32_flat", shape=(0,), maxshape=(None,), chunks=(max(1024, 65536),),
            dtype=np.float32, compression=compression, compression_opts=compression_opts,
        )
        self.rows: List[Dict[str, Any]] = []
        self.metadata_json: List[str] = []

    def _append_flat(self, ds: h5py.Dataset, arr: np.ndarray) -> Tuple[int, int]:
        arr = np.asarray(arr)
        if arr.size == 0:
            return -1, 0
        offset = int(ds.shape[0])
        ds.resize((offset + int(arr.size),))
        ds[offset: offset + int(arr.size)] = arr.reshape(-1)
        return offset, int(arr.size)

    def append_constant(self, *, meta: Dict[str, Any], z_value: float, cdf_value: float, bundle_idx: int) -> int:
        row = self._row_from_meta(meta, bundle_idx=bundle_idx)
        row.update({
            "dtype": "constant",
            "z_offset": -1,
            "cdf_offset": -1,
            "length": 0,
            "cdf_length": 0,
            "constant_z": float(z_value),
            "constant_cdf": float(cdf_value),
            "quant_scale": math.nan,
        })
        self.rows.append(row)
        self.metadata_json.append(json.dumps(meta, sort_keys=True))
        return len(self.rows) - 1

    def append_array(self, *, arr: np.ndarray, cdf: Optional[np.ndarray], meta: Dict[str, Any], bundle_idx: int) -> int:
        arr = np.asarray(arr)
        sm = meta.get("storage_meta", {}) or {}
        if arr.dtype == np.int16:
            z_offset, length = self._append_flat(self.z_i2, arr.astype(np.int16, copy=False))
            dtype = "int16"
            quant_scale = float(sm.get("quant_scale", math.nan))
        else:
            z_offset, length = self._append_flat(self.z_f4, arr.astype(np.float32, copy=False))
            dtype = "float32"
            quant_scale = math.nan
        if cdf is not None:
            cdf_offset, cdf_len = self._append_flat(self.cdf_f4, np.asarray(cdf, dtype=np.float32))
        else:
            cdf_offset, cdf_len = -1, 0
        row = self._row_from_meta(meta, bundle_idx=bundle_idx)
        row.update({
            "dtype": dtype,
            "z_offset": int(z_offset),
            "cdf_offset": int(cdf_offset),
            "length": int(length),
            "cdf_length": int(cdf_len),
            "constant_z": math.nan,
            "constant_cdf": math.nan,
            "quant_scale": quant_scale,
        })
        self.rows.append(row)
        self.metadata_json.append(json.dumps(meta, sort_keys=True))
        return len(self.rows) - 1


    def append_alias(self, *, source_index: int, meta: Dict[str, Any], bundle_idx: int) -> int:
        """Append a manifest row that reuses an existing table's stored data.

        This is used for exact read-depth threshold aliases: if two alpha cutoffs
        have the same integer threshold ceil(alpha*depth), their read-sampled
        count distributions are identical.  We therefore add a metadata row with
        the new alpha/alpha_index but point to the same flat-array offsets.  No
        data are duplicated and no PGF/FFT build is repeated.
        """
        src = dict(self.rows[int(source_index)])
        row = self._row_from_meta(meta, bundle_idx=bundle_idx)
        for key in ["dtype", "z_offset", "cdf_offset", "length", "cdf_length", "constant_z", "constant_cdf", "quant_scale"]:
            row[key] = src.get(key)
        self.rows.append(row)
        self.metadata_json.append(json.dumps(meta, sort_keys=True))
        return len(self.rows) - 1

    def _row_from_meta(self, meta: Dict[str, Any], *, bundle_idx: int) -> Dict[str, Any]:
        spec = meta.get("spec") or {}
        point = spec.get("point") or ((spec.get("base_point") or {}))
        return {
            "bundle_idx": int(bundle_idx),
            "certified": bool(meta.get("certified", False)),
            "status": str(meta.get("status", "")),
            "representation": str(meta.get("representation", "unknown")),
            "Kmax": int(meta.get("Kmax", -1)),
            "prefix_kmax": int(meta.get("prefix_kmax", -1)) if meta.get("prefix_kmax", None) is not None else -1,
            "saturation_start_k": int(meta.get("saturation_start_k", -1)) if meta.get("saturation_start_k", None) is not None else -1,
            "R": float(point.get("R", math.nan)),
            "T": float(point.get("T", math.nan)),
            "theta_f": float(point.get("theta_f", math.nan)),
            "N": float(point.get("N", math.nan)),
            "depth": int(point.get("depth", -1)),
            "u": float(point.get("u", math.nan)),
            "alpha": float(spec.get("alpha", math.nan)),
            "alpha_index": int(spec.get("alpha_index", -1)),
            "seconds": float(meta.get("seconds", 0.0)),
            "total_z_error_indicator": float(meta.get("total_z_error_indicator", math.nan)),
            "total_cdf_error_indicator": float(meta.get("total_cdf_error_indicator", math.nan)),
        }

    def close(self) -> None:
        str_dt = h5py.string_dtype(encoding="utf-8")
        grp = self.h5.create_group("manifest")
        if not self.rows:
            self.h5.attrs["n_tables"] = 0
            self.h5.close()
            return
        keys = list(self.rows[0].keys())
        string_keys = {"status", "representation", "dtype"}
        bool_keys = {"certified"}
        int_keys = {"bundle_idx", "Kmax", "prefix_kmax", "saturation_start_k", "depth", "alpha_index", "z_offset", "cdf_offset", "length", "cdf_length"}
        for key in keys:
            vals = [r.get(key) for r in self.rows]
            if key in string_keys:
                grp.create_dataset(key, data=np.array([str(v) for v in vals], dtype=object), dtype=str_dt)
            elif key in bool_keys:
                grp.create_dataset(key, data=np.array([bool(v) for v in vals], dtype=np.bool_))
            elif key in int_keys:
                grp.create_dataset(key, data=np.array([int(v) for v in vals], dtype=np.int64))
            else:
                grp.create_dataset(key, data=np.array([float(v) for v in vals], dtype=np.float64))
        meta_grp = self.h5.create_group("metadata_json")
        meta_grp.create_dataset("json", data=np.array(self.metadata_json, dtype=object), dtype=str_dt)
        self.h5.attrs["n_tables"] = int(len(self.rows))
        self.h5.attrs["n_certified"] = int(sum(1 for r in self.rows if r.get("certified")))
        self.h5.flush()
        self.h5.close()

    def __enter__(self) -> "StreamingHDF5Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class HDF5AdaptiveCache:
    """Read z/CDF values from a adaptive HDF5 cache."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.h5 = h5py.File(self.path, "r")
        if str(self.h5.attrs.get("format", "")) != "tailbin_adaptive_cache_hdf5_v1_0":
            raise ValueError(f"not a adaptive cache: {self.path}")
        self.manifest = self.h5["manifest"]

    def __len__(self) -> int:
        return int(self.h5.attrs.get("n_tables", len(self.manifest["Kmax"])))

    def metadata(self, index: int) -> Dict[str, Any]:
        raw = self.h5["metadata_json/json"][int(index)]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))

    def _row(self, index: int) -> Dict[str, Any]:
        i = int(index)
        out: Dict[str, Any] = {}
        for key, ds in self.manifest.items():
            val = ds[i]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            if isinstance(val, np.generic):
                val = val.item()
            out[key] = val
        return out

    def z(self, index: int, k: int) -> float:
        row = self._row(index)
        k = int(k)
        if k < 0 or k > int(row["Kmax"]):
            raise IndexError(f"k={k} outside table range 0..{int(row['Kmax'])}")
        rep = str(row["representation"])
        if rep == "constant":
            return float(row["constant_z"])
        if rep == "right_saturated_prefix" and k > int(row["prefix_kmax"]):
            meta = self.metadata(index)
            return float((meta.get("error_budget") or {}).get("z_clip", 7.05))
        if k > int(row["prefix_kmax"]):
            raise IndexError(f"k={k} exceeds stored prefix {int(row['prefix_kmax'])}")
        off = int(row["z_offset"])
        if str(row["dtype"]) == "int16":
            q = self.h5["z_int16_flat"][off + k]
            return float(q) / float(row["quant_scale"])
        return float(self.h5["z_float32_flat"][off + k])

    def cdf(self, index: int, k: int) -> float:
        row = self._row(index)
        k = int(k)
        if k < 0 or k > int(row["Kmax"]):
            raise IndexError(f"k={k} outside table range 0..{int(row['Kmax'])}")
        rep = str(row["representation"])
        if rep == "constant":
            return float(row["constant_cdf"])
        if rep == "right_saturated_prefix" and k > int(row["prefix_kmax"]):
            meta = self.metadata(index)
            return float(1.0 - float((meta.get("error_budget") or {}).get("clip_eps", 1e-12)))
        cdf_off = int(row.get("cdf_offset", -1))
        if cdf_off >= 0:
            return float(self.h5["cdf_float32_flat"][cdf_off + k])
        # Fallback via Gaussian CDF only if prefix CDF not stored.
        from scipy.stats import norm
        return float(norm.cdf(self.z(index, k)))

    def close(self) -> None:
        self.h5.close()


def _constant_meta_from_preflight(spec: ThetaBundleSpec, point: ParameterPoint, build: BuildConfig, budget: ErrorBudget, pre: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
    if pre["saturation_kind"] == "constant_right":
        z_value = float(budget.z_clip)
        cdf_value = float(1.0 - budget.clip_eps)
        status = "certified_moment_saturated_constant_right"
    elif pre["saturation_kind"] == "constant_left":
        z_value = -float(budget.z_clip)
        cdf_value = float(budget.clip_eps)
        status = "certified_moment_saturated_constant_left"
    else:
        raise ValueError("preflight row is not constant")
    meta = {
        "format": "tailbin_dense_z_cache_v1_0",
        "representation": "constant",
        "certified": True,
        "status": status,
        "spec": {"point": point.to_dict(), "alpha": float(spec.alpha), "alpha_index": int(spec.alpha_index)},
        "Kmax": int(build.Kmax),
        "prefix_kmax": -1,
        "saturation_start_k": 0,
        "seconds": 0.0,
        "build_config": asdict(build),
        "error_budget": asdict(budget),
        "preflight": pre,
        "storage_meta": {"storage": "constant", "constant_z": float(z_value), "constant_cdf": float(cdf_value)},
        "total_z_error_indicator": 0.0,
        "total_cdf_error_indicator": float(pre.get("tail_bound") or 0.0),
        "guarantee_notes": [
            "Stored directly in HDF5 manifest; no per-table NPZ file exists.",
            "The clipped z value is certified for every k in 0..Kmax by a one-sided moment bound.",
        ],
    }
    return meta, z_value, cdf_value


def build_adaptive_hdf5_from_config(
    config_path: str | Path,
    output_path: str | Path,
    *,
    limit_bundles: Optional[int] = None,
    n_shards: int = 1,
    shard_index: int = 0,
    compression: str = "gzip",
    compression_opts: int = 4,
    shard_mode: str = "modulo",
    shard_plan: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Build a direct adaptive HDF5 cache.

    For large runs, launch independent shard processes, for example:
        for i in {0..19}: tailbin-cache build-adaptive-hdf5 --n-shards 20 --shard-index i --output cache_shard_${i}.h5
    """
    cfg = config_from_yaml(config_path)
    grid = grid_from_dict(cfg)
    build_cfg, budget, _n_jobs = build_config_from_dict(cfg)
    plan_t0 = time.perf_counter()
    all_bundles = adaptive_bundle_specs(grid)
    if int(n_shards) < 1:
        raise ValueError("n_shards must be >=1")
    if not (0 <= int(shard_index) < int(n_shards)):
        raise ValueError("shard_index must satisfy 0 <= shard_index < n_shards")
    if shard_mode == "modulo":
        bundles = [(i, s) for i, s in all_bundles if int(i) % int(n_shards) == int(shard_index)]
    elif shard_mode == "balanced":
        from .shards import balanced_shard_plan, bundle_indices_for_shard, read_balanced_shard_plan
        if shard_plan is not None:
            plan = read_balanced_shard_plan(shard_plan)
        else:
            plan = balanced_shard_plan(grid, build_cfg, budget, n_shards=int(n_shards), limit_bundles=limit_bundles)
        keep = bundle_indices_for_shard(plan, int(shard_index))
        bundles = [(i, s) for i, s in all_bundles if int(i) in keep]
    else:
        raise ValueError("shard_mode must be 'modulo' or 'balanced'")
    if limit_bundles is not None and shard_mode == "modulo":
        bundles = bundles[: int(limit_bundles)]
    planning_seconds = float(time.perf_counter() - plan_t0)
    t0 = time.perf_counter()
    bundle_rows: List[Dict[str, Any]] = []
    with StreamingHDF5Writer(output_path, compression=compression, compression_opts=int(compression_opts)) as writer:
        for bundle_idx, spec in bundles:
            b0 = time.perf_counter()
            preflight = moment_preflight(spec, build_cfg, budget)
            constant_count = 0
            remaining_thetas: List[float] = []
            for pre in preflight:
                theta = float(pre["theta_f"])
                point = ParameterPoint(
                    R=spec.base_point.R, T=spec.base_point.T, theta_f=theta, N=spec.base_point.N,
                    depth=spec.base_point.depth, u=spec.base_point.u, ploidy_factor=spec.base_point.ploidy_factor, lam=spec.base_point.lam,
                    condition_on_survival=spec.base_point.condition_on_survival,
                )
                if pre["saturation_kind"] in {"constant_right", "constant_left"}:
                    meta, z_value, cdf_value = _constant_meta_from_preflight(spec, point, build_cfg, budget, pre)
                    writer.append_constant(meta=meta, z_value=z_value, cdf_value=cdf_value, bundle_idx=bundle_idx)
                    constant_count += 1
                else:
                    remaining_thetas.append(theta)
            nonconst_tables = 0
            nonconst_cert = 0
            if remaining_thetas:
                sub_spec = ThetaBundleSpec(base_point=spec.base_point, theta_values=remaining_thetas, alpha=spec.alpha, alpha_index=spec.alpha_index)
                with tempfile.TemporaryDirectory(prefix="tailbin_v05_") as tmp:
                    meta = AdaptiveThetaBundleBuilder(sub_spec, build_cfg, budget).build_bundle(Path(tmp), table_index_prefix=f"{bundle_idx:06d}_")
                    for p in sorted(Path(tmp).glob("*.npz")):
                        arr, cdf, row_meta = _read_npz_payload(p)
                        writer.append_array(arr=arr, cdf=cdf, meta=row_meta, bundle_idx=bundle_idx)
                        nonconst_tables += 1
                        nonconst_cert += int(bool(row_meta.get("certified", False)))
            bundle_rows.append({
                "bundle_idx": int(bundle_idx),
                "R": float(spec.base_point.R),
                "T": float(spec.base_point.T),
                "N": float(spec.base_point.N),
                "depth": int(spec.base_point.depth),
                "alpha": float(spec.alpha),
                "alpha_index": int(spec.alpha_index),
                "n_theta": int(len(spec.theta_values)),
                "n_constant": int(constant_count),
                "n_nonconstant": int(nonconst_tables),
                "n_nonconstant_certified": int(nonconst_cert),
                "seconds": float(time.perf_counter() - b0),
            })
    elapsed = float(time.perf_counter() - t0)
    # Read h5 attrs after writer closes.
    with h5py.File(output_path, "r") as h5:
        n_tables = int(h5.attrs.get("n_tables", 0))
        n_cert = int(h5.attrs.get("n_certified", 0))
        n_constant = int(np.sum(np.asarray(h5["manifest/representation"], dtype=str) == "constant")) if n_tables else 0
        size_bytes = int(Path(output_path).stat().st_size)
    summary = {
        "format": "tailbin_adaptive_hdf5_build_summary_v1_0",
        "config_path": str(config_path),
        "output_path": str(output_path),
        "grid": grid.to_dict(),
        "build_config": asdict(build_cfg),
        "error_budget": asdict(budget),
        "sharding": {"n_shards": int(n_shards), "shard_index": int(shard_index), "shard_mode": str(shard_mode)},
        "n_bundles_expected_total": int(len(all_bundles)),
        "n_bundles_attempted": int(len(bundles)),
        "n_tables_written": n_tables,
        "n_tables_certified": n_cert,
        "certified_fraction_written": float(n_cert / n_tables) if n_tables else 0.0,
        "n_constant_tables": n_constant,
        "n_nonconstant_tables": int(n_tables - n_constant),
        "elapsed_seconds": elapsed,
        "planning_seconds": planning_seconds,
        "total_wall_seconds_including_planning": elapsed + planning_seconds,
        "mean_seconds_per_bundle": float(sum(r["seconds"] for r in bundle_rows) / len(bundle_rows)) if bundle_rows else None,
        "output_bytes": size_bytes,
        "output_mb": float(size_bytes / 1e6),
        "bundle_rows": bundle_rows,
        "notes": "The direct HDF5 builder writes moment-saturated constant tables directly to HDF5 manifest, enforces T+Tb<=max_age, uses diploid frequency m/(2N) by default, and supports modulo or difficulty-balanced sharding.",
    }
    out_summary = Path(output_path).with_suffix(".summary.json")
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    out_csv = Path(output_path).with_suffix(".bundles.csv")
    if bundle_rows:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bundle_rows[0].keys()))
            w.writeheader(); w.writerows(bundle_rows)
    return summary
