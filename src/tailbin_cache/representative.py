from __future__ import annotations

"""Deterministic representative base-point selection for O2 calibration."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import csv
import json
import math

from .grid import grid_from_dict
from .hdf5_adaptive import _unique_base_points
from .runner import config_from_yaml


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_plan_bundle_rows(plan_dir: str | Path) -> List[Dict[str, Any]]:
    path = Path(plan_dir)
    if path.is_dir():
        path = path / "plan_bundles.csv"
    if not path.exists():
        raise FileNotFoundError(f"plan bundle CSV not found: {path}")
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def aggregate_base_points(config_path: str | Path, plan_dir: str | Path) -> List[Dict[str, Any]]:
    cfg = config_from_yaml(config_path)
    grid = grid_from_dict(cfg)
    base_points = _unique_base_points(grid)
    n_alpha = max(1, len(grid.alphas))
    groups: Dict[int, Dict[str, Any]] = {}
    for row in read_plan_bundle_rows(plan_dir):
        bundle_idx = _to_int(row.get("bundle_idx"))
        base_idx = bundle_idx // n_alpha
        if base_idx < 0 or base_idx >= len(base_points):
            continue
        bp = base_points[base_idx]
        theta_values = grid.valid_theta_values_for_T(bp.T)
        Tb_values = [float(theta) / float(bp.u) if float(bp.u) != 0.0 else math.inf for theta in theta_values]
        g = groups.setdefault(
            base_idx,
            {
                "base_idx": int(base_idx),
                "R": float(bp.R),
                "N": float(bp.N),
                "T": float(bp.T),
                "Tb": float(Tb_values[0]) if len(Tb_values) == 1 else None,
                "theta_f": float(theta_values[0]) if len(theta_values) == 1 else None,
                "Tb_values": Tb_values,
                "theta_values": theta_values,
                "depth": int(bp.depth),
                "n_theta": int(len(theta_values)),
                "n_bundles": 0,
                "n_constant_tables": 0,
                "n_prefix_tables": 0,
                "n_full_tables": 0,
                "max_prefix_kmax": -1,
                "total_v03_node_work_proxy": 0,
                "total_v04_node_work_proxy": 0,
                "max_v04_node_work_proxy": 0,
            },
        )
        n_theta = _to_int(row.get("n_theta"), int(g["n_theta"]))
        n_constant = _to_int(row.get("n_constant"))
        n_prefix = _to_int(row.get("n_prefix"))
        n_full = _to_int(row.get("n_full"))
        v03 = _to_int(row.get("v03_node_work_proxy"))
        v04 = _to_int(row.get("v04_node_work_proxy"))
        g["n_bundles"] = int(g["n_bundles"]) + 1
        g["n_constant_tables"] = int(g["n_constant_tables"]) + n_constant
        g["n_prefix_tables"] = int(g["n_prefix_tables"]) + n_prefix
        g["n_full_tables"] = int(g["n_full_tables"]) + n_full
        g["max_prefix_kmax"] = max(int(g["max_prefix_kmax"]), _to_int(row.get("max_prefix_kmax"), -1))
        g["total_v03_node_work_proxy"] = int(g["total_v03_node_work_proxy"]) + v03
        g["total_v04_node_work_proxy"] = int(g["total_v04_node_work_proxy"]) + v04
        g["max_v04_node_work_proxy"] = max(int(g["max_v04_node_work_proxy"]), v04)
        g["n_tables_planned"] = int(g["n_bundles"]) * n_theta
    return [groups[i] for i in sorted(groups)]


def _quantile_positions(n: int, quantiles: Iterable[float]) -> List[int]:
    if n <= 0:
        return []
    out = []
    for q in quantiles:
        pos = int(round(max(0.0, min(1.0, float(q))) * float(n - 1)))
        if pos not in out:
            out.append(pos)
    return out


def select_representative_base_points(
    points: List[Dict[str, Any]],
    *,
    sample_size: int = 40,
    strategy: str = "representative_hard",
) -> List[Dict[str, Any]]:
    if not points:
        return []
    target = min(int(sample_size), len(points))
    selected: Dict[int, set[str]] = {}

    def add(point: Dict[str, Any], reason: str) -> None:
        idx = int(point["base_idx"])
        if idx in selected:
            selected[idx].add(reason)
        elif len(selected) < target:
            selected[idx] = {reason}

    def add_positions(label: str, candidates: List[Dict[str, Any]], positions: Iterable[int]) -> None:
        if not candidates:
            return
        for pos in positions:
            pos = max(0, min(len(candidates) - 1, int(pos)))
            add(candidates[pos], label)

    by_work = sorted(points, key=lambda p: (float(p.get("total_v04_node_work_proxy", 0)), int(p["base_idx"])))
    by_hard = list(reversed(by_work))
    by_full = sorted(points, key=lambda p: (int(p.get("n_full_tables", 0)), int(p.get("max_prefix_kmax", -1)), float(p.get("total_v04_node_work_proxy", 0)), -int(p["base_idx"])), reverse=True)

    # Reserve scientific boundaries before filling by work proxy so a modest
    # sample cannot accidentally omit the age-diagonal endpoints.
    for field in ("R", "N", "Tb"):
        values = sorted({float(p[field]) for p in points if p.get(field) is not None})
        if not values:
            continue
        for value, suffix in ((values[0], "low"), (values[-1], "high")):
            subset = [p for p in points if p.get(field) is not None and abs(float(p[field]) - value) <= 1e-10]
            subset = sorted(subset, key=lambda p: (float(p.get("total_v04_node_work_proxy", 0)), int(p["base_idx"])))
            add_positions(f"{field}_{suffix}_boundary", subset, _quantile_positions(len(subset), [0.0, 0.5, 1.0]))

    tb_values = sorted({float(p["Tb"]) for p in points if p.get("Tb") is not None})
    for pos in _quantile_positions(len(tb_values), [0.25, 0.50, 0.75]):
        value = tb_values[pos]
        subset = [p for p in points if p.get("Tb") is not None and abs(float(p["Tb"]) - value) <= 1e-10]
        subset = sorted(subset, key=lambda p: (float(p.get("total_v04_node_work_proxy", 0)), int(p["base_idx"])))
        add_positions("Tb_intermediate", subset, _quantile_positions(len(subset), [0.5, 1.0]))

    # Explicit high-risk planner predictions.
    add_positions("planner_full_or_large_prefix", by_full, range(max(4, target // 6)))

    # Work tiers: easy, middle, and hard predicted rows.
    add_positions("work_proxy_easiest", by_work, range(max(3, target // 8)))
    add_positions("work_proxy_median", by_work, _quantile_positions(len(by_work), [0.45, 0.50, 0.55]))
    add_positions("work_proxy_hardest", by_hard, range(max(5, target // 4)))

    # Fill any remaining slots with a stable spread across work quantiles, then
    # by hardest points. This keeps the sample deterministic and avoids taking
    # the first N repository-order rows.
    for pos in _quantile_positions(len(by_work), [i / max(1, target - 1) for i in range(target)]):
        if len(selected) >= target:
            break
        add(by_work[pos], "work_quantile_spread")
    for point in by_hard:
        if len(selected) >= target:
            break
        add(point, "hard_fill")

    selected_rows = []
    for rank, idx in enumerate(sorted(selected)):
        point = dict(next(p for p in points if int(p["base_idx"]) == idx))
        point["selection_rank"] = int(rank)
        point["selection_reasons"] = ";".join(sorted(selected[idx]))
        point["selection_strategy"] = str(strategy)
        selected_rows.append(point)
    return selected_rows


def write_representative_manifest(
    *,
    config_path: str | Path,
    plan_dir: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
    sample_size: int = 40,
    strategy: str = "representative_hard",
) -> Dict[str, Any]:
    points = aggregate_base_points(config_path, plan_dir)
    selected = select_representative_base_points(points, sample_size=sample_size, strategy=strategy)
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selection_rank",
        "base_idx",
        "R",
        "N",
        "T",
        "Tb",
        "theta_f",
        "depth",
        "n_theta",
        "n_bundles",
        "n_tables_planned",
        "n_constant_tables",
        "n_prefix_tables",
        "n_full_tables",
        "max_prefix_kmax",
        "total_v03_node_work_proxy",
        "total_v04_node_work_proxy",
        "max_v04_node_work_proxy",
        "selection_reasons",
        "selection_strategy",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row.get(k) for k in fieldnames})
    summary = {
        "format": "tailbin_representative_base_point_manifest_v1_0",
        "config_path": str(config_path),
        "plan_dir": str(plan_dir),
        "strategy": str(strategy),
        "sample_size_requested": int(sample_size),
        "sample_size_selected": int(len(selected)),
        "candidate_base_points": int(len(points)),
        "output_csv": str(csv_path),
        "output_json": str(json_path),
        "selected_base_points": selected,
        "notes": [
            "Selection is deterministic and based on full-plan work proxies plus scientific-axis boundaries.",
            "The manifest identifies explicit base_idx values for build-hdf5 --base-point-manifest.",
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
