#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from summarize_resource_calibration import (
    load_json_files,
    parse_accounting,
    parse_elapsed_seconds,
    parse_gpu_monitor_logs,
    parse_slurm_metadata,
    parse_time_file,
)


def quantile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    pos = max(0.0, min(1.0, q)) * float(len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception as exc:  # noqa: BLE001
        return {"parse_error": str(exc), "path": str(path)}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def classify_points(manifest_rows: list[dict[str, Any]], build_summary: dict[str, Any]) -> list[dict[str, Any]]:
    base_rows = build_summary.get("base_rows") or []
    by_idx = {int(row.get("base_idx")): row for row in base_rows if row.get("base_idx") is not None}
    seconds = [float(row.get("seconds", 0.0) or 0.0) for row in base_rows]
    q50 = quantile(seconds, 0.50)
    q75 = quantile(seconds, 0.75)
    q90 = quantile(seconds, 0.90)
    work_values = [float(row.get("total_v04_node_work_proxy", 0.0) or 0.0) for row in manifest_rows]
    w75 = quantile(work_values, 0.75)
    w90 = quantile(work_values, 0.90)

    out: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        idx = int(float(manifest["base_idx"]))
        build = by_idx.get(idx, {})
        sec = float(build.get("seconds", 0.0) or 0.0)
        failures = int(build.get("n_refinement_failures", 0) or 0)
        work = float(manifest.get("total_v04_node_work_proxy", 0.0) or 0.0)
        if failures > 0:
            label = "failed_deferred_problematic"
            reason = "refinement_failures"
        elif q90 is not None and sec >= q90:
            label = "hard"
            reason = "observed_seconds_ge_q90"
        elif q75 is not None and sec >= q75:
            label = "hard"
            reason = "observed_seconds_ge_q75"
        elif q50 is not None and sec >= q50:
            label = "moderate"
            reason = "observed_seconds_ge_q50"
        elif w90 is not None and work >= w90:
            label = "hard"
            reason = "planner_work_ge_q90"
        elif w75 is not None and work >= w75:
            label = "moderate"
            reason = "planner_work_ge_q75"
        else:
            label = "easy"
            reason = "low_observed_seconds_or_work"
        row = {
            "base_idx": idx,
            "classification": label,
            "classification_reason": reason,
            "seconds": sec if build else None,
            "n_refinement_failures": failures if build else None,
            "n_nonconstant": build.get("n_nonconstant"),
            "n_nonconstant_certified": build.get("n_nonconstant_certified"),
            "total_v04_node_work_proxy": work,
            "selection_reasons": manifest.get("selection_reasons", ""),
            "R": manifest.get("R"),
            "N": manifest.get("N"),
            "T": manifest.get("T"),
            "Tb": manifest.get("Tb"),
            "theta_f": manifest.get("theta_f"),
            "depth": manifest.get("depth"),
        }
        out.append(row)
    return out


def write_classification_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Tailbin Representative Grid B Calibration Summary")
    lines.append("")
    lines.append(f"Run ID: `{summary['run_id']}`")
    lines.append("This is resource calibration only, not production.")
    lines.append("")
    grid = summary.get("target_grid", {})
    lines.append("## Target Grid")
    lines.append("")
    lines.append(f"* Name: `{grid.get('name', 'local34_diag_v1_k10000_1k')}`")
    lines.append(f"* Kmax: `{grid.get('Kmax', 10000)}`")
    lines.append(f"* Full Grid B base points planned: `{grid.get('base_points', 1000)}`")
    lines.append(f"* Full Grid B alpha tables planned: `{grid.get('alpha_tables', 20000)}`")
    lines.append(f"* Representative base points selected: `{summary.get('sample_size_selected')}`")
    lines.append("")
    lines.append("## Timed Commands")
    lines.append("")
    lines.append("| Command | Exit | Elapsed | Max RSS |")
    lines.append("| --- | ---: | ---: | ---: |")
    for command in summary.get("commands", []):
        max_rss = command.get("max_rss_mb")
        max_rss_text = f"{max_rss} MB" if max_rss is not None else "unknown"
        lines.append(f"| `{command.get('name')}` | `{command.get('exit_status', 'unknown')}` | `{command.get('elapsed_wall', 'unknown')}` | `{max_rss_text}` |")
    lines.append("")
    lines.append("## Full Plan And Shards")
    lines.append("")
    plan = summary.get("plan_summary", {})
    shard = summary.get("shard_summary", {})
    lines.append(f"* Expected bundles: `{plan.get('n_bundles_expected')}`")
    lines.append(f"* Planned bundles: `{plan.get('n_bundles_planned')}`")
    lines.append(f"* Tables planned: `{plan.get('n_tables_planned')}`")
    lines.append(f"* Constant/prefix/full tables: `{plan.get('n_constant_tables')}` / `{plan.get('n_prefix_tables')}` / `{plan.get('n_full_tables')}`")
    lines.append(f"* Adaptive raw storage GB: `{plan.get('raw_storage_gb_adaptive')}`")
    lines.append(f"* Dense raw storage GB: `{plan.get('raw_storage_gb_dense')}`")
    lines.append(f"* Shards: `{shard.get('n_shards')}`")
    lines.append(f"* Shard load imbalance: `{shard.get('load_imbalance_ratio')}`")
    lines.append(f"* Total node work proxy: `{shard.get('total_node_work_proxy')}`")
    lines.append("")
    lines.append("## Representative Build")
    lines.append("")
    build = summary.get("build_summary", {})
    lines.append(f"* Base points attempted: `{build.get('n_base_points_attempted')}`")
    lines.append(f"* Tables attempted: `{build.get('n_tables_attempted')}`")
    lines.append(f"* Tables written: `{build.get('n_tables_written')}`")
    lines.append(f"* Tables certified: `{build.get('n_tables_certified')}`")
    lines.append(f"* Certified fraction: `{build.get('certified_fraction_written')}`")
    lines.append(f"* Refinement failures: `{build.get('n_refinement_failures')}`")
    lines.append(f"* Elapsed seconds: `{build.get('elapsed_seconds')}`")
    lines.append(f"* Mean/median/max seconds per base point: `{build.get('mean_seconds_per_base_point')}` / `{build.get('median_seconds_per_base_point')}` / `{build.get('max_seconds_per_base_point')}`")
    lines.append(f"* Output MB: `{build.get('output_mb')}`")
    lines.append(f"* Max z/CDF error indicators: `{build.get('max_total_z_error_indicator')}` / `{build.get('max_total_cdf_error_indicator')}`")
    lines.append("")
    lines.append("## Point Classification")
    lines.append("")
    counts = summary.get("classification_counts", {})
    for label in ("easy", "moderate", "hard", "failed_deferred_problematic"):
        lines.append(f"* `{label}`: `{counts.get(label, 0)}`")
    lines.append("")
    lines.append("Classification uses observed per-base-point build seconds and refinement failures when available, with planner work proxy as a fallback.")
    lines.append("")
    lines.append("## GPU Clarity")
    lines.append("")
    lines.append("The representative `build-hdf5` run uses the backend configured in `examples/local34_diag_v1_k10000_1k.yaml`, currently `pgf_backend: batched`, so the selected HDF5 build is CPU-based. `RUN_GPU_AUDIT=1` is a separate GPU health/correctness check. The builder can route through CuPy only when a config explicitly sets `pgf_backend: cupy`; this workflow does not imply production HDF5 building is GPU-accelerated.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("* Full plan: `plans/full_plan/`")
    lines.append("* Shard plan: `plans/full_shards/`")
    lines.append("* Selected manifest: `sample/selected_base_points.csv` and `sample/selected_base_points.json`")
    lines.append("* Representative build: `build/representative_sample.h5` and sidecar summaries")
    lines.append("* Classification: `classification/selected_base_point_classification.csv`")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Tailbin representative Grid B calibration outputs.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", default="results/o2_representative_calibration")
    args = parser.parse_args()

    run_dir = Path(args.results_root) / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    commands = [parse_time_file(path) for path in sorted((run_dir / "timing").glob("*.time.txt"))]
    cli_json = load_json_files(run_dir / "json")
    manifest_json = load_json(run_dir / "sample" / "selected_base_points.json")
    manifest_rows = load_manifest(run_dir / "sample" / "selected_base_points.csv")
    build_summary = load_json(run_dir / "build" / "representative_sample.summary.json")
    plan_summary = load_json(run_dir / "plans" / "full_plan" / "plan_summary.json")
    shard_summary = load_json(run_dir / "plans" / "full_shards" / "balanced_shards.summary.json")
    classification = classify_points(manifest_rows, build_summary)
    classification_path = run_dir / "classification" / "selected_base_point_classification.csv"
    write_classification_csv(classification_path, classification)
    counts: dict[str, int] = {}
    for row in classification:
        label = str(row["classification"])
        counts[label] = counts.get(label, 0) + 1

    summary: dict[str, Any] = {
        "format": "tailbin_representative_calibration_summary_v1_0",
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "target_grid": {
            "name": "local34_diag_v1_k10000_1k",
            "Kmax": 10000,
            "base_points": 1000,
            "alpha_tables": 20000,
            "age_constraint": "T + T_b = 34 exact",
        },
        "commands": commands,
        "metadata": parse_slurm_metadata(run_dir),
        "cli_outputs": cli_json,
        "plan_summary": plan_summary,
        "shard_summary": shard_summary,
        "manifest_summary": manifest_json,
        "sample_size_selected": len(manifest_rows),
        "build_summary": build_summary,
        "classification": classification,
        "classification_counts": counts,
        "classification_csv": str(classification_path),
        "accounting": parse_accounting(run_dir),
        "gpu_monitor_logs": parse_gpu_monitor_logs(run_dir),
        "gpu_audit": load_json(run_dir / "gpu_audit" / "gpu_backend_audit.json"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(run_dir / "summary.md", summary)
    print(f"Wrote {run_dir / 'summary.json'}")
    print(f"Wrote {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
