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


def load_progress(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            rows.append({"parse_error": str(exc), "raw": line})
    return rows


def parse_key_value_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
        elif "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


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
    lines.append(f"* Build stage: `{summary.get('build_stage')}`")
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
    if not shard:
        lines.append("* Shard planning: `skipped`")
    lines.append("")
    lines.append("## Representative Build")
    lines.append("")
    build = summary.get("build_summary", {})
    build_info = summary.get("build_runtime", {})
    lines.append(f"* RUN_GPU_BUILD: `{build_info.get('RUN_GPU_BUILD')}`")
    lines.append(f"* Requested build backend: `{build.get('requested_pgf_backend')}`")
    lines.append(f"* Actual build backend: `{build.get('actual_build_backend')}`")
    lines.append(f"* CUDA module: `{build_info.get('CUDA module')}`")
    lines.append(f"* CUDA_VISIBLE_DEVICES: `{build_info.get('CUDA_VISIBLE_DEVICES')}`")
    lines.append(f"* CuPy version: `{build_info.get('cupy_version')}`")
    lines.append(f"* GPU device: `{build_info.get('device_name')}`")
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
    progress = summary.get("progress", {})
    lines.append(f"* Progress file: `{progress.get('path')}`")
    lines.append(f"* Progress events: `{progress.get('n_events')}`")
    lines.append(f"* Completed base points in progress: `{progress.get('completed_base_points')}`")
    lines.append(f"* Error events in progress: `{progress.get('error_events')}`")
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
    lines.append("`RUN_GPU_AUDIT=1` runs a separate GPU health/correctness audit. `RUN_GPU_BUILD=1` requests a GPU allocation for the representative HDF5 build and passes `--pgf-backend cupy --require-pgf-backend cupy` so the expensive build path uses CuPy or fails before it starts.")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    warnings = summary.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"* {warning}")
    else:
        lines.append("* none")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("* Full plan: `plans/full_plan/`")
    lines.append("* Shard plan: `plans/full_shards/`")
    lines.append("* Selected manifest: `sample/selected_base_points.csv` and `sample/selected_base_points.json`")
    lines.append("* Representative build: `build/representative_sample.h5` and sidecar summaries")
    lines.append("* Progress log: `progress/build_representative_sample.progress.jsonl`")
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
    progress_path = run_dir / "progress" / "build_representative_sample.progress.jsonl"
    progress_events = load_progress(progress_path)
    build_runtime = parse_key_value_file(run_dir / "gpu_build" / "gpu_build_metadata.txt")
    cupy_check = (run_dir / "gpu_build" / "cupy_check.txt")
    if cupy_check.exists():
        for line in cupy_check.read_text(errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                build_runtime[k.strip()] = v.strip()
    classification = classify_points(manifest_rows, build_summary)
    classification_path = run_dir / "classification" / "selected_base_point_classification.csv"
    write_classification_csv(classification_path, classification)
    counts: dict[str, int] = {}
    for row in classification:
        label = str(row["classification"])
        counts[label] = counts.get(label, 0) + 1
    progress_summary = {
        "path": str(progress_path),
        "exists": bool(progress_path.exists()),
        "n_events": int(len(progress_events)),
        "completed_base_points": int(sum(1 for e in progress_events if e.get("event_type") == "finish_base_point")),
        "error_events": int(sum(1 for e in progress_events if e.get("event_type") == "error")),
        "last_event": progress_events[-1] if progress_events else None,
    }
    warnings: list[str] = []
    actual_backend = str(build_summary.get("actual_build_backend", "unknown"))
    if actual_backend not in {"cupy", "unknown"}:
        warnings.append(f"Representative build backend was `{actual_backend}`, not `cupy`.")
    output_bytes = build_summary.get("output_bytes")
    if output_bytes is not None and int(output_bytes) < 1_000_000:
        warnings.append(f"Representative HDF5 output is tiny: {output_bytes} bytes.")
    if int(build_summary.get("n_base_points_attempted", 0) or 0) == 0:
        warnings.append("No base point completed or no build summary was produced.")
    if not progress_summary["exists"] or progress_summary["n_events"] == 0:
        warnings.append("Progress file is missing or empty.")
    if shard_summary:
        warnings.append("Shard planning ran during representative calibration; keep CAL_RUN_SHARD_PLAN=0 unless shard balance is specifically being measured.")
    build_stage = "unknown"
    selected_env = run_dir / "sample" / "selected_manifest_for_build.env"
    if selected_env.exists():
        for line in selected_env.read_text(errors="replace").splitlines():
            if line.startswith("selected_stage="):
                build_stage = line.split("=", 1)[1].strip()

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
        "build_runtime": build_runtime,
        "build_stage": build_stage,
        "cli_outputs": cli_json,
        "plan_summary": plan_summary,
        "shard_summary": shard_summary,
        "manifest_summary": manifest_json,
        "sample_size_selected": len(manifest_rows),
        "build_summary": build_summary,
        "classification": classification,
        "classification_counts": counts,
        "classification_csv": str(classification_path),
        "progress": progress_summary,
        "warnings": warnings,
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
