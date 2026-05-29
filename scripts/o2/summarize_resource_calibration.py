#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def parse_elapsed_seconds(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        parts = value.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(value)
    except ValueError:
        return None


def parse_time_file(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": path.name.replace(".time.txt", ""),
        "time_file": str(path),
    }
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or ": " not in line:
            continue
        key, value = line.rsplit(": ", 1)
        key = key.strip()
        value = value.strip()
        if key == "Command being timed":
            out["command_from_time"] = value.strip('"')
        elif key.startswith("Elapsed (wall clock) time"):
            out["elapsed_wall"] = value
            out["elapsed_seconds"] = parse_elapsed_seconds(value)
        elif key == "Maximum resident set size (kbytes)":
            try:
                out["max_rss_kb"] = int(value)
                out["max_rss_mb"] = round(int(value) / 1024.0, 3)
            except ValueError:
                out["max_rss_kb"] = None
        elif key == "Exit status":
            try:
                out["exit_status"] = int(value)
            except ValueError:
                out["exit_status"] = value
    status_file = path.with_name(path.name.replace(".time.txt", ".status.txt"))
    if status_file.exists():
        status_text = status_file.read_text(errors="replace").strip()
        try:
            out["exit_status"] = int(status_text)
        except ValueError:
            out["exit_status"] = status_text
    command_file = path.with_name(path.name.replace(".time.txt", ".command.txt"))
    if command_file.exists():
        out["command_file"] = str(command_file)
        out["command_metadata"] = command_file.read_text(errors="replace").strip()
    return out


def load_json_files(directory: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    if not directory.exists():
        return loaded
    for path in sorted(directory.glob("*.json")):
        try:
            loaded[path.stem] = json.loads(path.read_text(errors="replace"))
        except Exception as exc:  # noqa: BLE001 - summary should preserve parse failures.
            loaded[path.stem] = {"parse_error": str(exc), "path": str(path)}
    return loaded


def compact_cli_json(cli_json: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for name, data in cli_json.items():
        if not isinstance(data, dict) or "parse_error" in data:
            compact[name] = data
            continue
        item: dict[str, Any] = {}
        dense = data.get("dense_storage")
        if isinstance(dense, dict):
            item["dense_storage"] = {
                "n_tables": dense.get("n_tables"),
                "Kmax": dense.get("Kmax"),
                "raw_GB": dense.get("raw_GB"),
                "compressed_GB_estimate": dense.get("compressed_GB_estimate"),
            }
        grid = data.get("grid")
        if isinstance(grid, dict):
            item["grid"] = grid
        for key in (
            "n_bundles_expected",
            "n_bundles_planned",
            "n_tables_planned",
            "n_constant_tables",
            "n_prefix_tables",
            "n_full_tables",
            "raw_storage_gb_adaptive",
            "raw_storage_gb_dense",
            "adaptive_storage_ratio_vs_dense",
            "v04_vs_v03_node_work_ratio",
            "elapsed_seconds",
            "n_shards",
            "total_node_work_proxy",
            "max_shard_load",
            "min_shard_load",
            "load_imbalance_ratio",
            "n_base_points_attempted",
            "n_tables_written",
            "n_tables_certified",
            "certified_fraction_written",
            "n_refinement_failures",
            "mean_seconds_per_base_point",
            "output_mb",
        ):
            if key in data:
                item[key] = data[key]
        compact[name] = item
    return compact


def parse_slurm_metadata(run_dir: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for path in (run_dir / "metadata.txt", run_dir / "collection_metadata.txt"):
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                metadata[key.strip()] = value.strip()
    return metadata


def parse_accounting(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    accounting_dir = run_dir / "accounting"
    if not accounting_dir.exists():
        return out
    for path in sorted(accounting_dir.glob("*.txt")):
        out[path.name] = path.read_text(errors="replace").strip()
    return out


def parse_gpu_monitor_logs(run_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted((run_dir / "gpu_audit").glob("**/*.gpulog")):
        text = path.read_text(errors="replace")
        util_values: list[int] = []
        mem_values: list[int] = []
        for line in text.splitlines():
            for match in re.finditer(r"(\d+)\s*%", line):
                util_values.append(int(match.group(1)))
            for match in re.finditer(r"(\d+)\s*MiB", line):
                mem_values.append(int(match.group(1)))
        entries.append(
            {
                "path": str(path),
                "line_count": len(text.splitlines()),
                "max_percent_observed": max(util_values) if util_values else None,
                "max_mib_observed": max(mem_values) if mem_values else None,
            }
        )
    return entries


def recommendation(commands: list[dict[str, Any]]) -> dict[str, Any]:
    max_rss_kb = max((c.get("max_rss_kb") or 0 for c in commands), default=0)
    max_elapsed = max((c.get("elapsed_seconds") or 0.0 for c in commands), default=0.0)
    suggested_mem_gb = max(1, math.ceil((max_rss_kb * 1.5) / (1024.0 * 1024.0))) if max_rss_kb else None
    suggested_minutes = max(15, math.ceil((max_elapsed * 2.0) / 60.0)) if max_elapsed else None
    return {
        "observed_max_rss_mb": round(max_rss_kb / 1024.0, 3) if max_rss_kb else None,
        "observed_max_elapsed_seconds": round(max_elapsed, 3) if max_elapsed else None,
        "rough_next_cpu_mem": f"{suggested_mem_gb}G" if suggested_mem_gb else "unknown",
        "rough_next_walltime_minutes": suggested_minutes,
        "notes": [
            "Review summary.md before launching production pilot.",
            "Use at least 1.5x observed MaxRSS and about 2x observed elapsed time, then round to O2-friendly resource tiers.",
            "Keep easy and hard shards separated; do not upscale all jobs because one shard is hard.",
            "If GPU audit metrics are absent, run smoke/audit or resource calibration with RUN_GPU_AUDIT=1 before GPU production decisions.",
        ],
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Tailbin O2 Resource Calibration Summary")
    lines.append("")
    lines.append(f"Run ID: `{summary['run_id']}`")
    git_commit = summary.get("metadata", {}).get("Current git commit") or summary.get("metadata", {}).get("Submit git commit")
    if git_commit:
        lines.append(f"Git commit: `{git_commit}`")
    lines.append("")
    lines.append("## Timed Commands")
    lines.append("")
    lines.append("| Command | Exit | Elapsed | Max RSS |")
    lines.append("| --- | ---: | ---: | ---: |")
    for command in summary.get("commands", []):
        elapsed = command.get("elapsed_wall") or "unknown"
        max_rss = command.get("max_rss_mb")
        max_rss_text = f"{max_rss} MB" if max_rss is not None else "unknown"
        lines.append(f"| `{command.get('name')}` | `{command.get('exit_status', 'unknown')}` | `{elapsed}` | `{max_rss_text}` |")
    lines.append("")
    lines.append("## CLI Output Highlights")
    lines.append("")
    cli = summary.get("cli_outputs_compact", {})
    if cli:
        lines.append("```json")
        lines.append(json.dumps(cli, indent=2, sort_keys=True))
        lines.append("```")
    else:
        lines.append("No parseable CLI JSON outputs were found.")
    lines.append("")
    lines.append("## GPU Audit")
    lines.append("")
    gpu = summary.get("gpu_audit")
    if gpu:
        lines.append("```json")
        lines.append(json.dumps(gpu, indent=2, sort_keys=True))
        lines.append("```")
    else:
        lines.append("No GPU audit JSON was bundled with this calibration run.")
    gpu_logs = summary.get("gpu_monitor_logs") or []
    if gpu_logs:
        lines.append("")
        lines.append("GPU monitor log scan:")
        lines.append("")
        for entry in gpu_logs:
            lines.append(f"* `{entry['path']}`: max percent token `{entry['max_percent_observed']}`, max MiB token `{entry['max_mib_observed']}`")
    lines.append("")
    lines.append("## Recommendation Stub")
    lines.append("")
    rec = summary.get("recommendation", {})
    lines.append(f"* Observed max RSS: `{rec.get('observed_max_rss_mb')}` MB")
    lines.append(f"* Observed max elapsed: `{rec.get('observed_max_elapsed_seconds')}` seconds")
    lines.append(f"* Rough next CPU memory tier: `{rec.get('rough_next_cpu_mem')}`")
    lines.append(f"* Rough next wall-time floor: `{rec.get('rough_next_walltime_minutes')}` minutes")
    lines.append("* Do not launch full production until this summary is reviewed.")
    lines.append("")
    lines.append("## Accounting Files")
    lines.append("")
    accounting = summary.get("accounting", {})
    if accounting:
        for name in accounting:
            lines.append(f"* `{name}`")
    else:
        lines.append("No accounting files were found.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Tailbin O2 resource calibration outputs.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", default="results/o2_resource_calibration")
    args = parser.parse_args()

    run_dir = Path(args.results_root) / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    timing_dir = run_dir / "timing"
    commands = [parse_time_file(path) for path in sorted(timing_dir.glob("*.time.txt"))]
    cli_json = load_json_files(run_dir / "json")
    gpu_audit = {}
    for candidate in (run_dir / "gpu_audit" / "gpu_backend_audit.json",):
        if candidate.exists():
            try:
                gpu_audit = json.loads(candidate.read_text(errors="replace"))
                break
            except Exception as exc:  # noqa: BLE001
                gpu_audit = {"parse_error": str(exc), "path": str(candidate)}
                break

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "commands": commands,
        "metadata": parse_slurm_metadata(run_dir),
        "cli_outputs": cli_json,
        "cli_outputs_compact": compact_cli_json(cli_json),
        "accounting": parse_accounting(run_dir),
        "gpu_audit": gpu_audit,
        "gpu_monitor_logs": parse_gpu_monitor_logs(run_dir),
        "recommendation": recommendation(commands),
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(run_dir / "summary.md", summary)
    print(f"Wrote {run_dir / 'summary.json'}")
    print(f"Wrote {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
