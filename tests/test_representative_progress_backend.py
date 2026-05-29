import json
from pathlib import Path

import pytest

from tailbin_cache.hdf5_alpha_monotone import _ProgressWriter, build_alpha_monotone_hdf5_from_config
from tailbin_cache.runner import build_config_from_dict, config_from_yaml


ROOT = Path(__file__).resolve().parents[1]
GRID_B_CONFIG = ROOT / "examples" / "local34_diag_v1_k10000_1k.yaml"


def test_progress_writer_jsonl(tmp_path):
    path = tmp_path / "progress.jsonl"
    progress = _ProgressWriter(path, echo_stdout=False)
    progress.emit("start_base_point", stage="test", base_idx=3, backend="cupy")
    progress.emit("finish_base_point", stage="test", base_idx=3, elapsed_seconds_base_point=1.25)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event_type"] for row in rows] == ["start_base_point", "finish_base_point"]
    assert rows[0]["backend"] == "cupy"


def test_required_gpu_backend_fails_before_build_for_cpu_config(tmp_path):
    cfg = GRID_B_CONFIG
    before = config_from_yaml(cfg)
    build_before, _, _ = build_config_from_dict(before)

    with pytest.raises(RuntimeError, match="requires pgf_backend"):
        build_alpha_monotone_hdf5_from_config(
            cfg,
            tmp_path / "should_not_build.h5",
            require_pgf_backend="cupy",
            base_point_manifest=None,
            progress_stdout=False,
        )

    after = config_from_yaml(cfg)
    build_after, _, _ = build_config_from_dict(after)
    assert build_before.pgf_backend == "batched"
    assert build_after.pgf_backend == "batched"
