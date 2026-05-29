import csv
from pathlib import Path

from tailbin_cache.representative import aggregate_base_points, select_representative_base_points


ROOT = Path(__file__).resolve().parents[1]
GRID_B_CONFIG = ROOT / "examples" / "local34_diag_v1_k10000_1k.yaml"


def _write_fake_grid_b_plan(path):
    path.mkdir(parents=True, exist_ok=True)
    fields = [
        "bundle_idx",
        "R",
        "T",
        "N",
        "depth",
        "alpha",
        "alpha_index",
        "n_theta",
        "n_constant",
        "n_prefix",
        "n_full",
        "max_prefix_kmax",
        "v03_node_work_proxy",
        "v04_node_work_proxy",
    ]
    with (path / "plan_bundles.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for base_idx in range(1000):
            for alpha_index in range(20):
                hard = int(base_idx % 97 == 0)
                writer.writerow(
                    {
                        "bundle_idx": base_idx * 20 + alpha_index,
                        "R": 0.0,
                        "T": 0.0,
                        "N": 0.0,
                        "depth": 90,
                        "alpha": alpha_index,
                        "alpha_index": alpha_index,
                        "n_theta": 1,
                        "n_constant": 0 if hard else 1,
                        "n_prefix": 1 if not hard else 0,
                        "n_full": hard,
                        "max_prefix_kmax": 10000 if hard else base_idx % 100,
                        "v03_node_work_proxy": 10000 if hard else base_idx + alpha_index,
                        "v04_node_work_proxy": 10000 if hard else base_idx + alpha_index,
                    }
                )


def test_representative_selector_is_deterministic_and_includes_boundaries(tmp_path):
    plan_dir = tmp_path / "plan"
    _write_fake_grid_b_plan(plan_dir)
    points = aggregate_base_points(GRID_B_CONFIG, plan_dir)
    first = select_representative_base_points(points, sample_size=40)
    second = select_representative_base_points(points, sample_size=40)

    assert [p["base_idx"] for p in first] == [p["base_idx"] for p in second]
    assert len(first) == 40
    assert "work_proxy_easiest" in first[0]["selection_reasons"]
    assert min(p["R"] for p in first) == 0.01
    assert max(p["R"] for p in first) == 1.0
    assert min(p["N"] for p in first) == 10000.0
    assert max(p["N"] for p in first) == 100000000.0
    assert min(p["Tb"] for p in first) == 0.0
    assert max(p["Tb"] for p in first) == 20.0
    assert any("planner_full_or_large_prefix" in p["selection_reasons"] for p in first)
    assert any("work_proxy_easiest" in p["selection_reasons"] for p in first)
    assert any("work_proxy_median" in p["selection_reasons"] for p in first)
    assert any("work_proxy_hardest" in p["selection_reasons"] for p in first)
    assert [p["base_idx"] for p in first] != list(range(40))


def test_representative_stage_sizes_and_ordering(tmp_path):
    plan_dir = tmp_path / "plan"
    _write_fake_grid_b_plan(plan_dir)
    points = aggregate_base_points(GRID_B_CONFIG, plan_dir)

    easy = select_representative_base_points(points, sample_size=1, strategy="easy_first")
    stratified = select_representative_base_points(points, sample_size=8, strategy="representative_stratified")
    representative = select_representative_base_points(points, sample_size=40, strategy="representative_hard")
    hard_first = select_representative_base_points(points, sample_size=8, strategy="representative_hard_first")

    assert len(easy) == 1
    assert len(stratified) == 8
    assert len(representative) == 40
    assert "work_proxy_easiest" in easy[0]["selection_reasons"]
    assert "work_proxy_easiest" in stratified[0]["selection_reasons"]
    assert "work_proxy_easiest" in representative[0]["selection_reasons"]
    assert "work_proxy_hardest" not in representative[0]["selection_reasons"]
    assert "work_proxy_hardest" in hard_first[0]["selection_reasons"] or "planner_full_or_large_prefix" in hard_first[0]["selection_reasons"]


def test_representative_selector_preserves_explicit_base_points(tmp_path):
    plan_dir = tmp_path / "plan"
    _write_fake_grid_b_plan(plan_dir)
    points = aggregate_base_points(GRID_B_CONFIG, plan_dir)
    selected = select_representative_base_points(points, sample_size=40)

    assert len({p["base_idx"] for p in selected}) == 40
    assert all(p["n_theta"] == 1 for p in selected)
    assert all(abs(p["T"] + p["Tb"] - 34.0) <= 1e-9 for p in selected)
