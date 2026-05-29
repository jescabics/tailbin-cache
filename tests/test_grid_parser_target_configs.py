from pathlib import Path

import pytest

from tailbin_cache.grid import grid_from_dict
from tailbin_cache.runner import config_from_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_theta_f_values_are_explicit_theta_axis():
    grid = grid_from_dict(
        {
            "R_values": [0.5],
            "T_values": [10],
            "theta_f_values": [0, 100, 600],
            "N_values": [10000],
            "depth_values": [90],
            "alphas": [0.05],
            "u": 20.0,
            "age_constraint_mode": "upper_bound",
            "max_age": 100,
        }
    )
    assert [p.theta_f for p in grid.parameter_points()] == [0.0, 100.0, 600.0]
    assert [p.Tb for p in grid.parameter_points()] == [0.0, 5.0, 30.0]


def test_rejects_ambiguous_background_axes():
    with pytest.raises(ValueError, match="Specify only one"):
        grid_from_dict(
            {
                "T_values": [10],
                "Tb_values": [0, 1],
                "theta_f_values": [0, 20],
            }
        )


def test_local34_exact_diagonal_target_grid_counts():
    cfg = config_from_yaml(ROOT / "examples" / "local34_diag_v1_k10000_1k.yaml")
    grid = grid_from_dict(cfg)
    assert grid.n_parameter_points == 1000
    assert grid.n_tables == 20000

    pairs = {(round(p.T, 10), round(p.Tb, 10), round(p.theta_f, 10)) for p in grid.parameter_points()}
    assert len(pairs) == 10
    assert all(abs(p.T + p.Tb - 34.0) <= 1e-9 for p in grid.parameter_points())


def test_full100k_upper_bound_target_grid_counts():
    cfg = config_from_yaml(ROOT / "examples" / "full100k_v1_k50000.yaml")
    grid = grid_from_dict(cfg)
    assert grid.n_parameter_points == 505000
    assert grid.n_tables == 10100000
    assert all(p.T + p.Tb <= 100.0 + 1e-12 for p in grid.parameter_points())
