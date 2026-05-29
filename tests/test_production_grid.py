from tailbin_cache.grid import grid_from_dict, write_default_config
from tailbin_cache.runner import config_from_yaml, build_config_from_dict


def test_production_config_prunes_age_and_uses_diploid(tmp_path):
    p = tmp_path / 'prod.yaml'
    write_default_config(p, smoke=False)
    cfg = config_from_yaml(p)
    grid = grid_from_dict(cfg)
    assert grid.ploidy_factor == 2.0
    assert grid.enforce_age_constraint is True
    assert all(pt.total_age <= 100.0 + 1e-12 for pt in grid.parameter_points())
    build, budget, n_jobs = build_config_from_dict(cfg)
    assert build.Kmax == 50000
    assert budget.target_max_abs_z_error <= 0.01
    assert n_jobs == 20
