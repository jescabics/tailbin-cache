from tailbin_cache.grid import write_default_config
from tailbin_cache.hdf5_alpha_monotone import build_alpha_monotone_hdf5_from_config
from tailbin_cache.hdf5_adaptive import HDF5AdaptiveCache


def test_build_and_read_smoke_cache(tmp_path):
    cfg = tmp_path / 'smoke.yaml'
    out = tmp_path / 'smoke.h5'
    write_default_config(cfg, smoke=True)
    summary = build_alpha_monotone_hdf5_from_config(cfg, out, limit_base_points=1)
    assert summary['n_tables_written'] >= 1
    assert summary['n_tables_certified'] == summary['n_tables_written']
    c = HDF5AdaptiveCache(out)
    try:
        z = c.z(0, 0)
        cdf = c.cdf(0, 0)
        meta = c.metadata(0)
        assert -8.0 <= z <= 8.0
        assert 0.0 <= cdf <= 1.0
        assert meta.get('certified') is True
    finally:
        c.close()
