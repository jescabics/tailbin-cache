from tailbin_cache.cli import main


def test_cli_estimate_and_plan(tmp_path, capsys):
    cfg = tmp_path / 'smoke.yaml'
    assert main(['init-config', '--output', str(cfg)]) == 0
    assert cfg.exists()
    assert main(['estimate', '--config', str(cfg)]) == 0
    out = capsys.readouterr().out
    assert 'dense_storage' in out
    plan_dir = tmp_path / 'plan'
    assert main(['plan', '--config', str(cfg), '--output-dir', str(plan_dir), '--limit-bundles', '1']) == 0
    out = capsys.readouterr().out
    assert 'n_tables_planned' in out
