import json

import pytest

from biasbrick import cli


def test_export_writes_files_and_refuses_a_used_directory(tmp_path):
    out = tmp_path/'export'
    cli.main(['export', '--params', 'configs/bb-mean.json', '--out', str(out)])
    for f in ('brick.stl', 'brick.step', 'geometry.json', 'params.json', 'views.png', 'stack.png', 'sections.png'):
        assert (out/f).exists(), f
    g = json.loads((out/'geometry.json').read_text())
    assert g['fragments'] == 1 and g['params']['U'] == 41.0
    with pytest.raises(SystemExit):
        cli.main(['export', '--params', 'configs/bb-mean.json', '--out', str(out)])


def test_size_from_measured_error(tmp_path, capsys):
    out = tmp_path/'size.json'
    cli.main(['size', '--approach-error', '../brixzle/configs/error-exp3-r4.json', '--error-stat', 'p95',
              '--out', str(out)])
    assert json.loads(out.read_text())['params']['U'] == 68.0
    assert 'U = 68.0 mm' in capsys.readouterr().out
