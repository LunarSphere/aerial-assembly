import json

from dfbrick import cli

ERROR = '../brixzle/configs/error-exp3-r4.json'


def test_size_reports_payload_and_source(tmp_path):
    out = tmp_path/'size.json'
    cli.main(['size', '--approach-error', ERROR, '--error-stat', 'mean', '--out', str(out)])
    r = json.loads(out.read_text())
    assert r['params']['U'] == 44.5 and r['feasible']
    assert r['sized_for']['source']['run'] == 'runs/exp3-r4'


def test_drop_records_error_provenance(tmp_path):
    out = tmp_path/'drop'
    cli.main(['drop', '--samples', '2', '--approach-error', ERROR, '--error-stat', 'max', '--out', str(out)])
    s = json.loads((out/'summary.json').read_text())
    assert s['config']['error']['source']['statistic'] == 'max'
    assert s['config']['error']['radius_mm'] > 38.
    assert s['n'] == 2 and s['capture_sweep_dx'] is not None
