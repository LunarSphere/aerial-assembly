from copy import deepcopy
from dataclasses import asdict
import subprocess

import numpy as np
import pytest

from aerial_assembly import cad_workflow, cli, render
from aerial_assembly.config import read_json, write_json


@pytest.mark.parametrize('command', [
    'batch', 'sweep', 'compare', 'cad-rank', 'demo-grid', 'demo', 'prepare',
    'prepare-mjcf', 'inspect-mjcf',
    'validate', 'drop', 'converge',
])
def test_removed_commands_are_rejected(command):
    with pytest.raises(SystemExit) as error:
        cli.main([command])
    assert error.value.code == 2


@pytest.mark.parametrize('flag', ['--count', '--seed'])
def test_removed_drop_options_are_rejected(flag):
    with pytest.raises(SystemExit) as error:
        cli.main(['cad-drop', 'two_peg_block', '--out', 'unused', flag, '2'])
    assert error.value.code == 2


def test_configuration_uses_single_drop_defaults(tmp_path):
    path = tmp_path/'config.json'
    write_json(path, {'release': {'height': .02}})
    physics, settings, release = cli.configuration(path)
    assert physics.timestep == .1
    assert physics.contact_timeconst == .2
    assert settings.duration == 1.
    assert settings.dwell == .1
    assert release.height == .02
    write_json(path, {'envelope': {}})
    with pytest.raises(ValueError, match='Unknown drop configuration'):
        cli.configuration(path)


@pytest.fixture
def local_drop(bundle, monkeypatch, tmp_path, grid_physics):
    # Exercise real simulation/output orchestration with synthetic test geometry.
    monkeypatch.setattr(cad_workflow, 'prepare_local_export', lambda *a, **kw: deepcopy(bundle))
    config = tmp_path/'config.json'
    write_json(config, {'physics': asdict(grid_physics),
                       'trial': {'duration': .002, 'dwell': .002}})
    output = tmp_path/'run'
    args = ['cad-drop', 'local-export', '--config', str(config), '--out', str(output)]
    return args, output


def test_single_drop_outputs_and_overwrite_protection(local_drop):
    args, output = local_drop
    assert cli.main(args) == 0
    assert {p.name for p in output.iterdir()} == {
        'scene.xml', 'geometry.json', 'geometry_validation.json', 'experiment.json',
        'code_hashes.json', 'summary.json', 'trial_00000',
    }
    summary = read_json(output/'summary.json')
    assert summary['valid']
    assert summary['status'] == 'unsettled_timeout'
    assert set(read_json(output/'experiment.json')) == {'release', 'physics', 'settings'}
    with np.load(output/'trial_00000/trajectory.npz') as saved:
        assert saved['state'].shape == (2, 14)
    before = (output/'summary.json').read_bytes()
    assert cli.main(args) == 2
    assert (output/'summary.json').read_bytes() == before


@pytest.mark.parametrize('failure', ['infeasible', 'colliders', 'path', 'numerics'])
def test_diagnostic_drops_and_invalid_exit_codes(local_drop, monkeypatch, failure):
    args, output = local_drop
    validate = cad_workflow.validate_geometry
    def gate(model, bundle):
        report = validate(model, bundle)
        if failure != 'numerics':
            report['passed'] = False
        if failure == 'infeasible':
            report['target']['penetration'] = .004
        if failure == 'colliders':
            report['socket_probes']['passed'] = False
        return report
    monkeypatch.setattr(cad_workflow, 'validate_geometry', gate)
    if failure == 'numerics':
        run = cad_workflow.run_drop
        def invalid(*a, **kw):
            result, trace = run(*a, **kw)
            result.update(status='invalid', invalid_reason='solver_warning_or_nonfinite_state')
            return result, trace
        monkeypatch.setattr(cad_workflow, 'run_drop', invalid)
    assert cli.main(args) == (0 if failure == 'infeasible' else 2)
    assert (output/'trial_00000/trajectory.npz').exists()
    assert read_json(output/'summary.json')['valid'] == (failure == 'infeasible')


def test_video_failure_preserves_recorded_drop(local_drop, monkeypatch):
    args, output = local_drop
    def fail(*a, **kw):
        raise subprocess.CalledProcessError(1, 'renderer')
    monkeypatch.setattr(render.subprocess, 'run', fail)
    assert cli.main(args + ['--video']) == 2
    assert (output/'trial_00000/trajectory.npz').exists()
    assert read_json(output/'summary.json')['valid']


def test_render_and_replay_dispatch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, 'replay', lambda *args: calls.append(args))
    monkeypatch.setattr(render, 'render_video', lambda *args: calls.append(args))
    assert cli.main(['replay', 'run', '--collisions']) == 0
    assert calls.pop() == ('run', 0, True)
    assert cli.main(['render', 'run', '--out', str(tmp_path/'drop.mp4')]) == 0
    assert calls.pop() == ('run', str(tmp_path/'drop.mp4'), 0)
