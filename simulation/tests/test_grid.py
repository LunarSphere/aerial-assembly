from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from aerial_assembly import cad_workflow, cli, grid
from aerial_assembly.config import Release, TrialSettings, read_json, rotation, write_json
from aerial_assembly.geometry import bounds_points
from aerial_assembly.model import initial_state
from aerial_assembly.simulation import meets_reference, reference_metrics, run_drop


def test_inclusive_decimal_grid_and_six_axes():
    g = grid.Grid({'dx_mm': [-.2, .2, .1], 'theta_z_deg': [-1, 1, 1]})
    assert g.total == 15
    assert g.state(0)['dx_mm'] == -.2
    assert g.state(1)['theta_z_deg'] == 0
    assert g.state(3)['dx_mm'] == -.1
    assert g.state(14)['dx_mm'] == .2
    assert g.state(14)['theta_z_deg'] == 1
    for spec in ([0, 1, 0], [0, 1, -.1], [2, 1, 1], [0, 1, .3], [0, float('inf'), 1]):
        with pytest.raises(ValueError):
            grid.Grid({'dx_mm': spec})
    with pytest.raises(ValueError):
        grid.Grid({'dz_mm': [-1, 1, 1]})
    assert grid.Grid({name: [-30, 30, 1] for name in grid.AXES if name != 'dz_mm'}).total == 61**5


def test_profile_center_rotation_and_clearance(bundle):
    points = bounds_points(bundle)
    pivot = (points.min(axis=0)+points.max(axis=0))/2
    a = initial_state(bundle, Release(), pivot=pivot, points=points)
    release = Release(offset=(.012, -.003), height=.03, rpy_deg=(8, -12, 6))
    b = initial_state(bundle, release, pivot=pivot, points=points)
    R = rotation(b['qpos'][3:])
    delta = R.apply(pivot)+b['qpos'][:3] - (rotation(a['qpos'][3:]).apply(pivot)+a['qpos'][:3])
    assert delta[:2] == pytest.approx([.012, -.003])
    assert (R.apply(points)+b['qpos'][:3])[:, 2].min()-points[:, 2].max() == pytest.approx(.03)


@pytest.fixture
def local_grid(bundle, monkeypatch, tmp_path, refined_physics):
    monkeypatch.setattr(cad_workflow, 'prepare_local_export', lambda *a, **kw: deepcopy(bundle))
    config = tmp_path/'config.json'
    write_json(config, {'grid_search': {'dx_mm': [0, 2, 1]},
                       'physics': asdict(refined_physics),
                       'trial': {'duration': .0001, 'dwell': .00005}})
    out = tmp_path/'grid'
    args = ['cad-grid', 'local', '--config', str(config), '--out', str(out)]
    return config, out, args


def test_cli_resume_conversion_and_recording(local_grid):
    config, out, args = local_grid
    assert cli.main(args + ['--dry-run']) == 0
    assert not out.exists()
    assert cli.main(args + ['--max-states', '1', '--record-trial', '0']) == 0
    report = read_json(out/'grid_summary.json')
    assert report['completed_states'] == 1
    assert report['success_fraction_total'] is None
    assert (out/'trial_00000/trajectory.npz').exists()
    assert cli.main(args) == 2
    with (out/'grid_results.jsonl').open('ab') as stream:
        stream.write(b'{"index":1')
    assert cli.main(args + ['--resume']) == 0
    rows = [json.loads(s) for s in (out/'grid_results.jsonl').read_text().splitlines()]
    assert [r['index'] for r in rows] == [0, 1, 2]
    assert rows[2]['initial_state']['release']['offset'] == [.002, 0]
    assert rows[2]['initial_state']['release']['height'] == .03
    assert read_json(out/'grid_summary.json')['complete']
    before = (out/'grid_results.jsonl').read_bytes()
    assert cli.main(args + ['--resume']) == 0
    assert (out/'grid_results.jsonl').read_bytes() == before
    values = read_json(config)
    values['grid_search']['dx_mm'] = [0, 3, 1]
    write_json(config, values)
    assert cli.main(args + ['--resume']) == 2
    assert (out/'grid_results.jsonl').read_bytes() == before


def test_interruption_checkpoints_and_summary(local_grid, monkeypatch):
    _, out, args = local_grid
    run = grid.run_drop
    calls = 0
    def interrupted(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return run(*a, **kw)
    monkeypatch.setattr(grid, 'run_drop', interrupted)
    assert cli.main(args) == 130
    assert read_json(out/'grid_summary.json')['completed_states'] == 1
    monkeypatch.setattr(grid, 'run_drop', run)
    assert cli.main(args + ['--resume']) == 0
    assert read_json(out/'grid_summary.json')['completed_states'] == 3


def test_reference_dwell_and_invalid_override(bundle, model, tmp_path):
    settings = TrialSettings(duration=.15, dwell=.1)
    state = {'qpos': [*bundle['target']['pos'], *bundle['target']['quat']], 'qvel': [0]*6}
    baseline, trace = run_drop(model, bundle, state, settings)
    reference = {'qpos': trace[-1, 1:8].tolist(), 'final': baseline['final']}
    cad_workflow.save_trial(tmp_path/'trial_00000', baseline, trace)
    write_json(tmp_path/'summary.json', {'valid': True, 'collision_probes_passed': True})
    loaded = grid.load_reference(tmp_path, bundle, settings)
    assert loaded['final_state_accepted']
    assert loaded['qpos'] == reference['qpos']
    with pytest.raises(ValueError, match='same geometry'):
        grid.load_reference(tmp_path, {**bundle, 'asset_hash': 'changed'}, settings)
    result, _ = run_drop(model, bundle, state, settings, reference=reference)
    assert result['status'] == 'success'
    assert result['success_mode'] == 'reference'
    assert result['final_target_dwell'] >= .1
    short, _ = run_drop(model, bundle, state, TrialSettings(duration=.0001, dwell=.0001), reference=reference)
    assert short['status'] != 'success'
    bad = {'qpos': [0, 0, .039, 1, 0, 0, 0], 'qvel': [0]*6}
    invalid, _ = run_drop(model, bundle, bad, settings, reference=reference)
    assert invalid['status'] == 'invalid'
    m = reference_metrics(baseline['final'], trace[-1, 1:4], trace[-1, 4:8], bundle, reference)
    assert meets_reference(m, bundle, settings, reference)
    assert meets_reference({**m, 'legs_inside': False}, bundle, settings, reference)
    assert not meets_reference({**m, 'floor_contact': True}, bundle, settings, reference)
    assert not meets_reference({**m, 'reference_translation_error': .01}, bundle, settings, reference)


def test_success_count_matches_individual_simulation(local_grid, bundle, refined_physics):
    config, out, args = local_grid
    write_json(config, {'grid_search': {'dx_mm': [0, 300, 300], 'dz_mm': [10, 10, 1]},
                       'physics': asdict(refined_physics),
                       'rotation_center': 'com', 'trial': {'duration': .3, 'dwell': .1}})
    assert cli.main(args + ['--record-successes']) == 0
    report = read_json(out/'grid_summary.json')
    assert report['successful_states'] == 1
    assert report['status_counts']['missed_receiver'] == 1
    assert report['success_fraction_total'] == .5
    assert report['successful_bounds']['dx_mm'] == [0, 0]
    assert (out/'trial_00000/trajectory.npz').exists()
    assert not (out/'trial_00001').exists()


def test_invalid_outcome_and_collision_gate(local_grid, monkeypatch):
    _, out, args = local_grid
    run = grid.run_drop
    def invalid(*a, **kw):
        result, trace = run(*a, **kw)
        return {**result, 'status': 'invalid', 'invalid_reason': 'solver_warning_or_nonfinite_state'}, trace
    monkeypatch.setattr(grid, 'run_drop', invalid)
    assert cli.main(args + ['--max-states', '1']) == 2
    assert read_json(out/'grid_summary.json')['invalid_states'] == 1
    assert read_json(out/'grid_summary.json')['successful_states'] == 0


@pytest.mark.parametrize('probe_failure', [True, False])
def test_geometry_gate_prevents_grid_scoring(local_grid, monkeypatch, probe_failure):
    _, out, args = local_grid
    validate = cad_workflow.validate_geometry
    def rejected(*a, **kw):
        report = validate(*a, **kw)
        report['passed'] = False
        if probe_failure:
            report['socket_probes']['passed'] = False
        return report
    monkeypatch.setattr(cad_workflow, 'validate_geometry', rejected)
    assert cli.main(args) == 2
    assert not (out/'grid_results.jsonl').exists()


def test_insertion_grid_does_not_require_reference_or_flush_feasibility(local_grid, monkeypatch):
    config, out, args = local_grid
    values = read_json(config)
    values['success'] = {'mode': 'insertion'}
    write_json(config, values)
    validate = cad_workflow.validate_geometry
    def flush_infeasible(*a, **kw):
        return {**validate(*a, **kw), 'passed': False}
    monkeypatch.setattr(cad_workflow, 'validate_geometry', flush_infeasible)
    assert cli.main(args) == 0
    report = read_json(out/'grid_summary.json')
    assert report['success_mode'] == 'insertion'
    assert not report['geometry_feasible']
    assert report['completed_states'] == 3


def test_27_state_config_uses_insertion_scoring():
    path = Path(__file__).parents[1]/'experiment_configs/cad-experiment-grid-fast-reference-27.json'
    g, _, _, _, success = grid.configuration(path)
    assert g.total == 27
    assert success == {'mode': 'insertion'}


def test_reference_grid_accepts_infeasible_flush_target(local_grid, monkeypatch, bundle, model, tmp_path,
                                                      refined_physics):
    config, out, args = local_grid
    # A saved, settled synthetic reference exercises the complete reference-file path.
    state = {'qpos': [*bundle['target']['pos'], *bundle['target']['quat']], 'qvel': [0]*6}
    result, trace = run_drop(model, bundle, state, TrialSettings(duration=.15, dwell=.1))
    reference_dir = tmp_path/'reference'
    cad_workflow.save_trial(reference_dir/'trial_00000', result, trace)
    write_json(reference_dir/'summary.json', {'valid': True, 'collision_probes_passed': True})
    write_json(config, {'grid_search': {'dz_mm': [10, 10, 1]},
                       'physics': asdict(refined_physics),
                       'trial': {'duration': .3, 'dwell': .1},
                       'success': {'mode': 'reference', 'reference_run': str(reference_dir)}})
    validate = cad_workflow.validate_geometry
    def infeasible(*a, **kw):
        return {**validate(*a, **kw), 'passed': False}
    monkeypatch.setattr(cad_workflow, 'validate_geometry', infeasible)
    assert cli.main(args + ['--record-successes']) == 0
    report = read_json(out/'grid_summary.json')
    assert report['success_mode'] == 'reference'
    assert report['successful_states'] == 1
    assert not report['geometry_feasible']
    assert (out/'trial_00000/trajectory.npz').exists()
