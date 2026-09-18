"""Deterministic, streaming CAD grids. Checkpoints commit one state at a time."""
from collections import Counter
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from importlib.metadata import version

import numpy as np

from . import cad_workflow
from .config import Physics, Release, TrialSettings, digest, read_json, write_json, rotation
from .geometry import bounds_points
from .model import build_model, initial_state
from .simulation import meets_reference, reference_metrics, run_drop


AXES = ('dx_mm', 'dy_mm', 'dz_mm', 'theta_x_deg', 'theta_y_deg', 'theta_z_deg')


class Grid:
    def __init__(self, values):
        if not isinstance(values, dict) or set(values) - set(AXES):
            raise ValueError(f'Grid axes must be drawn from {AXES}')
        self.axes = {}
        self.counts = []
        for name in AXES:
            spec = values.get(name, [30, 30, 1] if name == 'dz_mm' else [0, 0, 1])
            if not isinstance(spec, list) or len(spec) != 3:
                raise ValueError(f'{name} requires [minimum, maximum, step]')
            try:
                low, high, step = (Decimal(str(v)) for v in spec)
                if not all(v.is_finite() and math.isfinite(float(v)) for v in (low, high, step)):
                    raise ValueError('Grid values must be finite')
                if step <= 0 or high < low or (name == 'dz_mm' and low < 0):
                    raise ValueError(f'Invalid range for {name}')
                intervals = (high-low)/step
                if intervals != intervals.to_integral_value():
                    raise ValueError(f'{name}: step must divide the inclusive range exactly')
            except InvalidOperation as error:
                raise ValueError(f'Invalid numeric range for {name}') from error
            self.axes[name] = (low, high, step)
            self.counts.append(int(intervals)+1)
        self.total = math.prod(self.counts)

    def state(self, index):
        if not 0 <= index < self.total:
            raise ValueError('Grid index out of range')
        values = {}
        for name, count in reversed(list(zip(AXES, self.counts))):
            index, digit = divmod(index, count)
            low, _, step = self.axes[name]
            values[name] = float(low + digit*step)
        return {name: values[name] for name in AXES}

    def describe(self):
        return {'axes': {k: [float(v) for v in vs] for k, vs in self.axes.items()},
                'axis_order': list(AXES), 'counts': dict(zip(AXES, self.counts)),
                'total_states': self.total}


def configuration(path):
    values = read_json(path)
    if not isinstance(values, dict) or set(values) - {'grid_search', 'physics', 'trial', 'success', 'rotation_center'}:
        raise ValueError('Unknown grid configuration section')
    grid = Grid(values['grid_search'])
    physics = Physics(**values.get('physics', {}))
    settings = TrialSettings(**({'duration': 1., 'dwell': .1} | values.get('trial', {})))
    if not math.isclose(round(settings.duration/physics.timestep)*physics.timestep,
                        settings.duration, abs_tol=1e-10):
        raise ValueError('Duration must be an integer multiple of timestep')
    center = values.get('rotation_center', 'bounds_center')
    if center not in ('bounds_center', 'com'):
        raise ValueError('rotation_center must be bounds_center or com')
    success = values.get('success', {'mode': 'flush'})
    if (not isinstance(success, dict) or set(success) - {'mode', 'reference_run'} or
            success.get('mode') not in ('flush', 'reference')):
        raise ValueError('success requires mode flush or reference')
    if (success['mode'] == 'reference') != bool(success.get('reference_run')):
        raise ValueError('reference mode requires reference_run; flush mode does not use one')
    return grid, physics, settings, center, success


def load_reference(directory, bundle, settings):
    directory = Path(directory)
    result = read_json(directory/'trial_00000/result.json')
    summary = read_json(directory/'summary.json')
    if (result['asset_hash'] != bundle['asset_hash'] or not summary['valid'] or
            not summary['collision_probes_passed'] or result['status'] == 'invalid' or
            result['invalid_reason'] or result['touched_floor'] or any(result['warnings']) or
            result['max_penetration'] > bundle['max_penetration']):
        raise ValueError('Reference must be numerically valid and use the same geometry/mass')
    with np.load(directory/'trial_00000/trajectory.npz', allow_pickle=False) as saved:
        last = saved['state'][-1]
    if last.shape != (14,) or not np.isfinite(last).all() or not math.isclose(last[0], result['final']['time'], abs_tol=1e-8):
        raise ValueError('Reference trajectory must contain its final state')
    rotation(last[4:8])
    reference = {'qpos': last[1:8].tolist(), 'final': result['final'],
                 'asset_hash': result['asset_hash'], 'initial_state': result['initial_state'],
                 'original_status': result['status'], 'source_run': str(directory.resolve())}
    metrics = reference_metrics(result['final'], last[1:4], last[4:8], bundle, reference)
    if not meets_reference(metrics, bundle, settings, reference):
        raise ValueError('Reference final state does not satisfy stability/support thresholds')
    reference['final_state_accepted'] = True
    return reference


def _atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def _rows(path):
    """Read committed lines, preserving a crash-torn tail before removing it."""
    if not path.exists():
        return
    with path.open('rb+') as stream:
        index = 0
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                # Only an incomplete last line is recoverable; interior corruption fails.
                backup = path.with_name(f'{path.name}.partial-{time.time_ns()}')
                backup.write_bytes(line)
                stream.truncate(start)
                break
            row = json.loads(line)
            if row['index'] != index:
                raise ValueError('Checkpoint indices must be contiguous and unique')
            index += 1
            yield row


def cad_grid(directory, output, *, config, density=600., mass_grams=None,
             cache='assets/cad-cache', resume=False, max_states=None,
             record_successes=False, record_trials=(), progress_every=10, progress=print):
    grid, physics, settings, center, success = configuration(config)
    if (max_states is not None and max_states < 1) or progress_every < 1:
        raise ValueError('State limit and progress interval must be positive')
    if any(i < 0 or i >= grid.total for i in record_trials):
        raise ValueError('Recorded trial index is outside this grid')
    output = Path(output)
    if resume:
        if not (output/'grid_search.json').is_file():
            raise ValueError('Resume requires an initialized grid output')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another process is using this grid output') from error
        return _run(directory, output, grid, physics, settings, center, success,
                    density, mass_grams, cache, resume, max_states, record_successes,
                    record_trials, progress_every, progress)


def _run(directory, output, grid, physics, settings, center, success, density,
         mass_grams, cache, resume, max_states, record_successes, record_trials,
         progress_every, progress):
    bundle = cad_workflow.prepare_download(directory, density=density, mass_grams=mass_grams,
                                           cache=cache, progress=progress)
    reference = load_reference(success['reference_run'], bundle, settings) if success['mode'] == 'reference' else None
    manifest = {**grid.describe(), 'physics': asdict(physics), 'trial': asdict(settings),
                'rotation_center': center, 'success': success, 'reference': reference,
                'geometry_hash': bundle['asset_hash'], 'schema_version': 1,
                'versions': {'python': platform.python_version(),
                             **{name: version(name) for name in ('mujoco', 'numpy', 'scipy', 'trimesh', 'tetgen')}},
                'code_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob('*.py')}}
    if resume and digest(read_json(output/'grid_search.json')) != digest(manifest):
        raise ValueError('Resume configuration, reference, geometry, code or versions changed')
    model, xml = build_model(bundle, physics)
    if resume:
        gate = read_json(output/'geometry_validation.json')
    else:
        gate = cad_workflow.validate_geometry(model, bundle)
        write_json(output/'geometry.json', bundle)
        (output/'scene.xml').write_text(xml)
        write_json(output/'geometry_validation.json', gate)
        write_json(output/'grid_search.json', manifest)
    if not gate['socket_probes']['passed']:
        raise ValueError('Collision probes failed; cannot score a grid')
    if reference is None and not gate['passed']:
        raise ValueError('Flush target is infeasible; use an accepted reference or repair geometry')
    points = bounds_points(bundle)
    pivot = (points.min(axis=0)+points.max(axis=0))/2 if center == 'bounds_center' else np.asarray(bundle['inertial']['com'])
    counts = Counter()
    bounds = {}
    completed = 0
    elapsed = 0.

    def accumulate(row):
        nonlocal completed, elapsed
        if row['state'] != grid.state(completed):
            raise ValueError('Checkpoint state does not match grid')
        completed += 1
        elapsed += row['wall_seconds']
        counts[row['status']] += 1
        if row['success']:
            for k, v in row['state'].items():
                a, b = bounds.get(k, [v, v])
                bounds[k] = [min(a, v), max(b, v)]

    for row in _rows(output/'grid_results.jsonl'):
        accumulate(row)

    def summary():
        successes = counts['success']
        report = {'total_states': grid.total, 'completed_states': completed,
                  'remaining_states': grid.total-completed, 'complete': completed == grid.total,
                  'successful_states': successes, 'invalid_states': counts['invalid'],
                  'status_counts': dict(counts), 'success_fraction_completed': successes/completed if completed else None,
                  'success_fraction_total': successes/grid.total if completed == grid.total else None,
                  'successful_bounds': bounds, 'wall_seconds': elapsed,
                  'estimated_remaining_seconds': elapsed/completed*(grid.total-completed) if completed else None,
                  'success_mode': success['mode'], 'geometry_feasible': gate['passed']}
        _atomic_json(output/'grid_summary.json', report)
        return report

    stop = min(grid.total, completed+max_states) if max_states is not None else grid.total
    progress(f'Grid: {completed}/{grid.total} committed states; running through {stop}.')
    try:
        with (output/'grid_results.jsonl').open('a') as stream:
            for index in range(completed, stop):
                state = grid.state(index)
                release = Release(offset=(state['dx_mm']/1000, state['dy_mm']/1000),
                                  height=state['dz_mm']/1000,
                                  rpy_deg=tuple(state[k] for k in AXES[3:]))
                initial = initial_state(bundle, release, pivot=pivot, points=points)
                record = record_successes or index in record_trials
                start = time.monotonic()
                result, trace = run_drop(model, bundle, initial, settings, record=record,
                                        reference=reference,
                                        progress=lambda message: progress(f'State {index}: {message}'))
                accepted = result['status'] == 'success'
                recorded = index in record_trials or (record_successes and accepted)
                if recorded:
                    # An uncommitted recording may exist after interruption; safely regenerate it.
                    trial = output/f'trial_{index:05d}'
                    trial.mkdir(exist_ok=True)
                    write_json(trial/'result.json', result)
                    np.savez_compressed(trial/'trajectory.npz', state=trace,
                                        columns=np.array(['time','x','y','z','qw','qx','qy','qz',
                                                          'vx','vy','vz','wx_local','wy_local','wz_local']))
                row = {'index': index, 'state': state, 'success': accepted, 'status': result['status'],
                       'invalid_reason': result['invalid_reason'], 'final': result['final'],
                       'initial_state': initial, 'settling_time': result['settling_time'],
                       'final_target_dwell': result['final_target_dwell'],
                       'max_penetration': result['max_penetration'], 'warnings': result['warnings'],
                       'touched_floor': result['touched_floor'], 'recorded': recorded,
                       'wall_seconds': time.monotonic()-start}
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
                accumulate(row)
                if completed % progress_every == 0:
                    summary()
                    progress(f'Grid: {completed}/{grid.total}; {counts["success"]} successes, {counts["invalid"]} invalid.')
    finally:
        report = summary()
    return report
