from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import csv
import math
import hashlib

import numpy as np

from .config import Envelope, Physics, Release, TrialSettings, digest, write_json
from .model import build_model, initial_state
from .simulation import run_drop, validate_geometry


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return None
    p = successes/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/(1+z*z/total)
    return [max(0, center-half), min(1, center+half)]


def summarize(results, iid=True):
    counts = Counter(r['status'] for r in results)
    n = len(results)
    valid = n-counts['invalid']
    successes = counts['success']
    times = [r['settling_time'] for r in results if r['settling_time'] is not None]
    return {'trials': n, 'valid_trials': valid, 'counts': dict(counts),
            'invalid_fraction': counts['invalid']/n if n else None,
            'success_fraction_among_valid': successes/valid if valid else None,
            'wilson95_among_valid': wilson(successes, valid) if iid else None,
            'success_fraction_all_attempts': successes/n if n else None,
            'mean_success_settling_time': float(np.mean(times)) if times else None,
            'eligible_for_ranking': n > 0 and counts['invalid'] == 0,
            'interpretation': 'Conditional on the specified release distribution and uncalibrated physics; invalid trials are reported separately.'}


def save_trial(directory, result, trace):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'result.json', result)
    if len(trace):
        np.savez_compressed(directory/'trajectory.npz', state=trace,
                            columns=np.array(['time','x','y','z','qw','qx','qy','qz','vx','vy','vz','wx_local','wy_local','wz_local']))


def batch(bundle, releases, output, physics=Physics(), settings=TrialSettings(), seed=None, record=False, progress=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, xml = build_model(bundle, physics)
    (output/'scene.xml').write_text(xml)
    write_json(output/'geometry.json', bundle)
    write_json(output/'code_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in Path(__file__).parent.glob('*.py')})
    gate = validate_geometry(model, bundle)
    write_json(output/'geometry_validation.json', gate)
    if not gate['passed']:
        raise ValueError(f'Geometry gate failed; inspect {output}/geometry_validation.json')
    release_data = [asdict(r) for r in releases]
    write_json(output/'experiment.json', {'seed': seed, 'releases': release_data,
                                         'releases_hash': digest(release_data), 'physics': asdict(physics),
                                         'settings': asdict(settings)})
    results = []
    for i, release in enumerate(releases):
        result, trace = run_drop(model, bundle, initial_state(bundle, release), settings, record=record)
        result.update(trial_index=i, seed=seed)
        save_trial(output/f'trial_{i:05d}', result, trace)
        results.append(result)
        # Persist progress after every trial so interruption retains completed work.
        write_json(output/'summary.json', summarize(results, iid=seed is not None))
        if progress:
            progress(i, result)
    with (output/'results.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial','status','invalid_reason','x_offset_m','y_offset_m','height_m','settling_time_s',
                         'position_error_m','angle_error_rad','max_penetration_m'])
        for i, (r, release) in enumerate(zip(results, releases)):
            writer.writerow([i,r['status'],r['invalid_reason'],*release.offset,release.height,r['settling_time'],
                             r['final']['translation_error'],r['final']['angle_error'],r['max_penetration']])
    return summarize(results, iid=seed is not None)


def convergence(bundle, output, physics=Physics(), settings=TrialSettings()):
    """Paired aligned/miss/offset checks; halve dt without changing physical contact response."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    releases = [Release(), Release(offset=(0.3, 0)), Release(offset=(0.001, 0))]
    reports = []
    for label, p in [('base', physics), ('half_dt', replace(physics, timestep=physics.timestep/2))]:
        reports.append(batch(bundle, releases, output/label, p, settings, record=True))
    from .config import read_json
    comparisons = []
    for i in range(len(releases)):
        a = read_json(output/'base'/f'trial_{i:05d}'/'result.json')
        b = read_json(output/'half_dt'/f'trial_{i:05d}'/'result.json')
        comparable = a['status'] != 'invalid' and b['status'] != 'invalid'
        # Radial clearance allows a continuum of seated positions, not one exact center.
        close = a['status'] == 'missed_receiver' or (abs(a['final']['translation_error']-b['final']['translation_error']) <= min(settings.translation_tolerance, bundle['clearance'])
                 and abs(a['final']['angle_error']-b['final']['angle_error']) <= settings.angle_tolerance/5)
        comparisons.append({'trial': i, 'statuses': [a['status'], b['status']],
                            'passed': comparable and a['status'] == b['status'] and close})
    result = {'passed': all(c['passed'] for c in comparisons), 'comparisons': comparisons}
    write_json(output/'convergence.json', result)
    return result


def compare_candidates(candidates, releases, output, physics=Physics(), settings=TrialSettings(), seed=None, progress=None):
    """A small exhaustive search over prepared bundles, with identical release samples."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ranking = []
    for index, bundle in enumerate(candidates):
        try:
            summary = batch(bundle, releases, output/f'candidate_{index:03d}', physics, settings, seed, progress=progress)
        except ValueError as error:
            # A CAD/geometry rejection must not abort the remaining candidates.
            summary = {'eligible_for_ranking': False, 'rejection_reason': str(error), 'trials': 0}
        ranking.append({'candidate': index, 'asset_hash': bundle['asset_hash'], 'parameters': bundle.get('parameters', {}),
                        **summary})
        write_json(output/'candidates.json', ranking)
    eligible = [r for r in ranking if r['eligible_for_ranking']]
    eligible.sort(key=lambda r: (-r['success_fraction_all_attempts'],
                                r['mean_success_settling_time'] if r['mean_success_settling_time'] is not None else math.inf))
    result = {'ranking': eligible, 'excluded_invalid_candidates': [r for r in ranking if not r['eligible_for_ranking']],
              'release_samples_hash': digest([asdict(r) for r in releases]),
              'next_gate': 'Rerun finalists with timestep and collision refinement, independent release seeds, and calibrated physical parameters.'}
    write_json(output/'ranking.json', result)
    return result
