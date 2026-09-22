from copy import deepcopy

import numpy as np

from aerial_assembly.model import build_model
from aerial_assembly.simulation import pose_metrics, validate_geometry


def test_target_and_insertion_path_are_open(bundle, model):
    assert validate_geometry(model, bundle)['passed']


def test_closed_socket_is_rejected(bundle):
    b = deepcopy(bundle)
    b['collision'].append({'name': 'incorrect_solid_hole', 'type': 'cylinder',
                           'size': [0.005, 0.017], 'pos': [-0.02, 0, 0.023]})
    assert not validate_geometry(build_model(b)[0], b)['passed']


def test_missing_socket_floor_is_rejected(bundle):
    b = deepcopy(bundle)
    b['collision'] = [p for p in b['collision'] if p['name'] != 'floor_0']
    report = validate_geometry(build_model(b)[0], b)
    assert not report['passed']
    assert any(f['feature'] == 'floor' for f in report['socket_probes']['failures'])


def test_rim_pose_cannot_count_as_seated(bundle):
    pose = np.asarray(bundle['target']['pos']) + [0.007, 0, 0.03]
    metrics = pose_metrics(bundle, pose, bundle['target']['quat'])
    assert not metrics['legs_inside']
    assert metrics['max_seating_gap'] > 0.02
