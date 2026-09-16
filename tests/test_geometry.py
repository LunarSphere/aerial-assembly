from copy import deepcopy

import numpy as np
import pytest
import trimesh

from aerial_assembly.config import write_json
from aerial_assembly.geometry import load_bundle, prepare_geometry, save_bundle
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


def test_declared_bottoming_fails(bundle, tmp_path):
    b = deepcopy(bundle)
    b['legs'][0]['target_depth'] = 0.038
    with pytest.raises(ValueError, match='bottom out'):
        save_bundle(b, tmp_path/'bad.json')


def test_mm_export_round_trip_and_units_guard(bundle, tmp_path):
    b = deepcopy(bundle)
    b.pop('asset_hash')
    b.pop('prepared')
    b['mesh_units'] = 'mm'
    ramp = b['collision'][-1]
    mesh = trimesh.Trimesh(vertices=np.asarray(ramp['vertices'])*1000, faces=ramp['faces'])
    mesh.export(tmp_path/'ramp.stl')
    b['collision'][-1] = {'name': 'ramp', 'file': 'ramp.stl'}
    write_json(tmp_path/'source.json', b)
    prepared = prepare_geometry(tmp_path/'source.json', tmp_path/'prepared.json')
    assert load_bundle(tmp_path/'prepared.json')['asset_hash'] == prepared['asset_hash']
    b['mesh_units'] = 'm'
    write_json(tmp_path/'wrong.json', b)
    with pytest.raises(ValueError, match='bounds'):
        prepare_geometry(tmp_path/'wrong.json', tmp_path/'bad.json')


def test_concave_collision_mesh_not_silently_convexified(bundle, tmp_path):
    b = deepcopy(bundle)
    a = trimesh.creation.box(extents=[.01,.01,.01])
    c = a.copy()
    c.apply_translation([.02,0,0])
    mesh = trimesh.util.concatenate([a,c])
    b['collision'] = [{'name':'concave', 'type':'mesh', 'vertices':mesh.vertices.tolist(), 'faces':mesh.faces.tolist()}]
    with pytest.raises(ValueError, match='convex'):
        save_bundle(b, tmp_path/'bad.json')


def test_bundle_hash_detects_changes(bundle, tmp_path):
    b = deepcopy(bundle)
    b['target']['pos'][2] += .001
    write_json(tmp_path/'changed.json', b)
    with pytest.raises(ValueError, match='hash mismatch'):
        load_bundle(tmp_path/'changed.json')
