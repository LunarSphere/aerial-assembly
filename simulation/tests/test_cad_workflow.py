from pathlib import Path

import numpy as np
import pytest
import trimesh

from aerial_assembly.cad_collision import clip, subtract
from aerial_assembly.cad_workflow import cad_drop, prepare_local_export, recognize_features
from aerial_assembly.config import TrialSettings
from aerial_assembly.model import build_model
from aerial_assembly.simulation import validate_geometry


@pytest.fixture(scope='module')
def local_export():
    directory = Path(__file__).resolve().parents[1]/'three_part_export'
    if not (directory/'assets').is_dir():
        pytest.skip('Optional integration requires three_part_export/assets.')
    return directory


def test_half_space_subtraction_preserves_volume_and_hole():
    block = trimesh.creation.box(extents=[.04,.04,.04])
    void = trimesh.creation.box(extents=[.01,.01,.06])
    pieces = subtract([block],void)
    assert sum(p.volume for p in pieces) == pytest.approx(.04**3-.01**2*.04, rel=1e-6)
    assert all(p.is_volume and p.is_convex for p in pieces)
    half = clip(block,np.array([1,0,0]),0)
    assert half.volume == pytest.approx(block.volume/2)


@pytest.mark.requires_three_part_export
def test_features_are_remeasured_after_peg_length_change(local_export):
    meshes = [trimesh.load_mesh(local_export/'assets'/f'part_{i}.stl') for i in [1,2,3]]
    before = recognize_features(meshes)
    for peg in meshes[:2]:
        top = peg.bounds[1,2]
        peg.vertices[:,2] = top+(peg.vertices[:,2]-top)*(30/38)
    after = recognize_features(meshes)
    assert before['legs'][0]['target_depth'] == pytest.approx(.038,abs=1e-8)
    assert after['legs'][0]['target_depth'] == pytest.approx(.030,abs=1e-8)
    assert before['sockets'] == after['sockets']
    assert after['target']['pos'] == pytest.approx(before['target']['pos'])


@pytest.mark.requires_three_part_export
def test_real_local_export_preserves_open_sockets_and_rejects_bottoming(local_export, tmp_path):
    bundle = prepare_local_export(local_export,cache=tmp_path/'cache')
    report = validate_geometry(build_model(bundle)[0],bundle)
    assert report['socket_probes']['passed']
    assert not report['passed']
    assert report['target']['penetration'] == pytest.approx(.004,abs=2e-6)
    assert bundle['inertial']['mass'] > .01
    again = prepare_local_export(local_export,cache=tmp_path/'cache')
    assert bundle['asset_hash'] == again['asset_hash']
    total = sum(trimesh.load_mesh(local_export/'assets'/f'part_{i}.stl').volume for i in [1,2,3])
    assert bundle['inertial']['mass'] == pytest.approx(600*total,rel=1e-5)


@pytest.mark.requires_three_part_export
def test_profile_rejects_unknown_topology(local_export):
    meshes = [trimesh.load_mesh(local_export/'assets'/f'part_{i}.stl') for i in [1,2,3]]
    with pytest.raises(ValueError,match='three unmerged'):
        recognize_features(meshes[:2])


@pytest.mark.requires_three_part_export
def test_infeasible_target_records_one_diagnostic_drop(local_export, tmp_path, refined_physics):
    report = cad_drop(local_export,tmp_path/'run',cache=tmp_path/'cache',
                      physics=refined_physics,
                      settings=TrialSettings(duration=.0001,dwell=.00005))
    assert report['target_infeasible']
    assert report['valid']
    assert not report['geometry_feasible']
    assert (tmp_path/'run/trial_00000/trajectory.npz').exists()
    assert not (tmp_path/'run/trial_00001').exists()
