from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull
import trimesh

from aerial_assembly.cad_collision import partition_solid
from aerial_assembly.cad_single import recognize_single
from aerial_assembly.cad_workflow import prepare_download
from aerial_assembly.geometry import part_mesh
from aerial_assembly.model import build_model
from aerial_assembly.simulation import pose_metrics, validate_geometry


def station():
    # One pointed peg, flat underside, tapered mouth, cylindrical blind socket.
    profile = [[0, -.025333333], [.003166667, -.022166666], [.003166667, 0],
               [.014, 0], [.014, .026666667], [.011666667, .026666667],
               [.003666667, .01281026], [.003666667, .004], [0, .004]]
    return trimesh.creation.revolve(profile, sections=32)


def two_stations():
    a, b = station(), station()
    a.apply_translation([-.03, 0, 0])
    b.apply_translation([.03, 0, .026666667])
    return trimesh.util.concatenate([a, b])


def test_single_mesh_features_follow_translation_and_scale():
    mesh = two_stations()
    original = recognize_single(mesh)
    assert original['target']['pos'][2] == pytest.approx(.026666667)
    assert original['sockets'][0]['depth'] == pytest.approx(.022666667)
    assert original['legs'][0]['target_depth'] == pytest.approx(.025333333)
    mesh.apply_scale(1.5)
    mesh.apply_translation([.1, -.2, .3])
    result = recognize_single(mesh)
    assert result['target']['pos'][2] == pytest.approx(.04)
    assert result['sockets'][0]['depth'] == pytest.approx(.034)
    assert result['legs'][0]['target_depth'] == pytest.approx(.038)


def test_single_mesh_rejects_missing_axis_and_tilt():
    with pytest.raises(ValueError, match='exactly two'):
        recognize_single(station())
    mesh = two_stations()
    mesh.apply_transform(trimesh.transformations.rotation_matrix(.2, [1, 0, 0]))
    with pytest.raises(ValueError, match='exactly two'):
        recognize_single(mesh)


def test_tetrahedral_collision_preserves_blind_hole_and_material():
    mesh = station()
    pieces = partition_solid(mesh)
    assert all(p.is_volume and p.is_convex for p in pieces)
    assert sum(p.volume for p in pieces) == pytest.approx(mesh.volume, rel=2e-5)
    def material(point):
        return any(np.all(ConvexHull(p.vertices).equations[:, :3] @ point +
                             ConvexHull(p.vertices).equations[:, 3] <= 1e-10) for p in pieces)
    assert not material([0, 0, .02])
    assert material([0, 0, .003])
    assert material([.013, 0, .02])


def test_profile_based_scoring_detects_bottoming():
    features = recognize_single(two_stations())
    metrics = pose_metrics(features, features['target']['pos'], [1, 0, 0, 0])
    assert not metrics['legs_inside']
    assert metrics['max_seating_gap'] == pytest.approx(0)
    assert metrics['max_depth_error'] == pytest.approx(0)


@pytest.mark.requires_mk2
def test_mk2_distinguishes_socket_floor_from_hollow_peg(tmp_path):
    directory = Path(__file__).resolve().parents[1]/'goat_mk2'
    if not (directory/'assets/part_1.stl').exists():
        pytest.fail('Integration blocked: restore the local goat_mk2 export.')
    bundle = prepare_download(directory, cache=tmp_path/'cache')
    assert len(bundle['visual']) == 1
    for socket, peg in zip(bundle['sockets'], bundle['legs']):
        assert socket['depth'] == pytest.approx(.022666667, abs=1e-7)
        assert peg['target_depth'] == pytest.approx(.025333333, abs=1e-7)
    assert sum(part_mesh(p).volume for p in bundle['collision']) == pytest.approx(
        part_mesh(bundle['visual'][0]).volume, rel=2e-5)
    gate = validate_geometry(build_model(bundle)[0], bundle)
    assert gate['socket_probes']['passed']
    assert not gate['passed']
    assert gate['target']['penetration'] > bundle['max_penetration']
    assert not gate['target']['legs_inside']
