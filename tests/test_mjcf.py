from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from aerial_assembly.cli import main
from aerial_assembly.config import write_json
from aerial_assembly.geometry import load_bundle
from aerial_assembly.mjcf import inspect_mjcf, prepare_mjcf
from aerial_assembly.model import build_model
from aerial_assembly.simulation import validate_geometry


@pytest.fixture
def exported(bundle, tmp_path):
    _, xml = build_model(bundle)
    root = ET.fromstring(xml)
    world = root.find('worldbody')
    world.remove(world.find("body[@name='lower']"))
    world.remove(world.find("geom[@name='catch_floor']"))
    body = world.find('body')
    body.set('pos', '0.3 -0.4 0.7')
    body.set('quat', '0.7071067811865476 0 0 0.7071067811865476')
    ET.SubElement(body, 'site', name='socket_0', pos=' '.join(map(str, bundle['sockets'][0]['mouth'])))
    path = tmp_path/'robot.xml'
    path.write_text(ET.tostring(root, encoding='unicode'))
    metadata = {k: deepcopy(v) for k, v in bundle.items()
                if k not in {'prepared', 'asset_hash', 'collision', 'visual', 'inertial'}}
    metadata['sockets'][0]['mouth'] = {'site': 'socket_0'}
    metadata['sockets'][0]['quat'] = {'site': 'socket_0', 'field': 'quat'}
    meta = tmp_path/'metadata.json'
    write_json(meta, metadata)
    return path, meta


def test_transformed_export_roundtrip_preserves_physics(exported, bundle, tmp_path):
    xml, meta = exported
    out = tmp_path/'prepared.json'
    assert main(['prepare-mjcf', str(xml), '--metadata', str(meta), '--body', 'upper', '--out', str(out)]) == 0
    imported = load_bundle(out)
    assert imported['inertial']['mass'] == pytest.approx(bundle['inertial']['mass'])
    assert np.allclose(imported['inertial']['com'], bundle['inertial']['com'], atol=1e-10)
    assert np.allclose(imported['inertial']['matrix'], bundle['inertial']['matrix'], atol=1e-12)
    assert np.allclose(imported['sockets'][0]['mouth'], bundle['sockets'][0]['mouth'])
    assert validate_geometry(build_model(imported)[0], imported)['passed']


def test_fixed_child_inertia_parallel_axis(exported, bundle, tmp_path):
    xml, meta = exported
    root = ET.parse(xml)
    child = ET.SubElement(root.find('worldbody/body'), 'body', name='rigid_child', pos='.1 0 0',
                          quat='0.7071067811865476 0 0 0.7071067811865476')
    ET.SubElement(child, 'inertial', mass='.2', pos='0 0 0', diaginertia='.0001 .0002 .0002')
    root.write(xml)
    imported = prepare_mjcf(xml, meta, tmp_path/'prepared.json')
    m = bundle['inertial']['mass']
    c = np.asarray(bundle['inertial']['com'])
    other = np.array([.1, 0, 0])
    com = (m*c + .2*other)/(m+.2)
    expected = np.array(bundle['inertial']['matrix']) + np.diag([.0002,.0001,.0002])
    for mass, center in [(m,c), (.2,other)]:
        delta = center-com
        expected += mass*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
    assert np.allclose(imported['inertial']['matrix'], expected, atol=1e-12)
    assert np.allclose(imported['inertial']['com'], com)


def test_scaled_external_stl_and_defaults(exported, bundle, tmp_path):
    xml, meta = exported
    root = ET.parse(xml)
    asset = root.find('asset/mesh')
    vertices = np.fromstring(asset.attrib.pop('vertex'), sep=' ').reshape(-1, 3)
    faces = np.fromstring(asset.attrib.pop('face'), sep=' ', dtype=int).reshape(-1, 3)
    trimesh.Trimesh(vertices=vertices*1000, faces=faces).export(tmp_path/'piece.stl')
    asset.set('file', 'piece.stl')
    asset.set('scale', '.001 .001 .001')
    root.write(xml)
    imported = prepare_mjcf(xml, meta, tmp_path/'prepared.json')
    assert np.allclose(imported['inertial']['com'], bundle['inertial']['com'])
    assert validate_geometry(build_model(imported)[0], imported)['passed']


def test_rejects_articulation_and_placeholder_mass(exported, tmp_path):
    xml, meta = exported
    root = ET.parse(xml)
    child = ET.SubElement(root.find('worldbody/body'), 'body', name='loose')
    ET.SubElement(child, 'joint', type='hinge')
    ET.SubElement(child, 'inertial', mass='1e-9', pos='0 0 0', diaginertia='1e-9 1e-9 1e-9')
    root.write(xml)
    report = inspect_mjcf(xml)
    assert any('Internal joints' in p for p in report['problems'])
    assert any('placeholder' in p for p in report['problems'])
    with pytest.raises(ValueError, match='not ready'):
        prepare_mjcf(xml, meta, tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()


def test_export_diagnostics_for_example_failure_modes(tmp_path):
    # Reproduce the supplied export's issues without depending on mutable user files.
    a = trimesh.creation.box(extents=[.01, .01, .01])
    b = a.copy()
    b.apply_translation([.02, 0, 0])
    trimesh.util.concatenate([a, b]).export(tmp_path/'concave.stl')
    root = ET.Element('mujoco')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'mesh', name='concave', file='concave.stl')
    ET.SubElement(asset, 'material', name='bad', rgba='nan nan nan nan')
    world = ET.SubElement(root, 'worldbody')
    for i in range(1, 4):
        body = ET.SubElement(world, 'body', name=f'part_{i}')
        ET.SubElement(body, 'freejoint')
        ET.SubElement(body, 'inertial', mass='1e-9', pos='0 0 0', diaginertia='1e-9 1e-9 1e-9')
        if i == 3:
            ET.SubElement(body, 'geom', type='mesh', mesh='concave', material='bad')
        else:
            ET.SubElement(body, 'geom', type='box', size='.01 .01 .01')
    xml = tmp_path/'robot.xml'
    xml.write_text(ET.tostring(root, encoding='unicode'))
    report = inspect_mjcf(xml)
    assert report['root_bodies'] == ['part_1', 'part_2', 'part_3']
    assert len(report['placeholder_mass_bodies']) == 3
    assert sum(not c['closed_convex'] for c in report['collision']) == 1
    assert any('colors' in p for p in report['problems'])


def test_wrong_bounds_and_unknown_site_rejected(exported, tmp_path):
    import json
    xml, meta = exported
    values = json.loads(meta.read_text())
    values['sockets'][0]['mouth'] = {'site': 'missing'}
    write_json(meta, values)
    with pytest.raises(ValueError, match='Unknown block site'):
        prepare_mjcf(xml, meta, tmp_path/'bad.json')
    values['sockets'][0]['mouth'] = {'site': 'socket_0'}
    values['expected_extents'] = [1, 1, 1]
    write_json(meta, values)
    with pytest.raises(ValueError, match='bounds'):
        prepare_mjcf(xml, meta, tmp_path/'bad.json')


def test_output_and_input_files_are_not_overwritten(exported):
    xml, meta = exported
    before = meta.read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        prepare_mjcf(xml, meta, meta)
    assert meta.read_bytes() == before
