import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from aerial_assembly.geometry import part_mesh
from aerial_assembly.mjcf import _load, _parts
from aerial_assembly.model import build_model


@pytest.mark.parametrize('external_mesh', [False, True])
def test_local_loader_preserves_transforms_and_mesh_scaling(bundle, tmp_path, external_mesh):
    _, xml = build_model(bundle)
    root = ET.fromstring(xml)
    world = root.find('worldbody')
    world.remove(world.find("body[@name='lower']"))
    world.remove(world.find("geom[@name='catch_floor']"))
    body = world.find('body')
    body.set('pos', '0.3 -0.4 0.7')
    body.set('quat', '0.7071067811865476 0 0 0.7071067811865476')
    if external_mesh:
        asset = root.find('asset/mesh')
        vertices = np.fromstring(asset.attrib.pop('vertex'), sep=' ').reshape(-1, 3)
        faces = np.fromstring(asset.attrib.pop('face'), sep=' ', dtype=int).reshape(-1, 3)
        trimesh.Trimesh(vertices=vertices*1000, faces=faces).export(tmp_path/'piece.stl')
        asset.set('file', 'piece.stl')
        asset.set('scale', '.001 .001 .001')
    path = tmp_path/'robot.xml'
    path.write_text(ET.tostring(root, encoding='unicode'))
    model, data = _load(path)
    parts = _parts(model, data, set(range(1, model.nbody)), np.zeros(3), np.eye(3))
    assert len(parts['collision']) == len(bundle['collision'])
    rotation = Rotation.from_euler('z', 90, degrees=True)
    for original, loaded in zip(bundle['collision'], parts['collision']):
        expected = rotation.apply(part_mesh(original).vertices) + [.3, -.4, .7]
        actual = part_mesh(loaded).vertices
        assert np.allclose(actual.min(axis=0), expected.min(axis=0), atol=1e-7)
        assert np.allclose(actual.max(axis=0), expected.max(axis=0), atol=1e-7)
