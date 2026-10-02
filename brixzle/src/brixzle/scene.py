"""MJCF scene: anchored bases, a floor, and a pool of parked brick bodies.

Follows ``aerial_assembly.chain._model``: compile once with every brick the
trial may need, park unused bricks far away under gravity compensation, and
activate them one at a time. Geometry arrives in millimetres and leaves in SI.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from aerial_assembly.config import Physics

from .cad import build_base
from .params import BrickParams

MM = 1e-3
PARK_Z = 2.0


def _numbers(values):
    return ' '.join(f'{float(v):.9g}' for v in np.ravel(values))


def _mesh(asset, name, vertices, faces=None):
    attrs = {'name': name, 'vertex': _numbers(np.asarray(vertices)*MM)}
    if faces is not None:
        attrs['face'] = ' '.join(str(int(i)) for i in np.ravel(faces))
    ET.SubElement(asset, 'mesh', attrs)


def parked_position(j):
    return np.array([1.0 + 0.2*j, 1.0, PARK_Z])


def build_scene(p: BrickParams, brick, bases, n_bricks, physics: Physics, visual=True):
    """Return (model, xml, info). ``bases`` are [(v0, v1)] voxel ranges."""
    root = ET.Element('mujoco', model='brixzle')
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='false')
    ET.SubElement(root, 'size', memory='256M')
    option = ET.SubElement(root, 'option', timestep=str(physics.timestep), gravity='0 0 -9.81',
                           integrator='implicitfast', solver='Newton', cone='elliptic',
                           iterations=str(physics.iterations), tolerance='1e-10')
    ET.SubElement(option, 'flag', nativeccd='enable', multiccd='enable', sleep='disable')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', condim='3', friction=f'{physics.friction} 0.005 0.0001', margin='0',
                  solref=f'{physics.contact_timeconst} {physics.contact_dampratio}',
                  solimp='0.95 0.99 0.0001 0.5 2')
    visual_el = ET.SubElement(root, 'visual')
    ET.SubElement(visual_el, 'global', offwidth='1280', offheight='960')
    asset = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -0.5 1.0', dir='0 0.4 -1', directional='true')
    base_bundles = []
    floor_z = 0.
    for i, (v0, v1) in enumerate(bases):
        base = build_base(p, v1 - v0)
        base_bundles.append(base)
        floor_z = min(floor_z, base['outer'].bounds[1])
        for j, piece in enumerate(base['collision']):
            name = f'base{i}_{j}'
            _mesh(asset, name, piece + np.array([v0*p.U, 0, 0]))
            ET.SubElement(world, 'geom', name=name, type='mesh', mesh=name, rgba='.55 .55 .58 1',
                          contype='1', conaffinity='1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 .01',
                  pos=f'0 0 {floor_z*MM - 0.0005}', rgba='.2 .22 .25 1', contype='1', conaffinity='1')
    for j, piece in enumerate(brick['collision']):
        _mesh(asset, f'brick_col_{j}', piece)
    if visual and 'mesh' in brick:
        _mesh(asset, 'brick_vis', brick['mesh'].vertices, brick['mesh'].faces)
    mass = brick['mass_g']*1e-3
    com = np.asarray(brick['com'])*MM
    I = np.asarray(brick['inertia'])*1e-9 if 'inertia' in brick else np.diag([1e-6]*3)
    for j in range(n_bricks):
        body = ET.SubElement(world, 'body', name=f'brick{j}', pos=_numbers(parked_position(j)),
                             gravcomp='1')
        ET.SubElement(body, 'freejoint', name=f'brick{j}_free')
        ET.SubElement(body, 'inertial', mass=str(mass), pos=_numbers(com),
                      fullinertia=_numbers([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))
        for k in range(len(brick['collision'])):
            ET.SubElement(body, 'geom', name=f'brick{j}_c{k}', type='mesh', mesh=f'brick_col_{k}',
                          group='3', rgba='.3 .6 .9 .4', contype='1', conaffinity='1')
        if visual and 'mesh' in brick:
            rgba = '.95 .55 .15 1' if j % 2 else '.25 .55 .85 1'
            ET.SubElement(body, 'geom', name=f'brick{j}_vis', type='mesh', mesh='brick_vis',
                          group='2', rgba=rgba, contype='0', conaffinity='0', mass='0')
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    info = {
        'bodies': [model.body(f'brick{j}').id for j in range(n_bricks)],
        'qpos': [model.jnt_qposadr[model.joint(f'brick{j}_free').id] for j in range(n_bricks)],
        'qvel': [model.jnt_dofadr[model.joint(f'brick{j}_free').id] for j in range(n_bricks)],
        'floor': model.geom('floor').id,
        'base_geoms': {model.geom(f'base{i}_{j}').id for i, b in enumerate(base_bundles)
                       for j in range(len(b['collision']))},
    }
    return model, xml, info


def activate(model, data, info, j, pos_mm, quat, linvel=(0, 0, 0), angvel=(0, 0, 0)):
    a, v = info['qpos'][j], info['qvel'][j]
    data.qpos[a:a+3] = np.asarray(pos_mm)*MM
    data.qpos[a+3:a+7] = quat
    data.qvel[v:v+3] = linvel
    data.qvel[v+3:v+6] = angvel
    model.body_gravcomp[info['bodies'][j]] = 0.
    mujoco.mj_forward(model, data)


def park(model, data, info, j):
    activate(model, data, info, j, parked_position(j)/MM, [1, 0, 0, 0])
    model.body_gravcomp[info['bodies'][j]] = 1.


def body_pose(data, info, j):
    a = info['qpos'][j]
    return data.qpos[a:a+3]/MM, data.qpos[a+3:a+7].copy()
