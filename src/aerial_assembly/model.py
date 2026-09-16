from dataclasses import asdict
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .config import Physics, Release, quaternion, rotation
from .geometry import bounds_points


def numbers(values):
    return ' '.join(format(float(v), '.17g') for v in np.asarray(values).ravel())


def build_model(bundle, physics=Physics()):
    root = ET.Element('mujoco', model=bundle['name'])
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='false')
    option = ET.SubElement(root, 'option', timestep=str(physics.timestep), gravity='0 0 -9.81',
                           integrator='implicitfast', solver='Newton', cone='elliptic',
                           iterations=str(physics.iterations), tolerance='1e-10')
    ET.SubElement(option, 'flag', nativeccd='enable', multiccd='enable', autoreset='disable', sleep='disable')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', condim='3', friction=f'{physics.friction} 0 0', margin='0',
                  solref=f'{physics.contact_timeconst} {physics.contact_dampratio}',
                  solimp='0.95 0.99 0.0001 0.5 2')
    asset = ET.SubElement(root, 'asset')
    geometry = {}
    for group in ['collision', 'visual']:
        for i, part in enumerate(bundle.get(group, [])):
            key = f'{group}_{i}'
            attrs = {'type': part['type']}
            if part['type'] == 'mesh':
                ET.SubElement(asset, 'mesh', name=key, vertex=numbers(part['vertices']),
                              face=' '.join(str(v) for f in part['faces'] for v in f))
                attrs['mesh'] = key
            else:
                attrs.update(size=numbers(part['size']), pos=numbers(part.get('pos', [0, 0, 0])),
                             quat=numbers(part.get('quat', [1, 0, 0, 0])))
            geometry[key] = attrs
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -0.3 0.7', dir='0 0 -1')
    floor_z = float(bounds_points(bundle)[:, 2].min() - 0.1)
    ET.SubElement(world, 'geom', name='catch_floor', type='plane', size='1 1 .01',
                  pos=f'0 0 {floor_z}', contype='1', conaffinity='2', rgba='.3 .3 .3 1')
    for name in ['lower', 'upper']:
        body = ET.SubElement(world, 'body', name=name)
        if name == 'upper':
            ET.SubElement(body, 'freejoint', name='release', align='false')
            inertia = bundle['inertial']
            I = np.asarray(inertia['matrix'])
            ET.SubElement(body, 'inertial', mass=str(inertia['mass']), pos=numbers(inertia['com']),
                          fullinertia=numbers([I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2]]))
        for key, attrs in geometry.items():
            visual = key.startswith('visual')
            ET.SubElement(body, 'geom', name=f'{name}_{key}', **attrs,
                          contype='0' if visual else ('2' if name == 'upper' else '1'),
                          conaffinity='0' if visual else ('1' if name == 'upper' else '2'),
                          group='2' if visual else '3',
                          rgba='.35 .65 .8 1' if name == 'upper' else '.55 .55 .6 1')
    xml = ET.tostring(root, encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml), xml


def initial_state(bundle, release=Release()):
    target_R = rotation(bundle['target']['quat'])
    # Extrinsic xyz Euler perturbation in world coordinates, about upper COM.
    R = Rotation.from_euler('xyz', release.rpy_deg, degrees=True) * target_R
    com = np.asarray(bundle['inertial']['com'])
    pos = np.asarray(bundle['target']['pos']) + target_R.apply(com) - R.apply(com)
    points = bounds_points(bundle)
    # Conservative, reproducible height definition: block AABBs fully separated.
    lift = max(0.0, float(points[:,2].max() - (R.apply(points)+pos)[:,2].min()))
    pos += [release.offset[0], release.offset[1], lift + release.height]
    omega_world = np.asarray(release.angular_velocity)
    origin_v = np.asarray(release.velocity) - np.cross(omega_world, R.apply(com))
    return {'qpos': [*pos.tolist(), *quaternion(R)],
            'qvel': [*origin_v.tolist(), *R.inv().apply(omega_world).tolist()],
            'release': asdict(release), 'clearance_lift': lift}


def set_state(model, state):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    qpos, qvel = np.asarray(state['qpos']), np.asarray(state['qvel'])
    if qpos.shape != (7,) or qvel.shape != (6,) or not np.isfinite([*qpos, *qvel]).all():
        raise ValueError('State needs finite qpos[7] and qvel[6]')
    rotation(qpos[3:])
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    return data
