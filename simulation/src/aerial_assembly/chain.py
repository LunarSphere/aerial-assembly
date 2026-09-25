"""Incrementally add two-peg blocks to a floor-supported complement block."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .cad_collision import partition_solid
from .cad_workflow import _inertia, prepare_local_export
from .cad_single import circular_rings
from .config import Physics, read_json, write_json
from .geometry import bounds_points, part_mesh
from .mjcf import _load, _parts


def prepare_complement(directory, mass_grams=480., cache='assets/cad-cache', progress=print):
    """Prepare a socket-only watertight block, preserving its cavities in collision."""
    directory = Path(directory).resolve()
    model, data = _load(directory/'robot.xml')
    if model.nu or model.neq or any(int(j) != mujoco.mjtJoint.mjJNT_FREE for j in model.jnt_type):
        raise ValueError('Complement export must contain rigid parts only')
    parts = _parts(model, data, set(range(1, model.nbody)), np.zeros(3), np.eye(3))
    visual = parts['visual'] or parts['collision']
    meshes = [part_mesh(p) for p in visual]
    if len(meshes) != 1 or not meshes[0].is_volume:
        raise ValueError('Complement must be one watertight solid')
    mesh = meshes[0]
    rings = circular_rings(mesh)
    centers = []
    for ring in rings:
        if ring['inner'] and not any(np.linalg.norm(ring['center']-c) < 1e-6 for c in centers):
            centers.append(ring['center'])
    if len(centers) != 2:
        raise ValueError('Complement must have exactly two vertical circular sockets')
    sockets = []
    for center in sorted(centers, key=lambda c: c[0]):
        axis = [r for r in rings if r['inner'] and np.linalg.norm(r['center']-center) < 1e-6]
        axis.sort(key=lambda r: r['z'])
        if len(axis) < 3 or any(axis[i]['radius'] > axis[i+1]['radius']+1e-7 for i in range(len(axis)-1)):
            raise ValueError('Complement sockets need a floor, throat, and widening mouth')
        floor, throat, mouth = axis[0], axis[-2], axis[-1]
        sockets.append({'mouth': [float(center[0]), float(center[1]), mouth['z']],
                        'quat': [1., 0., 0., 0.], 'depth': mouth['z']-floor['z'],
                        'lead_depth': mouth['z']-throat['z'],
                        'opening_radius': mouth['radius'], 'throat_radius': throat['radius'],
                        'radius_profile': [[mouth['z']-r['z'], r['radius']] for r in reversed(axis)]})
    if mass_grams <= 0 or not np.isfinite(mass_grams):
        raise ValueError('Complement mass must be positive')
    density = mass_grams/1000/mesh.volume
    inertia = _inertia(meshes, density)
    progress('Partitioning complement collision geometry while preserving sockets…')
    collision_meshes = partition_solid(mesh)
    collision = [{'name': f'complement_{i}', 'type': 'mesh', 'vertices': m.vertices.tolist(),
                  'faces': m.faces.tolist()} for i, m in enumerate(collision_meshes)]
    source_files = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in ('.xml', '.stl', '.obj')}
    bundle = {'name': 'two_peg_block_complement', 'visual': visual, 'collision': collision,
              'sockets': sockets, 'inertial': inertia, 'mass_grams': mass_grams,
              'bounds': mesh.bounds.tolist(), 'source_files_sha256': source_files}
    bundle['asset_hash'] = hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest()
    return bundle


def _geometry_xml(asset, body, part, index, collision, dynamic, asset_prefix):
    name = f'{body}_{"col" if collision else "vis"}_{index}'
    attrs = {'name': name, 'type': part['type'], 'group': '3' if collision else '2',
             'contype': ('2' if dynamic else '1') if collision else '0',
             'conaffinity': ('3' if dynamic else '2') if collision else '0',
             'rgba': '.9 .45 .12 1' if body == 'base' else '.35 .65 .8 1'}
    if part['type'] == 'mesh':
        key = f'{asset_prefix}_{index}'
        if asset.find(f"mesh[@name='{key}']") is None:
            ET.SubElement(asset, 'mesh', name=key, vertex=' '.join(map(str, np.asarray(part['vertices']).ravel())),
                          face=' '.join(map(str, np.asarray(part['faces'], dtype=int).ravel())))
        attrs['mesh'] = key
    else:
        attrs['size'] = ' '.join(map(str, part['size']))
        attrs['pos'] = ' '.join(map(str, part.get('pos', [0, 0, 0])))
        attrs['quat'] = ' '.join(map(str, part.get('quat', [1, 0, 0, 0])))
    return ET.Element('geom', attrs)


def _model(base, stepper, max_blocks, physics, *, complement_fixed=False, base_floor_z=0.):
    root = ET.Element('mujoco', model='two_peg_block_chain')
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='false')
    if len(base['collision'])+max_blocks*len(stepper['collision']) > 1000:
        ET.SubElement(root, 'size', memory='128M')
    option = ET.SubElement(root, 'option', timestep=str(physics.timestep), gravity='0 0 -9.81',
                           integrator='implicitfast', solver='Newton', cone='elliptic',
                           iterations=str(physics.iterations), tolerance='1e-10')
    ET.SubElement(option, 'flag', nativeccd='enable', multiccd='enable', autoreset='disable', sleep='disable')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', condim='3', friction=f'{physics.friction} 0 0', margin='0',
                  solref=f'{physics.contact_timeconst} {physics.contact_dampratio}',
                  solimp='0.95 0.99 0.0001 0.5 2')
    asset = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -0.3 0.8', dir='0 0 -1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 .01', pos='0 0 0',
                  contype='1', conaffinity='2', rgba='.22 .22 .24 1')
    bundle_for = [base]+[stepper]*max_blocks
    for i, bundle in enumerate(bundle_for):
        name = 'base' if i == 0 else f'block_{i}'
        attrs = {'name': name}
        if i == 0 and complement_fixed:
            attrs['pos'] = f'0 0 {base_floor_z}'
        if i:
            attrs.update(pos=f'{1+.1*i} 0 2', gravcomp='1')
        body = ET.SubElement(world, 'body', attrs)
        ET.SubElement(body, 'freejoint', name=name+'_free', align='false')
        inertial = bundle['inertial']
        I = np.asarray(inertial['matrix'])
        ET.SubElement(body, 'inertial', mass=str(inertial['mass']), pos=' '.join(map(str, inertial['com'])),
                      fullinertia=' '.join(map(str, [I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2]])))
        for j, part in enumerate(bundle['collision']):
            body.append(_geometry_xml(asset, name, part, j, True, True,
                                      'base_collision' if i == 0 else 'stepper_collision'))
        for j, part in enumerate(bundle['visual']):
            body.append(_geometry_xml(asset, name, part, j, False, i > 0,
                                      'base_visual' if i == 0 else 'stepper_visual'))
    if complement_fixed:
        equality = ET.SubElement(root, 'equality')
        ET.SubElement(equality, 'weld', name='base_world_lock', body1='base', active='true')
    xml = ET.tostring(root, encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml), xml


def _inside(socket, point, allowance=0.):
    delta = np.asarray(point)-np.asarray(socket['mouth'])
    depth = -delta[2]
    profile = np.asarray(socket['radius_profile'])
    radius = float(np.interp(depth, profile[:, 0], profile[:, 1]))
    return 0 <= depth <= socket['depth'] and np.linalg.norm(delta[:2]) <= radius+allowance


def _poses(bundle, pos, quat):
    from scipy.spatial.transform import Rotation
    q = np.asarray(quat)
    R = Rotation.from_quat(q[[1, 2, 3, 0]])
    return R, np.asarray(pos)


def _joint_addresses(model, name):
    joint_id = model.joint(name).id
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def cad_chain(base_directory, stepper_directory, output, *, max_blocks=20,
              base_mass_grams=480., block_mass_grams=26., config=None, cache='assets/cad-cache',
              progress=print):
    """Add one idealized block per stage, preserving all bodies' prior states."""
    if max_blocks < 1:
        raise ValueError('max_blocks must be positive')
    values = read_json(config) if config else {}
    unknown = set(values)-{'physics', 'settle_seconds', 'insertion_margin', 'complement_fixed'}
    if unknown:
        raise ValueError(f'Unknown chain config fields: {sorted(unknown)}')
    physics = Physics(**values.get('physics', {'timestep': .002, 'contact_timeconst': .005}))
    settle_seconds = float(values.get('settle_seconds', 1.0))
    margin = float(values.get('insertion_margin', .0005))
    complement_fixed = values.get('complement_fixed', False)
    if not isinstance(complement_fixed, bool):
        raise ValueError('complement_fixed must be a boolean')
    if settle_seconds <= 0 or margin < 0 or not np.isfinite([settle_seconds, margin]).all():
        raise ValueError('settle_seconds must be positive and insertion_margin nonnegative')
    base = prepare_complement(base_directory, base_mass_grams, cache, progress)
    stepper = prepare_local_export(stepper_directory, mass_grams=block_mass_grams,
                                   cache=cache, progress=progress)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Set the complement on the floor before compiling so a fixed weld captures
    # its intended floor-resting pose as the equality reference.
    base_low = np.min(bounds_points(base), axis=0)
    base_floor_z = -float(base_low[2])
    # Compile every copy once. Unused blocks remain parked far from the floor
    # with gravity compensation until their placement stage is activated.
    model, scene_xml = _model(base, stepper, max_blocks, physics,
                              complement_fixed=complement_fixed, base_floor_z=base_floor_z)
    (output/'scene.xml').write_text(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # Set the complement on the floor. Its sockets are ordered left-to-right.
    base_qa, base_da = _joint_addresses(model, 'base_free')
    data.qpos[base_qa:base_qa+7] = [0., 0., base_floor_z, 1., 0., 0., 0.]
    data.qvel[base_da:base_da+6] = 0.
    mujoco.mj_forward(model, data)
    stages = []
    dt = physics.timestep
    steps = round(settle_seconds/dt)
    if not math.isclose(steps*dt, settle_seconds, abs_tol=1e-10):
        raise ValueError('settle_seconds must be an integer multiple of timestep')
    for added in range(1, max_blocks+1):
        parent_i = added-1
        parent_bundle = base if parent_i == 0 else stepper
        # The outward socket and rear peg make a level, forward-growing chain.
        socket_i = int(np.argmax([s['mouth'][0] for s in parent_bundle['sockets']]))
        leg_i = int(np.argmin([leg['tip'][0] for leg in stepper['legs']]))
        socket = parent_bundle['sockets'][socket_i]
        leg = stepper['legs'][leg_i]
        depth = min(float(leg['target_depth']), float(socket['depth'])-margin)
        if depth <= 0:
            raise ValueError('Peg is too long for the selected socket or insertion margin')
        # Preserve level orientation and place the peg tip inside the socket.
        parent_name = 'base' if parent_i == 0 else f'block_{parent_i}'
        parent_qa, _ = _joint_addresses(model, parent_name+'_free')
        parent_state = data.qpos[parent_qa:parent_qa+7].copy()
        parent_R, parent_pos = _poses(parent_bundle, parent_state[:3], parent_state[3:7])
        world_mouth = parent_R.apply(socket['mouth'])+parent_pos
        world_tip = world_mouth + parent_R.apply([0, 0, -depth])
        new_pos = world_tip-parent_R.apply(leg['tip'])
        new_state = np.array([*new_pos, *parent_state[3:7]])
        if not _inside(socket, parent_R.inv().apply(world_tip-parent_pos), margin):
            raise ValueError('Generated placement does not put the selected peg tip in the socket')
        other_leg = 1-leg_i
        other_tip = new_pos+parent_R.apply(stepper['legs'][other_leg]['tip'])
        if any(_inside(s, parent_R.inv().apply(other_tip-parent_pos))
               for s in parent_bundle['sockets']):
            raise ValueError('Level placement inserts both pegs; cannot satisfy the one-peg placement rule')
        body_name = f'block_{added}'
        body_id = model.body(body_name).id
        qa, da = _joint_addresses(model, body_name+'_free')
        model.body_gravcomp[body_id] = 0.
        data.qpos[qa:qa+7] = new_state
        data.qvel[da:da+6] = 0.
        mujoco.mj_forward(model, data)
        stride = max(1, round(.01/dt))
        trace = []
        failure = None
        floor_id = model.geom('floor').id
        for step in range(steps):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or np.any(data.warning.number):
                failure = {'kind': 'solver_warning', 'stage': added}
                break
            if (step+1) % stride == 0 or step+1 == steps:
                active_qpos, active_qvel = [], []
                for body_index in range(added+1):
                    active_name = 'base' if body_index == 0 else f'block_{body_index}'
                    aq, ad = _joint_addresses(model, active_name+'_free')
                    active_qpos.extend(data.qpos[aq:aq+7])
                    active_qvel.extend(data.qvel[ad:ad+6])
                trace.append([data.time, *active_qpos, *active_qvel])
            # A disconnected peg is not collapse by itself; stop only when a
            # stepping block reaches the floor.
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                if contact.dist > 0 or floor_id not in (contact.geom1, contact.geom2):
                    continue
                other_geom = contact.geom2 if contact.geom1 == floor_id else contact.geom1
                body_id = int(model.geom_bodyid[other_geom])
                if body_id != model.body('base').id:
                    failure = {'kind': 'block_on_floor', 'stage': added,
                               'block': model.body(body_id).name, 'time': float(data.time)}
                    break
            if failure:
                break
        # Record only the active assembly; parked copies stay out of the trace.
        states = []
        for i in range(added+1):
            name = 'base' if i == 0 else f'block_{i}'
            qa, da = _joint_addresses(model, name+'_free')
            states.append(np.r_[data.qpos[qa:qa+7], data.qvel[da:da+6]])
        status = ('invalid' if failure and failure['kind'] == 'solver_warning' else
                  'collapse' if failure else 'stable')
        stage = {'added_blocks': added, 'status': status,
                 'failure': failure, 'states': [s.tolist() for s in states],
                 'physics': asdict(physics), 'settle_seconds': settle_seconds,
                 'compiled_block_capacity': max_blocks,
                 'complement_fixed': complement_fixed}
        write_json(output/f'stage_{added:03d}.json', stage)
        np.savez_compressed(output/f'stage_{added:03d}.npz', trace=np.asarray(trace))
        stages.append(stage)
        progress(f"Chain: {added} blocks; {'collapse' if failure else 'stable'}")
        if failure:
            break
    stable = max((s['added_blocks'] for s in stages if s['status'] == 'stable'), default=0)
    terminal_status = stages[-1]['status'] if stages and stages[-1]['status'] != 'stable' else 'limit_reached'
    summary = {'status': terminal_status,
               'max_stable_blocks': stable,
               'first_failing_addition': (stages[-1]['added_blocks'] if stages and stages[-1]['status'] == 'collapse' else None),
               'failure': stages[-1]['failure'] if stages else None,
               'base_mass_grams': base_mass_grams, 'block_mass_grams': block_mass_grams,
               'max_blocks': max_blocks, 'base_asset_hash': base['asset_hash'],
               'stepper_asset_hash': stepper['asset_hash'], 'stages': len(stages),
               'complement_fixed': complement_fixed}
    write_json(output/'summary.json', summary)
    return summary
