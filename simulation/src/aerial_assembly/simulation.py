from dataclasses import asdict
import math
import platform
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import __version__
from .config import TrialSettings, rotation


def pose_metrics(bundle, pos, quat):
    R = rotation(quat)
    pos = np.asarray(pos)
    target = bundle['target']
    translation = float(np.linalg.norm(pos-np.asarray(target['pos'])))
    angle = float((rotation(target['quat']).inv()*R).magnitude())
    gaps = []
    for pair in bundle['seating_pairs']:
        # Normal displacement measures flushness; pose/leg checks enforce lateral placement.
        normal = np.asarray(pair.get('normal', [0, 0, 1]))
        normal = normal / np.linalg.norm(normal)
        gaps.append(abs(float(np.dot(R.apply(pair['upper'])+pos-pair['lower'], normal))))
    depths, depth_errors, inside = [], [], []
    for leg, socket in zip(bundle['legs'], bundle['sockets']):
        S = rotation(socket.get('quat', [1, 0, 0, 0]))
        pts = S.inv().apply(R.apply(leg['surface_probes'])+pos-socket['mouth'])
        tip = S.inv().apply(R.apply(leg['tip'])+pos-socket['mouth'])
        d = -pts[:, 2]
        if 'radius_profile' in socket:
            profile = np.asarray(socket['radius_profile'])
            allowed_radius = np.interp(d, profile[:, 0], profile[:, 1])
        elif socket['lead_depth']:
            blend = np.clip(1-d/socket['lead_depth'], 0, 1)
            allowed_radius = socket['throat_radius'] + blend*(socket['opening_radius']-socket['throat_radius'])
        else:
            allowed_radius = np.full(len(d), socket['throat_radius'])
        allowance = bundle['max_penetration']
        inside.append(bool(np.all((np.linalg.norm(pts[:,:2], axis=1) <= allowed_radius+allowance) &
                                  (d >= -allowance) & (d <= socket['depth']+allowance))))
        depths.append(float(-tip[2]))
        depth_errors.append(abs(float(-tip[2])-leg['target_depth']))
    return {'translation_error': translation, 'angle_error': angle, 'max_seating_gap': max(gaps),
            'insertion_depths': depths, 'max_depth_error': max(depth_errors), 'legs_inside': all(inside)}


def contact_metrics(model, data, forces=True):
    upper = model.body('upper').id
    lower = model.body('lower').id
    floor = model.geom('catch_floor').id
    penetration, floor_penetration, support, floor_contact = 0.0, 0.0, 0.0, False
    contact_force = np.zeros(6)
    for i, c in enumerate(data.contact):
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if upper not in (b1, b2):
            continue
        if floor in (c.geom1, c.geom2) and c.dist <= 0:
            floor_contact = True
            floor_penetration = max(floor_penetration, float(-c.dist))
        if lower in (b1, b2):
            penetration = max(penetration, float(-c.dist))
            if forces:
                mujoco.mj_contactForce(model, data, i, contact_force)
                force_world = c.frame.reshape(3,3).T @ contact_force[:3]
                support += float(force_world[2]) * (1 if b2 == upper else -1)
    return {'penetration': penetration, 'floor_penetration': floor_penetration,
            'support_force_z': support, 'floor_contact': floor_contact}


def measure(model, data, bundle):
    p = pose_metrics(bundle, data.qpos[:3], data.qpos[3:7])
    R = rotation(data.qpos[3:7])
    omega = R.apply(data.qvel[3:6])
    com_v = data.qvel[:3] + np.cross(omega, R.apply(bundle['inertial']['com']))
    return {**p, **contact_metrics(model, data), 'time': float(data.time),
            'linear_speed': float(np.linalg.norm(com_v)), 'angular_speed': float(np.linalg.norm(omega))}


def meets_target(metrics, bundle, settings):
    return (metrics['translation_error'] <= settings.translation_tolerance and
            metrics['angle_error'] <= settings.angle_tolerance and
            metrics['max_seating_gap'] <= settings.gap_tolerance and
            metrics['max_depth_error'] <= settings.insertion_tolerance and metrics['legs_inside'] and
            metrics['linear_speed'] <= settings.linear_speed_tolerance and
            metrics['angular_speed'] <= settings.angular_speed_tolerance and
            metrics['support_force_z'] >= bundle['inertial']['mass']*9.81*0.05 and
            metrics['penetration'] <= bundle['max_penetration'] and not metrics['floor_contact'])


def classify(final, held_for, bundle, settings, invalid_reason=None, touched_floor=False):
    if invalid_reason:
        return 'invalid', invalid_reason
    if touched_floor:
        return 'missed_receiver', None
    if held_for + 1e-10 >= settings.dwell:
        return 'success', None
    if final['linear_speed'] > settings.linear_speed_tolerance or final['angular_speed'] > settings.angular_speed_tolerance:
        return 'unsettled_timeout', None
    if final['translation_error'] <= settings.translation_tolerance and final['angle_error'] <= settings.angle_tolerance:
        if (final.get('legs_inside', False) and final.get('max_seating_gap', math.inf) <= settings.gap_tolerance
                and final.get('max_depth_error', math.inf) <= settings.insertion_tolerance):
            return 'unsettled_timeout', None  # Correct pose reached too late to satisfy the dwell.
        return 'incomplete_insertion', None
    return 'stationary_misalignment', None


def reference_metrics(metrics, pos, quat, bundle, reference):
    """Compare COM and orientation to a user-accepted, saved assembly pose."""
    com = bundle['inertial']['com']
    R, ref_R = rotation(quat), rotation(reference['qpos'][3:])
    delta = np.asarray(pos) + R.apply(com) - (np.asarray(reference['qpos'][:3]) + ref_R.apply(com))
    return {**metrics, 'reference_translation_error': float(np.linalg.norm(delta)),
            'reference_angle_error': float((ref_R.inv()*R).magnitude()),
            'reference_depth_error': float(np.max(np.abs(
                np.asarray(metrics['insertion_depths']) - reference['final']['insertion_depths'])))}


def meets_reference(metrics, bundle, settings, reference):
    return (metrics['reference_translation_error'] <= settings.translation_tolerance and
            metrics['reference_angle_error'] <= settings.angle_tolerance and
            metrics['reference_depth_error'] <= settings.insertion_tolerance and
            metrics['max_seating_gap'] <= reference['final']['max_seating_gap'] + settings.gap_tolerance and
            metrics['linear_speed'] <= settings.linear_speed_tolerance and
            metrics['angular_speed'] <= settings.angular_speed_tolerance and
            metrics['support_force_z'] >= bundle['inertial']['mass']*9.81*.05 and
            not metrics['floor_contact'])


def run_drop(model, bundle, initial, settings=TrialSettings(), record=True, progress=None,
             reference=None):
    from .model import set_state
    data = set_state(model, initial)
    dt = float(model.opt.timestep)
    steps = round(settings.duration/dt)
    if not math.isclose(steps*dt, settings.duration, abs_tol=1e-10):
        raise ValueError('Duration must be an integer multiple of timestep')
    final = measure(model, data, bundle)
    if reference is not None:
        final = reference_metrics(final, data.qpos[:3], data.qpos[3:], bundle, reference)
    invalid = 'intersecting_initial_state' if max(final['penetration'], final['floor_penetration']) > 1e-8 else None
    held_since = None
    max_penetration = final['penetration']
    touched_floor = final['floor_contact']
    trace = []
    stride = max(1, round(settings.log_interval/dt))
    progress_stride = max(1, round(.1/dt))
    def log():
        trace.append([float(data.time), *data.qpos.copy().tolist(), *data.qvel.copy().tolist()])
    if record:
        log()
    for step in range(steps if invalid is None else 0):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or np.any(data.warning.number):
            invalid = 'solver_warning_or_nonfinite_state'
            break
        # mj_step integrates after contact evaluation: synchronize all diagnostics.
        mujoco.mj_forward(model, data)
        final = measure(model, data, bundle)
        if reference is not None:
            final = reference_metrics(final, data.qpos[:3], data.qpos[3:], bundle, reference)
        max_penetration = max(max_penetration, final['penetration'])
        touched_floor |= final['floor_contact']
        accepted = (meets_target(final, bundle, settings) if reference is None else
                    meets_reference(final, bundle, settings, reference))
        if accepted:
            if held_since is None:
                held_since = float(data.time)
        else:
            held_since = None
        if record and ((step+1) % stride == 0 or step+1 == steps):
            log()
        if progress and (step+1) % progress_stride == 0:
            progress(f"Simulated {data.time:.2f}/{settings.duration:g} s; "
                     f"gap {final['max_seating_gap']*1000:.3f} mm")
    if reference is None and max_penetration > bundle['max_penetration'] and invalid is None:
        invalid = 'excessive_penetration'
    if np.any(data.warning.number):
        invalid = 'solver_warning_or_nonfinite_state'
    held_for = 0.0 if held_since is None else float(data.time)-held_since
    status, reason = classify(final, held_for, bundle, settings, invalid, touched_floor)
    if reference is not None and status not in ('success', 'invalid', 'missed_receiver'):
        moving = (final['linear_speed'] > settings.linear_speed_tolerance or
                  final['angular_speed'] > settings.angular_speed_tolerance)
        status = ('unsettled_timeout' if moving or meets_reference(final, bundle, settings, reference)
                  else 'reference_mismatch')
    result = {'schema_version': 1, 'status': status, 'invalid_reason': reason,
              'success_mode': 'flush' if reference is None else 'reference',
              'asset_hash': bundle['asset_hash'], 'geometry_source': bundle['source'],
              'initial_state': initial, 'settings': asdict(settings),
              'physics': {'timestep': dt, 'friction': float(model.geom_friction[-1,0]),
                          'solref': model.geom_solref[-1].tolist(), 'solimp': model.geom_solimp[-1].tolist(),
                          'integrator': 'implicitfast', 'solver': 'Newton', 'iterations': int(model.opt.iterations),
                          'cone': 'elliptic', 'condim': 3, 'margin': 0.0},
              'versions': {'aerial_assembly': __version__, 'python': platform.python_version(),
                           'mujoco': mujoco.__version__, 'numpy': np.__version__},
              'final': final, 'elapsed_simulation_time': float(data.time),
              'max_penetration': max_penetration, 'touched_floor': bool(touched_floor),
              'final_target_dwell': held_for,
              'settling_time': held_since if status == 'success' else None,
              'warnings': data.warning.number.tolist()}
    return result, np.asarray(trace)


def collision_query(model, data):
    """Update contacts without allocating or solving constraint forces."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_collision(model, data)


def socket_probe_checks(bundle):
    """Independent tiny-sphere queries detect filled bores, missing walls/floors."""
    from .model import build_model
    _, xml = build_model(bundle)
    root = ET.fromstring(xml)
    world = root.find('worldbody')
    world.remove(world.find("body[@name='upper']"))
    body = ET.SubElement(world, 'body', name='probe')
    ET.SubElement(body, 'freejoint', align='false')
    ET.SubElement(body, 'inertial', mass='.001', pos='0 0 0', diaginertia='1e-9 1e-9 1e-9')
    epsilon = min(1e-6, bundle['clearance']/100)
    ET.SubElement(body, 'geom', name='probe', type='sphere', size=str(epsilon), contype='2', conaffinity='1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    probe = model.geom('probe').id
    tests, failures = 0, []
    delta = bundle['clearance']*.2
    for idx, socket in enumerate(bundle['sockets']):
        R = rotation(socket.get('quat', [1,0,0,0]))
        locations = [('center', [0,0,-socket['depth']/2], False),
                     ('floor', [0,0,-socket['depth']-delta], True)]
        if 'radius_profile' in socket:
            profile = socket['radius_profile']
            sections = [(f'profile_{i}', (a[0]+b[0])/2, (a[1]+b[1])/2)
                        for i, (a, b) in enumerate(zip(profile, profile[1:]))]
        else:
            sections = [('bore', (socket['lead_depth']+socket['depth'])/2, socket['throat_radius']),
                        ('lead', socket['lead_depth']/2, (socket['throat_radius']+socket['opening_radius'])/2)]
        for label, d, radius in sections:
            if d <= 0:
                continue
            for a in np.linspace(0,2*np.pi,16,endpoint=False):
                for side, hit in [(-1,False),(1,True)]:
                    r = radius+side*delta
                    locations.append((label, [r*np.cos(a),r*np.sin(a),-d], hit))
        for label, pos, expected in locations:
            data.qpos[:3] = R.apply(pos)+socket['mouth']
            data.qpos[3:7] = [1,0,0,0]
            collision_query(model, data)
            hit = any(probe in (c.geom1,c.geom2) and c.dist <= 0 for c in data.contact)
            tests += 1
            if hit != expected:
                failures.append({'socket':idx,'feature':label,'point':np.asarray(pos).tolist(),'expected_material':expected})
    return {'passed': not failures, 'queries': tests, 'failures': failures}


def validate_geometry(model, bundle, samples=81):
    """Collision queries only: target and an aligned insertion path, without gravity."""
    from .model import initial_state
    from .config import Release
    target = [*bundle['target']['pos'], *bundle['target']['quat']]
    data = mujoco.MjData(model)
    data.qpos[:] = target
    collision_query(model, data)
    metric = {**pose_metrics(bundle, data.qpos[:3], data.qpos[3:7]),
              **contact_metrics(model, data, forces=False)}
    start = initial_state(bundle, Release(height=0))['qpos'][:3]
    max_path = 0.0
    worst_fraction = 0.0
    for t in np.linspace(0, 1, samples):
        data.qpos[:3] = (1-t)*np.asarray(start) + t*np.asarray(target[:3])
        collision_query(model, data)
        penetration = contact_metrics(model, data, forces=False)['penetration']
        if penetration > max_path:
            max_path, worst_fraction = penetration, float(t)
    probes = socket_probe_checks(bundle)
    passed = (probes['passed'] and metric['legs_inside'] and metric['max_depth_error'] <= 1e-8 and
              metric['max_seating_gap'] <= 1e-8 and max_path <= bundle['max_penetration'])
    return {'passed': passed, 'target': metric, 'socket_probes': probes, 'insertion_path_max_penetration': max_path,
            'worst_path_fraction': worst_fraction, 'path_samples': samples,
            'limitation': 'Discrete aligned path check; does not certify all surface errors or tilted insertion.'}
