"""Offline plan checks (no radio): every waypoint inside the geofence, speed/acceleration limits.

Frames are resolved at their nominal poses (fixture slot, lattice seat), which is what the
camera should report within a few millimetres; the margin in the fence covers the rest.
"""
import math

import numpy as np

from . import poly7
from .frames import Pose


class PlanError(ValueError):
    pass


def _targets(plan):
    """[(brick, phase, op, world target (3,), yaw, duration)] in mission order at nominal poses."""
    out = []
    home = np.asarray(plan['home']['pos'], float)
    s = plan['settings']
    out.append((None, 'TAKEOFF', 'takeoff', np.array([home[0], home[1], s['takeoff_z']]), 0., s['takeoff_s']))
    for b in plan['bricks']:
        for frame, segs in ((Pose.from_dict(b['pick']['slot_nominal']), b['pick']['segments']),
                            (Pose.from_dict(b['place']['frame']['nominal']), b['place']['segments'])):
            for seg in segs:
                if 'to' not in seg:
                    continue
                pos = frame.apply(seg['to'])
                if 'z_world' in seg:
                    pos[2] = seg['z_world']
                out.append((b['name'], seg['phase'], seg['op'], pos, frame.yaw + seg.get('yaw', 0.), seg['duration']))
    out.append((None, 'LAND', 'go_to', np.array([home[0], home[1], s['transit_z']]), 0., s['return_s']))
    out.append((None, 'LAND', 'land', home, 0., s['land_s']))
    return out


def check_plan(plan, line_speed_max=0.2):
    """Raise PlanError listing every violation; return a summary dict when the plan is clean."""
    s = plan['settings']
    lo, hi = np.asarray(s['geofence']['min']), np.asarray(s['geofence']['max'])
    v_max, a_max = s['limits']['v_max'], s['limits']['a_max']
    problems = []
    if s['setpoint_hz'] < 10:
        problems.append(f'setpoint_hz {s["setpoint_hz"]} < 10: the firmware watchdog would level the drone')
    prev = None
    peak_v = peak_a = 0.
    total = 0.
    for name, phase, op, pos, yaw, dur in _targets(plan):
        where = f'{name or "-"}/{phase}/{op}'
        if not np.all(np.isfinite(pos)) or not math.isfinite(dur) or dur <= 0:
            problems.append(f'{where}: non-finite target or non-positive duration')
            continue
        if np.any(pos < lo) or np.any(pos > hi):
            problems.append(f'{where}: target {np.round(pos, 3).tolist()} outside geofence')
        total += dur
        if prev is not None:
            if op in ('go_to', 'takeoff', 'land'):
                piece = poly7.plan_7th_order_no_jerk(0., dur, prev, 0., np.zeros(3), 0., np.zeros(3),
                                                     pos, 0., np.zeros(3), 0., np.zeros(3))
                v, a = poly7.limits(piece)
            else:
                v, a = float(np.linalg.norm(pos - prev))/dur, 0.
                if v > line_speed_max:
                    problems.append(f'{where}: streamed segment speed {v:.3f} m/s > {line_speed_max}')
            peak_v, peak_a = max(peak_v, v), max(peak_a, a)
            if v > v_max or a > a_max:
                problems.append(f'{where}: peak speed {v:.2f} m/s / accel {a:.2f} m/s^2 over limits')
        prev = pos
    if problems:
        raise PlanError('Plan failed offline checks:\n  ' + '\n  '.join(problems))
    return {'bricks': len(plan['bricks']), 'segments_s': total, 'peak_speed_mps': peak_v, 'peak_accel_mps2': peak_a,
            'geofence': [lo.tolist(), hi.tolist()], 'transit_z_m': s['transit_z'],
            'flight_time_budget_s': s['battery']['flight_time_s']}
