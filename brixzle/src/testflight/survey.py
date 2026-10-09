"""Lighthouse readings for the calibration: tag survey, launch-stand home, cage corners, check points.

Connects with cflib and only logs the state estimate: it never arms, never sends setpoints. The
drone is placed by hand; each reading waits for the Kalman variance to settle, then averages 2 s.
"""
import math
import time

import numpy as np

from flight.preflight import POSITION_NAMES, VARIANCE_NAMES, samples, wait_for_parameters, wait_for_position

YAW = 'stabilizer.yaw'       # degrees


class Reader:
    def __init__(self, uri):
        from cflib.crazyflie.log import LogConfig
        from flight.clock import RealClock
        from flight.hardware import connect
        self.LogConfig, self.clock = LogConfig, RealClock()
        self.scf = connect(uri)
        self.scf.open_link()
        wait_for_parameters(self.scf, self.clock)

    def read(self, seconds=2.0):
        """(position (3,), yaw_deg, std_mm) once the estimate has settled."""
        cf = self.scf.cf
        names = VARIANCE_NAMES + POSITION_NAMES
        with samples(cf, self.LogConfig, self.clock, names, period_ms=50) as q:
            wait_for_position(cf, q, self.clock)
        pos, yaw = [], []
        with samples(cf, self.LogConfig, self.clock, POSITION_NAMES + (YAW,), period_ms=50) as q:
            end = self.clock.time() + seconds
            while self.clock.time() < end:
                _, data = q.get(timeout=5)
                if isinstance(data, Exception):
                    raise data
                pos.append([float(data[n]) for n in POSITION_NAMES])
                yaw.append(math.radians(float(data[YAW])))
        pos = np.array(pos)
        yaw_deg = math.degrees(math.atan2(np.mean(np.sin(yaw)), np.mean(np.cos(yaw))))
        return pos.mean(axis=0), yaw_deg, float(pos.std(axis=0).max()*1e3)

    def close(self):
        self.scf.close_link()


def survey_tags(reader, ids, size_m, drone_height_m, ask=input, out=print):
    """Drone centred on each tag, nose along the tag's +x (rightwards when the tag reads upright)."""
    tags = {}
    for i in ids:
        ask(f'Put the drone centred on tag {i}, nose towards the tag\'s right edge, then press Enter')
        p, yaw, std = reader.read()
        tags[str(i)] = {'center': [float(p[0]), float(p[1]), float(p[2] - drone_height_m)], 'yaw_deg': yaw}
        out(f'tag {i}: {np.round(p*1e3, 1).tolist()} mm, yaw {yaw:.1f} deg (spread {std:.1f} mm)')
        if std > 2:
            out('  warning: the estimate moved more than 2 mm; check the base stations')
    zs = [t['center'][2] for t in tags.values()]
    if max(zs) - min(zs) > 0.01:
        out(f'warning: tag heights differ by {(max(zs) - min(zs))*1e3:.1f} mm; is the floor or the survey off?')
    return {'dictionary': None, 'size_m': size_m, 'drone_height_m': drone_height_m, 'tags': tags,
            'date': time.strftime('%Y-%m-%d %H:%M:%S')}


def survey_home(reader, ask=input, out=print):
    ask('Put the drone (gripper on) on the launch stand as for take-off, then press Enter')
    p, yaw, std = reader.read()
    out(f'home {np.round(p*1e3, 1).tolist()} mm, yaw {yaw:.1f} deg (spread {std:.1f} mm)')
    return {'home': [float(v) for v in p], 'home_yaw_deg': yaw}


def survey_fence(reader, ask=input, out=print, inset_m=0.15):
    """Two opposite inside corners of the cage, at floor level; the fence is inset from the net."""
    pts = []
    for name in ('first', 'opposite'):
        ask(f'Put the drone in the {name} inside corner of the cage, then press Enter')
        p, _, _ = reader.read()
        pts.append(p[:2])
        out(f'corner {np.round(p*1e3, 1).tolist()} mm')
    lo, hi = np.minimum(*pts) + inset_m, np.maximum(*pts) - inset_m
    return {'geofence': {'min': lo.tolist(), 'max': hi.tolist()}}
