"""A flight plan for the real cage: the compiled one-brick plan, moved into the Lighthouse frame.

The compiled plan's world has z = 0 at the seat level (top of the base strips) and a fixed
layout (fixture at y = -0.30, base at the origin). The field plan:
* shifts every absolute height by the seat level in the Lighthouse frame (floor + 11.5 mm);
* takes the launch-stand pose measured by ``testflight survey home``;
* drops ``base0`` from ``bases`` so the mission seats on the camera's platform pose (with the
  plan's ``bases`` entry the mission would ignore the camera);
* sets the nominal pick/seat frames to the sensed poses, so the offline checks and the dry run
  check this layout (the mission still re-senses before every pick and place);
* sizes the geofence from ``field.json`` (the cage), or from the layout plus a margin.
"""
import copy
import math
from pathlib import Path

import numpy as np

from flight.frames import Pose

ROOT = Path(__file__).resolve().parents[2]           # the brixzle project
DEFAULTS = {'params': ROOT/'configs/ab.json', 'gripper': ROOT/'configs/gripper-fork-msh2-30.json',
            'fly': ROOT/'configs/fly-msh2-30.json', 'structure': 'tower'}


def compile_plan(params=None, gripper=None, fly=None, sigma_mm=1.5, sigma_deg=1.0):
    """One-brick tower plan (needs CadQuery). The camera sigma widens the mission's seat tolerance."""
    import json
    from dataclasses import replace
    from brixzle import flyassemble as FA, gripper as Gr
    from brixzle.cli import load_params
    p = load_params(str(params or DEFAULTS['params']))
    gp = Gr.params_from_dict(json.loads(Path(gripper or DEFAULTS['gripper']).read_text()))
    cfg = FA.config_from_dict(json.loads(Path(fly or DEFAULTS['fly']).read_text()))
    cfg.camera = replace(cfg.camera, sigma_pos_mm=sigma_mm, sigma_angle_deg=sigma_deg)
    plan, *_ = FA.make_plan(p, gp, DEFAULTS['structure'], cfg, 1)
    return plan


def adapt_plan(plan, sensed, field, floor_z, seat_level, names):
    """Field plan (new dict). ``sensed``: {name: Pose}; ``field``: {'home': [x, y, z], ...};
    ``floor_z(x, y)``: the floor in the Lighthouse frame; ``seat_level``: seat height above the floor."""
    if 'home' not in field:
        raise ValueError('field.json has no "home": run `testflight survey home` with the drone on the stand')
    brick_name, plat_name = names['brick'], names['platform']
    if plat_name not in sensed:
        raise ValueError(f'the camera does not see the platform ({plat_name})')
    plat = sensed[plat_name]
    dz = float(floor_z(*plat.pos[:2])) + seat_level
    new = copy.deepcopy(plan)
    s = new['settings']
    s['transit_z'] += dz
    s['emergency_land_z'] += dz
    for b in new['bricks']:
        for segs in (b['pick']['segments'], b['pick'].get('abort', []), b['place']['segments']):
            for seg in segs:
                if 'z_world' in seg:
                    seg['z_world'] += dz
        if b['name'] == brick_name:
            if brick_name not in sensed:
                raise ValueError(f'the camera does not see the brick ({brick_name})')
            b['pick']['slot_nominal'] = sensed[brick_name].to_dict()
        frame = b['place']['frame']
        if frame['supporter'] == plat_name:
            frame['nominal'] = plat.compose(Pose.from_dict(frame['rel'])).to_dict()
    home = [float(v) for v in field['home']]
    new['home'] = {'pos': home, 'yaw': math.radians(field.get('home_yaw_deg', 0.))}
    s['takeoff_z'] = home[2] + s['planner']['takeoff_mm']*1e-3
    new['bases'] = {k: v for k, v in new['bases'].items() if k != plat_name}
    pts = [home[:2]] + [list(p.pos[:2]) for p in sensed.values()]
    m = field.get('fence_margin_m', 0.25)
    if 'geofence' in field:
        lo_xy, hi_xy = np.asarray(field['geofence']['min'], float), np.asarray(field['geofence']['max'], float)
    else:
        lo_xy, hi_xy = np.min(pts, axis=0) - m, np.max(pts, axis=0) + m
    floor_lo = min(float(floor_z(*p)) for p in pts)
    s['geofence'] = {'min': [*map(float, lo_xy), floor_lo - 0.04], 'max': [*map(float, hi_xy), s['transit_z'] + 0.3]}
    retime(new)
    new['name'] = f'{plan["name"]}:field'
    new['field'] = {'seat_level_lh_m': dz, 'sensed': {k: v.to_dict() for k, v in sensed.items()},
                    'note': 'Lighthouse frame; base0 comes from the camera at every place (not from bases)'}
    new.pop('sim', None)          # the simulator rebuilds the sim layout, not this one
    return new


RETIMED = ('GOTO_PREPICK', 'TRANSIT')     # the legs whose length depends on the layout


def retime(plan):
    """Re-derive the transit legs' durations (and the flight home) from this layout's distances.

    The compiled plan timed them for the simulator's layout; the speed is the planner's v_transit.
    """
    s = plan['settings']
    cfg = s['planner']
    home = np.asarray(plan['home']['pos'], float)
    prev = np.array([home[0], home[1], s['takeoff_z']])
    for b in plan['bricks']:
        for frame, segs in ((Pose.from_dict(b['pick']['slot_nominal']), b['pick']['segments']),
                            (Pose.from_dict(b['place']['frame']['nominal']), b['place']['segments'])):
            for seg in segs:
                if 'to' not in seg:
                    continue
                pos = frame.apply(seg['to'])
                if 'z_world' in seg:
                    pos[2] = seg['z_world']
                if seg['op'] == 'go_to' and seg['phase'] in RETIMED:
                    seg['duration'] = max(cfg['min_go_to_s'], float(np.linalg.norm(pos - prev))/cfg['v_transit'])
                prev = pos
    back = float(np.linalg.norm(prev[:2] - home[:2]))
    s['return_s'] = max(s['return_s'], back/cfg['v_transit'])


def field_problems(plan, sensed, names, field):
    """Layout problems the offline checks cannot see."""
    out = []
    home = np.asarray(plan['home']['pos'][:2])
    for k in (names['brick'], names['platform']):
        if k in sensed and np.linalg.norm(np.asarray(sensed[k].pos[:2]) - home) < 0.15:
            out.append(f'{k} is within 15 cm of the launch stand')
    if names['brick'] in sensed and names['platform'] in sensed:
        d = np.linalg.norm(np.subtract(sensed[names['brick']].pos[:2], sensed[names['platform']].pos[:2]))
        if d < 0.15:
            out.append(f'brick and platform are only {d*100:.0f} cm apart (prop guards span 14 cm)')
    return out
