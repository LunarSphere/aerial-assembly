"""Survey in the Lighthouse world: metres, Z up, quaternion order wxyz."""
import math

import numpy as np

from ..frames import Pose
from .artifacts import VERSION, read, write

ROLES = ('origin', 'x', 'xy', 'check', 'check', 'check')


def vector(value, name, n=3):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError(f'{name}: expected {n} finite numbers')
    return a


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name}: expected a finite positive number')
    return value


def pose(value, name):
    vector(value['pos'], name)
    q = vector(value['quat'], name + ' quaternion', 4)
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-6):
        raise ValueError(f'{name}: quaternion must have unit norm')
    return Pose.from_dict(value)


def validate(rig):
    if rig.get('version') != VERSION or rig.get('family') != '25h9':
        raise ValueError('Expected survey version 1 with AprilTag family 25h9')
    tags = rig['tags']
    ids = [t['id'] for t in tags]
    if (len(tags) < 5 or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids)
            or not {0, 1, 2}.issubset(ids)):
        raise ValueError('Supply unique tags 0, 1, 2 and at least two check tags')
    roles = [t['role'] for t in tags]
    if any(roles.count(r) != 1 for r in ('origin', 'x', 'xy')) or any(r not in ROLES for r in roles):
        raise ValueError('Assign one origin, one x, one xy and at least two check tags')
    for t in tags:
        positive(t['size_m'], 'black-square side length')
        p = pose(t['pose'], f'tag {t["id"]}')
        if np.dot(p.rotate([0, 0, 1]), [0, 0, 1]) < .999:
            raise ValueError('Tags must lie flat, printed face up; enter yaw only')
    refs = {t['role']: t for t in tags if t['role'] != 'check'}
    # Support offset is sample reference minus tag centre at zero yaw, with level drone.
    offset = vector(rig['sample_offset_m'], 'sample offset')
    points = {k: np.asarray(t['pose']['pos']) + offset for k, t in refs.items()}
    placement_references = rig.get('reference_method') == 'human_placements'
    if not placement_references and not np.allclose(points['origin'], [0, 0, 0], atol=1e-6):
        raise ValueError('Origin tag centre + sample offset must equal [0,0,0]')
    if not placement_references and not np.allclose(points['x'], [1, 0, 0], atol=1e-6):
        raise ValueError('X reference sample must be exactly [1,0,0] metres')
    if not placement_references and (abs(points['xy'][2]) > 1e-6 or abs(points['xy'][1]) < .1):
        raise ValueError('XY reference must share Z=0 and be at least 0.1 m off the X axis')
    vector(rig['body_to_sample_m'], 'body-to-sample offset')
    lo = vector(rig['envelope_body_m']['min'], 'drone/gripper envelope min')
    hi = vector(rig['envelope_body_m']['max'], 'drone/gripper envelope max')
    if np.any(lo >= hi) or np.any(lo > 0) or np.any(hi < 0):
        raise ValueError('Body envelope must enclose the body origin and have positive dimensions')
    fence_lo = vector(rig['geofence']['min'], 'geofence min')
    fence_hi = vector(rig['geofence']['max'], 'geofence max')
    if np.any(fence_lo >= fence_hi):
        raise ValueError('Geofence min must be smaller than max')
    vector(rig['home']['pos'], 'launch body position')
    if not math.isfinite(float(rig['home'].get('yaw', 0))):
        raise ValueError('Nonfinite launch yaw')
    if abs(rig['home'].get('yaw', 0)) > 1e-9:
        raise ValueError('First hover experiment requires launch yaw=0')
    if not math.isfinite(float(rig['emergency_land_body_z_m'])):
        raise ValueError('Supply measured emergency landing body height in the Lighthouse world')
    positive(rig['mass_kg'], 'measured total mass')
    if rig['mass_kg'] > .15:
        raise ValueError('Mass must be in kg, for the unloaded drone with battery/deck/gripper')
    if len(rig['base_stations']) != 2 or len(set(rig['base_stations'])) != 2 or any(
            type(v) is not int or not 0 <= v < 16 for v in rig['base_stations']):
        raise ValueError('Supply two distinct zero-based Lighthouse station IDs (channel minus one)')
    return rig


def load(path):
    return validate(read(path))


def corners(tag):
    """OpenCV decoded corner order: TL, TR, BR, BL, printed face +Z."""
    s = tag['size_m'] / 2
    p = Pose.from_dict(tag['pose'])
    return np.array([p.apply(v) for v in [(-s, s, 0), (s, s, 0), (s, -s, 0), (-s, -s, 0)]])


def survey(path, ask=input, layout_path=None, tag_ids=None, measure_checks=False):
    """Ask for physical measurements; derive coordinate offsets internally."""
    from pathlib import Path
    if Path(path).exists():
        raise ValueError(f'{path} already exists; choose a new survey filename')
    layout = read(layout_path) if layout_path else None
    if layout is not None:
        if layout.get('kind') != 'tag-layout' or layout.get('family') != '25h9':
            raise ValueError('Expected an estimated 25h9 tag layout')
        baseline = positive(layout['reference_distance_m'], 'estimated baseline')
        estimated = {t['id']: t for t in layout['tags']}
        tag_ids = sorted(estimated)
        if not {0, 1, 2}.issubset(tag_ids) or len(tag_ids) < 5:
            raise ValueError('Layout must contain the three reference tags and at least two check tags')
        print(f'Loaded camera estimates for tags {tag_ids}; estimated 0-to-1 distance: {baseline*100:.2f} cm.')
        print('These are initial estimates. Measure the actual 0-to-1 distance and ensure it is exactly 100 cm.')
        print('If you move a tag, recapture the layout. Check tags need independent measurements before flight acceptance.')
    else:
        estimated = {}
        tag_ids = sorted(tag_ids if tag_ids is not None else range(6))
    if len(set(tag_ids)) != len(tag_ids) or len(tag_ids) < 5 or not {0, 1, 2}.issubset(tag_ids):
        raise ValueError('Choose tags 0, 1, 2 and at least two unique check tags')

    def number(label, unit, default=None, positive_only=False):
        suffix = f' [default {default:g}]' if default is not None else ''
        while True:
            answer = ask(f'{label} ({unit}){suffix}: ').strip()
            try:
                value = float(answer) if answer else (float(default) if default is not None else math.nan)
                if not math.isfinite(value) or (positive_only and value <= 0):
                    raise ValueError
                return value
            except ValueError:
                print('Enter one ' + ('positive ' if positive_only else '') + f'number in {unit}; for example ' +
                      ('60' if positive_only else '0') + '. Measure this distance; no coordinate calculation is needed.')

    def yes(label):
        while True:
            answer = ask(label + ' [Y/n]: ').strip().lower()
            if answer in ('', 'y', 'yes'):
                return True
            if answer in ('n', 'no'):
                return False
            print('Press Enter for yes, or type n for no.')

    print('STEP 1 — Choose the origin, like cfclient')
    print(f'Lay the {len(tag_ids)} AprilTags {tag_ids} flat on the same level surface.')
    print('Put the drone on its support above tag 0, with the CENTRE OF THE LIGHTHOUSE DECK')
    print('directly above the CENTRE OF TAG 0. The fork must hang clear of the surface.')
    print('This placement will define the Lighthouse origin. Keep the drone level.')
    print('Measure straight UP from the printed tag surface to the middle of the four')
    print('small light sensors on the TOP Lighthouse deck. Include the support height.')
    height = number('Height of the Lighthouse sensors above the tag surface', 'mm', positive_only=True) / 1000
    data = dict(version=VERSION, family='25h9', tags=[], sample_offset_m=[0., 0., height])

    print('STEP 2 — Mark the positive X direction')
    print('Place tag 1 exactly 100 cm from tag 0, centre to centre. That direction is +X.')
    print('For Lighthouse measurement you will move the SAME support and drone from tag 0')
    print('to tag 1, keeping the same height. Face the nose from tag 0 toward tag 1.')
    print('Looking along that direction: +Y is LEFT, and -Y is RIGHT.')
    same_size = yes('Are all the printed tag squares the same size?')
    shared_size = None
    if same_size:
        default_size = float(np.median([t['size_m'] for t in estimated.values()]))*1000 if estimated else None
        shared_size = number('Side of the printed BLACK square, excluding the white margin (verify with ruler)',
                             'mm', default=default_size, positive_only=True) / 1000

    print('STEP 3 — Record the tag layout with a tape measure')
    print('Tag 0 and tag 1 positions are filled in automatically.')
    print('For other tags, measure forward/back and left/right from tag 0 along these axes.')
    print('Use negative numbers for backward or right. All tags stay on the same surface.')
    print('Rotation: 0 degrees means the printed top of the tag points LEFT (+Y),')
    print('and its printed right edge points along +X. Positive rotation is counterclockwise from above.')
    for i in tag_ids:
        role = {0: 'origin', 1: 'x', 2: 'xy'}.get(i, 'check')
        print(f'Tag {i}: ' + {'origin': 'origin', 'x': '1 metre along +X',
                              'xy': 'off the X-axis, to define the floor plane',
                              'check': 'independent camera check and hover target'}[role])
        estimate = estimated.get(i)
        independently_measured = not estimated or (measure_checks and role == 'check')
        size = shared_size if same_size else number('Black-square side', 'mm', positive_only=True) / 1000
        if estimate is not None and measure_checks and role == 'check':
            print('Independently measure this check tag; do not copy camera estimates.')
            size = number('Measured black-square side', 'mm', positive_only=True) / 1000
        if role == 'origin':
            x, y = 0., 0.
        elif role == 'x':
            x, y = 1., 0.
        elif estimate is not None and not independently_measured:
            x, y = np.asarray(estimate['pose']['pos'])[:2] / baseline
        else:
            x = number(f'Tag {i}: forward from tag 0 (+), or backward (-)', 'cm') / 100
            while True:
                y = number(f'Tag {i}: left from the X-axis (+), or right (-)', 'cm') / 100
                if role != 'xy' or abs(y) >= .1:
                    break
                print('For tag 2, choose a spot at least 10 cm to either side of the X-axis.')
        default_yaw = math.degrees(Pose.from_dict(estimate['pose']).yaw) if estimate else 0
        yaw = math.radians(default_yaw if estimate is not None else number(f'Tag {i}: rotation', 'degrees', default=0))
        data['tags'].append(dict(id=i, role=role, size_m=size,
                                 pose=dict(pos=[float(x), float(y), -height], quat=[math.cos(yaw/2), 0, 0, math.sin(yaw/2)]),
                                 survey_source='manual' if independently_measured else layout['source'],
                                 independently_verified=independently_measured))
        print(f'Tag {i}: X={x*100:.1f} cm, Y={y*100:.1f} cm, rotation={math.degrees(yaw):.1f} degrees.')

    print('STEP 4 — Measure the drone with the fork attached')
    print('The motor plane is the level passing through the four propeller hubs.')
    print('The following small distances locate the deck sensors relative to that plane.')
    print('If the deck is centred left/right and front/back, accept 0 for those two distances.')
    dx = number('Deck sensor centre forward of the motor-plane centre (+), or behind (-)', 'mm', default=0) / 1000
    dy = number('Deck sensor centre left of the motor-plane centre (+), or right (-)', 'mm', default=0) / 1000
    dz = number('Deck sensor centre ABOVE the propeller-hub plane', 'mm', positive_only=True) / 1000
    data['body_to_sample_m'] = [dx, dy, dz]
    if yes('Use the origin support (tag 0) as the launch and return stand?'):
        data['home'] = dict(pos=[-dx, -dy, -dz], yaw=0.)
    else:
        print('Centre the Lighthouse deck over the chosen launch point, with the nose along +X.')
        x = number('Launch point: forward from tag 0 (+), or backward (-)', 'cm') / 100
        y = number('Launch point: left from the X-axis (+), or right (-)', 'cm') / 100
        z = number('Height of the deck sensors above the tag surface at launch', 'mm', positive_only=True) / 1000
        data['home'] = dict(pos=[x-dx, y-dy, z-height-dz], yaw=0.)
    print('Place the assembled drone gently on the clear surface it could land on in an emergency.')
    ez = number('Propeller-hub height above the TAG surface in that resting position', 'mm', positive_only=True) / 1000
    data['emergency_land_body_z_m'] = ez - height
    data['mass_kg'] = number('Total drone + battery + Lighthouse deck + fork, with NO brick', 'grams', positive_only=True) / 1000
    print('Measure the furthest point in each direction from the motor-plane centre.')
    print('Include prop guards, legs, fork and steel pin tips. Enter positive distances.')
    front = number('Furthest point FORWARD', 'mm', positive_only=True) / 1000
    back = number('Furthest point BACKWARD', 'mm', positive_only=True) / 1000
    left = number('Furthest point LEFT', 'mm', positive_only=True) / 1000
    right = number('Furthest point RIGHT', 'mm', positive_only=True) / 1000
    below = number('Lowest point BELOW the propeller-hub plane', 'mm', positive_only=True) / 1000
    above = number('Highest point ABOVE the propeller-hub plane', 'mm', positive_only=True) / 1000
    data['envelope_body_m'] = dict(min=[-back, -right, -below], max=[front, left, above])

    print('STEP 5 — Mark the clear flight area')
    print('Give the edges of an obstacle-free rectangle relative to tag 0.')
    print('The whole drone and fork must fit inside it, including a 2 cm margin.')
    xmin = number('Back edge along X (negative if behind tag 0)', 'cm') / 100
    xmax = number('Front edge along X', 'cm') / 100
    ymin = number('Right edge along Y (negative if right of tag 0)', 'cm') / 100
    ymax = number('Left edge along Y', 'cm') / 100
    ceiling = number('Maximum clear height above the tag surface', 'cm', positive_only=True) / 100
    data['geofence'] = dict(min=[xmin, ymin, -height-.02], max=[xmax, ymax, ceiling-height])
    while True:
        answer = ask('Lighthouse base-station CHANNELS (two numbers, e.g. 1 2): ')
        try:
            channels = [int(v) for v in answer.replace(',', ' ').split()]
            if len(channels) != 2 or len(set(channels)) != 2 or any(not 1 <= v <= 16 for v in channels):
                raise ValueError
            data['base_stations'] = [v-1 for v in channels]
            break
        except ValueError:
            print('Enter two different channel numbers between 1 and 16, e.g. 1 2.')
    validate(data)
    write(path, data)
    print('Survey saved. Coordinate conversions were calculated for you.')
    print('Next run flight-cal lighthouse with this rig file. It will prompt you to place')
    print('the drone at tag 0, then tag 1 (1 m along +X), then tag 2 (off the X-axis).')
    return data
