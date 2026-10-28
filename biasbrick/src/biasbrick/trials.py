"""MuJoCo trials for biasbricks, mirroring brixzle.trials and dfbrick.trials: drops and capture sweeps.

A brick is released as a free body above its seat at the aim point (``geometry.aim``: the X catchment
centre, and the one-sided Y offset) plus a sampled pose error, as in brixzle. The scene, settle loop,
error model and tolerances are brixzle's own; the error model's aim_bias is 0 because the aim is
biasbrick's. Course 0 is part A on an anchored base of two socket tops; a brick dropped "on a brick"
is a B onto a seated A. Metrics are simulated values under uncalibrated contact, not hardware rates.
"""
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from brixzle.scene import activate, body_pose, build_scene
from brixzle.trials import Tolerance, TrialConfig, _brick_geoms, run_until_settled  # noqa: F401

from . import cad, geometry as G
from .params import BBParams


def _rot(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _wxyz(r):
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


IDENTITY = np.array([1., 0., 0., 0.])


def bundles(p: BBParams):
    return {k: cad.build_brick(p, part=k) for k in ('A', 'B')}


def release_lift(p: BBParams, sample):
    """Height above the seat that clears everything under the brick at its release offset and tilt:
    the Y sawtooth at dy, an X sawtooth peak, the hanging tooth and the tilted ends; plus the sampled clearance."""
    bp = p.profile()
    tilt = (p.length/2*math.sin(math.radians(abs(sample['pitch'])))
            + (p.D + p.flange)*math.sin(math.radians(abs(sample['roll']))))
    return (float(G.rise(p, p.valley_y + sample['dy'])) + bp.amplitude + bp.tooth_L + bp.slot_extra + tilt
            + sample['clear'])


def release_pose(p: BBParams, seat, sample):
    pos, quat = seat
    rel = np.asarray(pos, float) + np.array([sample['dx'], sample['dy'], release_lift(p, sample)])
    err = Rotation.from_euler('xyz', [sample['roll'], sample['pitch'], sample['yaw']], degrees=True)
    return rel, _wxyz(err*_rot(quat))


def pose_error(data, info, j, seat, ref=None):
    """Position error (mm, in the seat's frame) and angle (deg) of body j against its seat.

    With ``ref`` = (k, ideal pose of k), the seat is carried along with body k's actual pose."""
    pos, quat = body_pose(data, info, j)
    tpos, tquat = np.asarray(seat[0], float), seat[1]
    if ref is not None:
        k, (ipos, iquat) = ref
        rpos, rquat = body_pose(data, info, k)
        delta = _rot(rquat)*_rot(iquat).inv()
        tpos, tquat = rpos + delta.apply(tpos - ipos), _wxyz(delta*_rot(tquat))
    err = _rot(tquat).inv().apply(pos - tpos)
    return err, math.degrees((_rot(tquat).inv()*_rot(quat)).magnitude())


def classify(err, angle, tol: Tolerance, on_floor):
    dx, dy, dz = err
    if on_floor:
        return 'fell'
    if abs(dx) <= tol.seat_xz and abs(dz) <= tol.seat_xz and abs(dy) <= tol.seat_y and angle <= tol.seat_angle:
        return 'seated'
    return 'jammed'


def scene(p: BBParams, bricks, n_bricks, cfg: TrialConfig, record=False):
    """Base receiving a course-0 A at both sockets; pooled bodies A (and B on top)."""
    static = G.base_pieces(p, [p.socket_x(0), p.socket_x(1)], 'A')
    parts = ['A', 'B'][:n_bricks]
    return build_scene(None, bricks, [], n_bricks, cfg.physics, visual=record, body_parts=parts, static=static)


def drop_trials(p: BBParams, bricks, cfg: TrialConfig, samples=32, seed=0, offsets=None, record=False,
                on_brick=False):
    """Drop one brick onto the base (A) or onto an ideally seated A (B) and judge its seat.

    Monte Carlo over ``cfg.error`` around the aim; ``offsets`` instead runs a deterministic sweep, each a
    dict of dx, dy (mm, from the seat) and yaw (deg) with everything else ideal.
    """
    ax, ay = G.aim(p, 'A')  # the base receives A, and a brick on a brick is B on A: both land on A
    support = ((0., 0., 0.), IDENTITY) if on_brick else None
    seat = ((0., 0., p.H0), IDENTITY) if on_brick else ((0., 0., 0.), IDENTITY)
    n = 2 if on_brick else 1
    j = n - 1
    model, xml, info = scene(p, bricks, n, cfg, record)
    data = mujoco.MjData(model)
    owner = _brick_geoms(model, info)
    rng = np.random.default_rng(seed)
    rows = []
    for item in (offsets if offsets is not None else range(samples)):
        mujoco.mj_resetData(model, data)
        if support is not None:
            activate(model, data, info, 0, *support)
            run_until_settled(model, data, info, 0, Tolerance(max_time=0.3), owner)
        sample = cfg.error.sample(rng)
        if offsets is not None:
            sample = {**sample, 'dx': ax, 'dy': ay, 'yaw': 0., 'roll': 0., 'pitch': 0., 'vel': [0., 0., 0.],
                      'clear': cfg.error.clearance[0], **item}
        else:
            sample = {**sample, 'err_dx': sample['dx'], 'err_dy': sample['dy'],
                      'dx': ax + sample['dx'], 'dy': ay + sample['dy']}
        sample['aim'] = [ax, ay]
        pos, quat = release_pose(p, seat, sample)
        activate(model, data, info, j, pos, quat, linvel=sample['vel'])
        frames = [] if record else None
        m = run_until_settled(model, data, info, j, cfg.tolerance, owner, frames=frames, watch=range(j))
        err, angle = pose_error(data, info, j, seat, (0, support) if support is not None else None)
        outcome = classify(err, angle, cfg.tolerance, m['floor'])
        rows.append({**sample, **m, 'outcome': outcome, 'err_mm': err.tolist(), 'angle_deg': angle})
        if record:
            rows[-1]['frames'] = frames
    return rows, xml


def sweep_offsets(p: BBParams, axis, below='A', step=1.0):
    """Offsets for a one-axis capture sweep around the aim: dx across the socket and past its edges,
    dy from over the nest face to past the flange, or yaw over +-30 deg."""
    ax, ay = G.aim(p, below)
    if axis == 'dx':
        vals = np.arange(-p.U/2 - 4, p.U/2 + 16 + 1e-9, step)  # the stroke side (+x) catches past the edge
    elif axis == 'dy':
        vals = np.arange(-4., p.D/2 + p.flange + 4 + 1e-9, step)
    elif axis == 'yaw':
        vals = np.arange(-30., 30. + 1e-9, step)
    else:
        raise ValueError(f'unknown sweep axis {axis!r}')
    return [{axis: float(v)} for v in vals]


def capture_band(rows, axis, around):
    """[min, max] of the contiguous seated band through ``around`` along one sweep axis (None if it fails there)."""
    vals = sorted((r[axis], r['outcome'] == 'seated') for r in rows)
    mid = min(range(len(vals)), key=lambda i: abs(vals[i][0] - around))
    if not vals[mid][1]:
        return None
    lo = hi = mid
    while lo > 0 and vals[lo - 1][1]:
        lo -= 1
    while hi < len(vals) - 1 and vals[hi + 1][1]:
        hi += 1
    return [vals[lo][0], vals[hi][0]]


def map_offsets(dxs, dys):
    """A dx x dy grid of offsets (mm, from the seat) for a 2D capture map."""
    return [{'dx': float(x), 'dy': float(y)} for x in dxs for y in dys]


def ring_score(rows, aim, radius, n=36):
    """Seated fraction of a capture map's cells hit by a ring of ``radius`` around ``aim`` (nearest cell;
    off the map counts as a miss): the success rate a fixed-radius error would see, other errors ideal."""
    cells = {(r['dx'], r['dy']): r['outcome'] == 'seated' for r in rows}
    xs, ys = sorted({k[0] for k in cells}), sorted({k[1] for k in cells})
    step = min(xs[1] - xs[0], ys[1] - ys[0])
    hits = 0
    for a in np.linspace(0, 2*math.pi, n, endpoint=False):
        x, y = aim[0] + radius*math.cos(a), aim[1] + radius*math.sin(a)
        if not (xs[0] - step/2 <= x <= xs[-1] + step/2 and ys[0] - step/2 <= y <= ys[-1] + step/2):
            continue
        cx = min(xs, key=lambda v: abs(v - x))
        cy = min(ys, key=lambda v: abs(v - y))
        hits += cells[(cx, cy)]
    return hits/n


def best_aim(rows, radius):
    """The map cell that maximises ``ring_score`` for a fixed-radius error, and its score."""
    best = max(((r['dx'], r['dy']) for r in rows), key=lambda a: (ring_score(rows, a, radius), -abs(a[0]), -abs(a[1])))
    return best, ring_score(rows, best, radius)


def structure_scene(p: BBParams, bricks, structure, order, cfg: TrialConfig, record=False):
    """Base under course 0 and one pooled body per placement, A or B by course."""
    from . import lattice as L
    static = G.base_pieces(p, L.base_sites(p, structure), 'A')
    return build_scene(None, bricks, [], len(order), cfg.physics, visual=record,
                       body_parts=[b.part for b in order], static=static)


def assemble(p: BBParams, bricks, structure, cfg: TrialConfig, seed=0, stop_on_fail=True, record=None,
             place=False):
    """Drop bricks in order at the aim plus sampled errors; judge each against its seat on its supporter.

    ``place`` instead sets each brick at its seat, at rest (an ideal carried placement, as a drone that
    lowers the brick on its fork rather than dropping it): it measures what the structure holds, apart
    from what a drop can seat.

    Collapse (as brixzle and dfbrick): a placed brick moves more than ``tolerance.drift`` while the new
    one settles, any brick touches the floor, or a brick sags more than H0/2 from its lattice seat.
    """
    from . import lattice as L
    order = L.assembly_order(structure)
    index = {b.name: j for j, b in enumerate(order)}
    model, xml, info = structure_scene(p, bricks, structure, order, cfg, record is not None)
    data = mujoco.MjData(model)
    owner = _brick_geoms(model, info)
    rng = np.random.default_rng(seed)
    ax, ay = G.aim(p, 'A')
    stages, status = [], 'complete'
    for j, b in enumerate(order):
        before = [body_pose(data, info, i) for i in range(j)]
        sample = cfg.error.sample(rng)
        if b.course < cfg.error.exact_courses:
            sample = {**sample, 'dx': 0., 'dy': 0., 'yaw': 0., 'roll': 0., 'pitch': 0., 'vel': [0., 0., 0.],
                      'clear': cfg.error.clearance[0]}
        sample = {**sample, 'err_dx': sample['dx'], 'err_dy': sample['dy'], 'dx': ax + sample['dx'],
                  'dy': ay + sample['dy'], 'aim': [ax, ay]}
        sup = L.supporter(structure, b)
        ref = (index[sup.name], L.pose(p, sup)) if sup is not None else None
        seat = L.pose(p, b)
        if place:
            sample = {'placed': True}
            activate(model, data, info, j, *seat)
        else:
            pos, quat = release_pose(p, seat, sample)
            activate(model, data, info, j, pos, quat, linvel=sample['vel'])
        m = run_until_settled(model, data, info, j, cfg.tolerance, owner, watch=range(j), frames=record)
        err, angle = pose_error(data, info, j, seat, ref)
        outcome = classify(err, angle, cfg.tolerance, m['floor'])
        moved, sag = 0., 0.
        for i, prev in enumerate(order[:j]):
            pos_i, quat_i = body_pose(data, info, i)
            turn = math.degrees((_rot(before[i][1]).inv()*_rot(quat_i)).magnitude())
            moved = max(moved, float(np.linalg.norm(pos_i - before[i][0])), turn*p.length/2*math.pi/180)
            sag = max(sag, float(np.linalg.norm(pose_error(data, info, i, L.pose(p, prev))[0])))
        collapsed = moved > cfg.tolerance.drift or sag > p.H0/2 or m['floor']
        stages.append({'brick': b.name, 'sample': sample, **m, 'outcome': outcome, 'err_mm': err.tolist(),
                       'angle_deg': angle, 'stage_motion_mm': moved, 'sag_mm': sag, 'collapsed': bool(collapsed)})
        if collapsed:
            status = 'collapse'
            break
        if outcome != 'seated' and stop_on_fail:
            status = f'placement_{outcome}'
            break
    stable = sum(1 for s in stages if s['outcome'] == 'seated' and not s['collapsed'])
    return {'structure': structure.name, 'status': status, 'placed': len(stages), 'total': len(order),
            'stable_bricks': stable, 'max_sag_mm': max([s['sag_mm'] for s in stages], default=0.),
            'stages': stages}, (model, data, info, xml)


def push_test(p: BBParams, bricks, structure, cfg: TrialConfig, rate=0.2, limit=3.0, direction=(1, 0, 0)):
    """Ideal assembly, then a ramped horizontal force on the top brick until it moves (as brixzle)."""
    from brixzle.trials import ErrorModel
    ideal = TrialConfig(cfg.physics, ErrorModel(ideal=True, aim_bias=0.), cfg.tolerance)
    result, (model, data, info, _) = assemble(p, bricks, structure, ideal)
    if result['status'] != 'complete':
        return {'status': 'assembly_failed', 'break_force_N': 0., 'assembly': result['status']}
    top = len(info['bodies']) - 1
    body = info['bodies'][top]
    start, _ = body_pose(data, info, top)
    d = np.asarray(direction, float)/np.linalg.norm(direction)
    force = 0.
    while force < limit:
        force += rate*model.opt.timestep
        data.xfrc_applied[body, :3] = force*d
        mujoco.mj_step(model, data)
        pos, _ = body_pose(data, info, top)
        if np.linalg.norm(pos - start) > cfg.tolerance.drift:
            return {'status': 'moved', 'break_force_N': float(force)}
    return {'status': 'held', 'break_force_N': float(force)}
