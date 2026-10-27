"""MuJoCo trials for DF bricks, mirroring brixzle.trials: drops, capture sweeps, assembly, push.

A brick is released as a free body at a sampled pose error above its seat, as in brixzle. The
scene, settle loop, error model and tolerances are brixzle's own (reused, not copied); what
changes is the lattice (square voxels, vertical insertion) and the base (receivers as static
convex pieces). Metrics are simulated values under uncalibrated contact, not hardware rates.
"""
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from brixzle.scene import activate, body_pose, build_scene
from brixzle.trials import ErrorModel, Tolerance, TrialConfig, _brick_geoms, run_until_settled  # noqa: F401

from . import geometry as G
from . import lattice as L
from .params import DFParams
from .structures import Structure


def _rot(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _wxyz(r):
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def release_lift(p: DFParams, clear):
    """Foot tips start ``clear`` above the ridges of same-course neighbours (the highest reach)."""
    return p.H0 + p.guide_h + clear


def release_pose(p: DFParams, b, sample, seat_pose=None):
    pos, quat = seat_pose if seat_pose is not None else L.pose(p, b)
    rel = pos + np.array([sample['dx'], sample['dy'], release_lift(p, sample['clear'])])
    err = Rotation.from_euler('xyz', [sample['roll'], sample['pitch'], sample['yaw']], degrees=True)
    return rel, _wxyz(err*_rot(quat))


def expected_seat(p, data, info, b, ref):
    """Seat of b carried along with supporter ref = (k, placement_k)'s actual pose."""
    k, rp = ref
    tpos, tquat = L.pose(p, b)
    rpos, rquat = body_pose(data, info, k)
    ipos, iquat = L.pose(p, rp)
    delta = _rot(rquat)*_rot(iquat).inv()
    return rpos + delta.apply(tpos - ipos), _wxyz(delta*_rot(tquat))


def pose_error(p, data, info, j, b, ref=None):
    """World position error (mm) and angle (deg) of body j against its seat (on its supporter)."""
    pos, quat = body_pose(data, info, j)
    tpos, tquat = expected_seat(p, data, info, b, ref) if ref is not None else L.pose(p, b)
    angle = math.degrees((_rot(tquat).inv()*_rot(quat)).magnitude())
    return pos - tpos, angle


def classify(p: DFParams, err, angle, tol: Tolerance, on_floor):
    dx, dy, dz = err
    if on_floor:
        return 'fell'
    if max(abs(dx), abs(dy)) <= tol.seat_xz and abs(dz) <= tol.seat_xz and angle <= tol.seat_angle:
        return 'seated'
    # Seated one or more voxels over: a neighbouring receiver caught it.
    near = [abs(v - p.U*round(v/p.U)) for v in (dx, dy)]
    if max(near) <= 2*tol.seat_xz and max(abs(dx), abs(dy)) > p.U/2 and abs(dz) <= 2*tol.seat_xz \
            and angle <= tol.seat_angle:
        return 'wrong_receiver'
    return 'jammed'


def scene(p: DFParams, brick, structure: Structure, n_bricks, cfg: TrialConfig, record=False, extra=()):
    static = G.base_pieces(p, L.base_receivers(p, structure, extra))
    model, xml, info = build_scene(None, {'DF': brick}, [], n_bricks, cfg.physics, visual=record,
                                   body_parts=['DF']*n_bricks, static=static)
    return model, xml, info


def _ring_cells(p: DFParams, structure, pad=1):
    """Base cells around course 0, so a missed drop meets a neighbouring receiver rather than a void."""
    own = [c for b in structure.bricks if b.course == 0 for c in L.cells(p, b)]
    out = set()
    for i, j in own:
        for di in range(-pad, pad + 1):
            for dj in range(-pad, pad + 1):
                out.add((i + di, j + dj))
    return sorted(out - set(own))


def drop_trials(p: DFParams, brick, cfg: TrialConfig, samples=32, seed=0, offsets=None, record=False,
                on_brick=False):
    """Drop one brick onto the base (or onto an ideally seated brick) and judge its seat.

    Monte Carlo over ``cfg.error``; ``offsets`` instead runs a deterministic sweep, each a dict
    of dx, dy (mm) and yaw (deg) with everything else ideal.
    """
    support = L.Placement(0, 0) if on_brick else None
    target = L.Placement(1, 0) if on_brick else L.Placement(0, 0)
    n = 2 if on_brick else 1
    j = n - 1
    base = Structure('drop', [L.Placement(0, 0)])
    model, xml, info = scene(p, brick, base, n, cfg, record, _ring_cells(p, base))
    data = mujoco.MjData(model)
    owner = _brick_geoms(model, info)
    rng = np.random.default_rng(seed)
    rows = []
    for item in (offsets if offsets is not None else range(samples)):
        mujoco.mj_resetData(model, data)
        if support is not None:
            activate(model, data, info, 0, *L.pose(p, support))
            run_until_settled(model, data, info, 0, Tolerance(max_time=0.3), owner)
        sample = cfg.error.sample(rng)
        if offsets is not None:
            sample = {**sample, 'dx': 0., 'dy': 0., 'yaw': 0., 'roll': 0., 'pitch': 0., 'vel': [0., 0., 0.],
                      'clear': cfg.error.clearance[0], 'aim': 0., **item}
        pos, quat = release_pose(p, target, sample)
        activate(model, data, info, j, pos, quat, linvel=sample['vel'])
        frames = [] if record else None
        m = run_until_settled(model, data, info, j, cfg.tolerance, owner, frames=frames, watch=range(j))
        ref = (0, support) if support is not None else None
        err, angle = pose_error(p, data, info, j, target, ref)
        outcome = classify(p, err, angle, cfg.tolerance, m['floor'])
        rows.append({**sample, **m, 'outcome': outcome, 'err_mm': err.tolist(), 'angle_deg': angle})
        if record:
            rows[-1]['frames'] = frames
    return rows, xml


def sweep_offsets(axis, span, step=1.0):
    """Offsets for a one-axis capture sweep: dx or dy (mm) or yaw (deg) over [-span, span]."""
    return [{axis: float(v)} for v in np.arange(-span, span + 1e-9, step)]


def capture_band(rows, axis):
    """[min, max] of the contiguous seated band through 0 along one sweep axis."""
    vals = sorted((r[axis], r['outcome'] == 'seated') for r in rows)
    zero = min(range(len(vals)), key=lambda i: abs(vals[i][0]))
    if not vals[zero][1]:
        return None
    lo = hi = zero
    while lo > 0 and vals[lo - 1][1]:
        lo -= 1
    while hi < len(vals) - 1 and vals[hi + 1][1]:
        hi += 1
    return [vals[lo][0], vals[hi][0]]


def assemble(p: DFParams, brick, structure: Structure, cfg: TrialConfig, seed=0, stop_on_fail=True,
             record=None):
    """Place bricks in order with sampled errors; judge each against its seat on its supporter.

    Collapse (as brixzle): a placed brick moves more than ``tolerance.drift`` while the new one
    settles, any brick touches the floor, or a brick sags more than H0/2 from its lattice seat.
    """
    order = L.assembly_order(p, structure)
    index = {b.name: j for j, b in enumerate(order)}
    model, xml, info = scene(p, brick, structure, len(order), cfg, record is not None,
                             _ring_cells(p, structure))
    data = mujoco.MjData(model)
    owner = _brick_geoms(model, info)
    rng = np.random.default_rng(seed)
    stages, placed = [], []
    status = 'complete'
    for j, b in enumerate(order):
        before = [body_pose(data, info, i) for i in range(j)]
        sample = cfg.error.sample(rng)
        if b.course < cfg.error.exact_courses:
            sample = dict(dx=0., dy=0., yaw=0., roll=0., pitch=0., clear=cfg.error.clearance[0],
                          vel=[0., 0., 0.], aim=0.)
        sup = L.supporter(p, structure, b)
        ref = (index[sup.name], sup) if sup is not None else None
        seat = expected_seat(p, data, info, b, ref) if cfg.error.track_supporter and ref else None
        pos, quat = release_pose(p, b, sample, seat)
        activate(model, data, info, j, pos, quat, linvel=sample['vel'])
        m = run_until_settled(model, data, info, j, cfg.tolerance, owner, watch=range(j), frames=record)
        err, angle = pose_error(p, data, info, j, b, ref)
        outcome = classify(p, err, angle, cfg.tolerance, m['floor'])
        moved, sag = 0., 0.
        for i, prev in enumerate(placed):
            pos_i, quat_i = body_pose(data, info, i)
            turn = math.degrees((_rot(before[i][1]).inv()*_rot(quat_i)).magnitude())
            moved = max(moved, float(np.linalg.norm(pos_i - before[i][0])), turn*p.length/2*math.pi/180)
            sag = max(sag, float(np.linalg.norm(pose_error(p, data, info, i, prev)[0])))
        collapsed = moved > cfg.tolerance.drift or sag > p.H0/2 or m['floor']
        stages.append({'brick': b.name, 'sample': sample, **m, 'outcome': outcome, 'err_mm': err.tolist(),
                       'angle_deg': angle, 'stage_motion_mm': moved, 'sag_mm': sag,
                       'collapsed': bool(collapsed)})
        placed.append(b)
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


def push_test(p: DFParams, brick, structure: Structure, cfg: TrialConfig, rate=0.2, limit=3.0,
              direction=(1, 0, 0)):
    """Ideal assembly, then a ramped horizontal force on the top brick until it moves (as brixzle)."""
    ideal = TrialConfig(cfg.physics, ErrorModel(ideal=True, aim_bias=0.), cfg.tolerance)
    result, (model, data, info, _) = assemble(p, brick, structure, ideal)
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
