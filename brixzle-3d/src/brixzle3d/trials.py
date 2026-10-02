"""MuJoCo trials: coarse-placement drops, sequential assembly, push test.

No drone dynamics: each brick is released as a free body at a sampled pose
error above its lattice seat. Results are simulated under uncalibrated
contact parameters, not hardware success rates.
"""
from dataclasses import asdict, dataclass, field
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import lattice as L
from .params import BrickParams
from .scene import MM, Physics, activate, body_pose, build_scene


@dataclass(frozen=True)
class ErrorModel:
    """Coarse drone placement: aim at the cavity, let the funnels do the rest."""
    sigma_xy: float = 4.0         # mm, each of X and Y
    clip_xy: float = 12.0
    sigma_yaw: float = 5.0        # deg
    sigma_tilt: float = 3.0       # deg
    clearance: tuple = (5., 30.)  # mm above the highest reach
    sigma_vel: float = 0.03       # m/s
    ideal: bool = False

    def sample(self, rng):
        if self.ideal:
            return dict(dx=0., dy=0., yaw=0., roll=0., pitch=0., clear=self.clearance[0], vel=[0., 0., 0.])
        dx, dy = np.clip(rng.normal(0, self.sigma_xy, 2), -self.clip_xy, self.clip_xy)
        return dict(dx=float(dx), dy=float(dy), yaw=float(rng.normal(0, self.sigma_yaw)),
                    roll=float(rng.normal(0, self.sigma_tilt)), pitch=float(rng.normal(0, self.sigma_tilt)),
                    clear=float(rng.uniform(*self.clearance)), vel=rng.normal(0, self.sigma_vel, 3).tolist())


@dataclass(frozen=True)
class Tolerance:
    seat_xy: float = 1.0
    seat_z: float = 1.0
    seat_angle: float = 3.0
    drift: float = 2.0          # mm a placed brick may move during one stage
    lin_speed: float = 0.002
    ang_speed: float = 0.05
    settle_hold: float = 0.08
    max_time: float = 1.5


@dataclass
class TrialConfig:
    physics: Physics = field(default_factory=Physics)
    error: ErrorModel = field(default_factory=ErrorModel)
    tolerance: Tolerance = field(default_factory=Tolerance)


def _rot(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _quat(r):
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def release_pose(p: BrickParams, b, sample):
    """Seat pose rotated by the sampled error about the brick's footprint centre, then lifted."""
    pos, quat = L.pose(p, b)
    seat = _rot(quat)
    err = Rotation.from_euler('xyz', [sample['roll'], sample['pitch'], sample['yaw']], degrees=True)
    centre_local = np.r_[np.mean(np.array(p.cells, dtype=float) + .5, axis=0)*p.U, p.H/2]
    centre = pos + seat.apply(centre_local)
    rot = err*seat
    lift = p.h + p.peg_L + sample['clear']
    new_pos = centre - rot.apply(centre_local) + np.array([sample['dx'], sample['dy'], lift])
    return new_pos, _quat(rot)


def pose_error(p, data, info, j, placement, ref=None):
    """Error of brick j against its seat, relative to brick ref=(k, placement_k) if given."""
    pos, quat = body_pose(data, info, j)
    tpos, tquat = L.pose(p, placement)
    target = _rot(tquat)
    if ref is not None:
        k, rp = ref
        rpos, rquat = body_pose(data, info, k)
        ipos, iquat = L.pose(p, rp)
        delta = _rot(rquat)*_rot(iquat).inv()
        tpos = rpos + delta.apply(tpos - ipos)
        target = delta*target
    angle = math.degrees((target.inv()*_rot(quat)).magnitude())
    return pos - tpos, angle


def classify(p, err, angle, tol: Tolerance, on_floor):
    if on_floor:
        return 'fell'
    if np.hypot(err[0], err[1]) <= tol.seat_xy and abs(err[2]) <= tol.seat_z and angle <= tol.seat_angle:
        return 'seated'
    nx, ny = err[0]/p.U, err[1]/p.U
    if angle <= tol.seat_angle and abs(err[2]) <= tol.seat_z and max(abs(nx), abs(ny)) > 0.5 \
            and abs(nx - round(nx)) < 0.1 and abs(ny - round(ny)) < 0.1:
        return 'wrong_cell'
    return 'jammed'


def _owners(model, info):
    owner = {}
    for j, b in enumerate(info['bodies']):
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == b:
                owner[g] = j
    return owner


def run_until_settled(model, data, info, j, tol: Tolerance, owner, watch=(), frames=None, fps=60,
                      force_window=0.005):
    """Step until brick j (and watched bricks) are still.

    ``f_max`` is the peak magnitude of the *net* contact force on brick j,
    averaged over ``force_window`` seconds. Summed normal magnitudes would count
    wedge squeeze (equal and opposite face forces) and single-step solver
    spikes, which reflect contact stiffness rather than impact load.
    """
    force = np.zeros(6)
    window = max(1, round(force_window/model.opt.timestep))
    history = []
    t0 = data.time
    still_since = None
    f_max, onsets, in_contact, floor_hit, pen_max = 0., 0, False, False, 0.
    moving = [j, *watch]
    every = max(1, round(1/(fps*model.opt.timestep)))
    step = 0
    while data.time - t0 < tol.max_time:
        mujoco.mj_step(model, data)
        if frames is not None and step % every == 0:
            frames.append(data.qpos.copy())
        step += 1
        net, touching = np.zeros(3), False
        for i in range(data.ncon):
            c = data.contact[i]
            if owner.get(c.geom1) != j and owner.get(c.geom2) != j:
                continue
            touching = True
            mujoco.mj_contactForce(model, data, i, force)
            # Contact-frame force acts on geom2 (frame rows are world axes); flip for geom1.
            world = c.frame.reshape(3, 3).T @ force[:3]
            net += world if owner.get(c.geom2) == j else -world
            pen_max = max(pen_max, -c.dist)
            if info['floor'] in (c.geom1, c.geom2):
                floor_hit = True
        history.append(net)
        if len(history) > window:
            history.pop(0)
        f_max = max(f_max, float(np.linalg.norm(sum(history)/window)))
        if touching and not in_contact:
            onsets += 1
        in_contact = touching
        v = [np.linalg.norm(data.qvel[info['qvel'][m]:info['qvel'][m]+3]) for m in moving]
        w = [np.linalg.norm(data.qvel[info['qvel'][m]+3:info['qvel'][m]+6]) for m in moving]
        if max(v) < tol.lin_speed and max(w) < tol.ang_speed and in_contact:
            still_since = still_since if still_since is not None else data.time
            if data.time - still_since >= tol.settle_hold:
                break
        else:
            still_since = None
    settled = still_since is not None and data.time - still_since >= tol.settle_hold
    return {'f_max': f_max, 'onsets': onsets, 'floor': floor_hit, 'penetration_mm': pen_max/MM,
            'time': (still_since if settled else data.time) - t0, 'settled': settled}


def drop_trials(p: BrickParams, brick, cfg: TrialConfig, samples=32, seed=0, offsets=None, record=False):
    """Drop one brick onto an anchored 5x5 plate; Monte Carlo or a deterministic (dx, dy) sweep."""
    plate = [(i, j) for i in range(-1, 4) for j in range(-1, 4)]
    model, xml, info = build_scene(p, brick, [plate], 1, cfg.physics, visual=record)
    data = mujoco.MjData(model)
    owner = _owners(model, info)
    target = L.Placement(0, 0, 0, 0)
    rng = np.random.default_rng(seed)
    rows = []
    for item in (offsets if offsets is not None else range(samples)):
        mujoco.mj_resetData(model, data)
        sample = cfg.error.sample(rng)
        if offsets is not None:
            dx, dy = item
            sample = dict(dx=float(dx), dy=float(dy), yaw=0., roll=0., pitch=0., clear=cfg.error.clearance[0],
                          vel=[0., 0., 0.])
        pos, quat = release_pose(p, target, sample)
        activate(model, data, info, 0, pos, quat, linvel=sample['vel'])
        frames = [] if record else None
        m = run_until_settled(model, data, info, 0, cfg.tolerance, owner, frames=frames)
        err, angle = pose_error(p, data, info, 0, target)
        row = {**sample, **m, 'outcome': classify(p, err, angle, cfg.tolerance, m['floor']),
               'err_mm': err.tolist(), 'angle_deg': angle}
        if record:
            row['frames'] = frames
        rows.append(row)
    return rows, xml


def summarize_drops(rows):
    outcomes = {}
    for r in rows:
        outcomes[r['outcome']] = outcomes.get(r['outcome'], 0) + 1
    n = len(rows)
    return {
        'n': n, 'p_success': outcomes.get('seated', 0)/n if n else 0., 'outcomes': outcomes,
        'f_max_N': float(np.percentile([r['f_max'] for r in rows], 95)) if rows else 0.,
        't_settle_s': float(np.mean([r['time'] for r in rows])) if rows else 0.,
        'collisions': float(np.mean([max(0, r['onsets'] - 1) + (r['penetration_mm'] > 1.0) for r in rows])) if rows else 0.,
        'max_penetration_mm': float(max(r['penetration_mm'] for r in rows)) if rows else 0.,
    }


def assemble(p: BrickParams, brick, structure, cfg: TrialConfig, seed=0, stop_on_fail=True, record=None):
    """Place bricks in planned order with sampled errors.

    A brick is judged against its seat on its actual primary supporter.
    Collapse = a placed brick moving more than ``drift`` in one stage, floor
    contact, or cumulative sag beyond half a course.
    """
    order = L.assembly_order(p, structure)
    graph = L.connectivity(p, structure)
    index = {b.name: j for j, b in enumerate(order)}
    model, xml, info = build_scene(p, brick, structure.bases, len(order), cfg.physics, visual=record is not None)
    data = mujoco.MjData(model)
    owner = _owners(model, info)
    rng = np.random.default_rng(seed)
    stages, placed, status = [], [], 'complete'
    for j, b in enumerate(order):
        before = [body_pose(data, info, i) for i in range(j)]
        sample = cfg.error.sample(rng)
        pos, quat = release_pose(p, b, sample)
        activate(model, data, info, j, pos, quat, linvel=sample['vel'])
        m = run_until_settled(model, data, info, j, cfg.tolerance, owner, watch=range(j), frames=record)
        sup = sorted((u for u in graph.predecessors(b.name) if u in index),
                     key=lambda u: -graph[u][b.name]['cells'])
        ref = (index[sup[0]], order[index[sup[0]]]) if sup else None
        err, angle = pose_error(p, data, info, j, b, ref)
        outcome = classify(p, err, angle, cfg.tolerance, m['floor'])
        moved, sag = 0., 0.
        for i, prev in enumerate(placed):
            pi, qi = body_pose(data, info, i)
            turn = math.degrees((_rot(before[i][1]).inv()*_rot(qi)).magnitude())
            moved = max(moved, float(np.linalg.norm(pi - before[i][0])), turn*p.U*math.pi/180)
            sag = max(sag, float(np.linalg.norm(pose_error(p, data, info, i, prev)[0])))
        collapsed = moved > cfg.tolerance.drift or sag > p.H/2 or m['floor']
        stages.append({'brick': b.name, 'sample': sample, **m, 'outcome': outcome, 'err_mm': err.tolist(),
                       'angle_deg': angle, 'stage_motion_mm': moved, 'sag_mm': sag, 'collapsed': bool(collapsed)})
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


def push_test(p, brick, structure, cfg: TrialConfig, rate=0.2, limit=3.0, direction=(1, 0, 0)):
    """Ideal assembly, then a ramped horizontal force on the last brick until it moves."""
    ideal = TrialConfig(cfg.physics, ErrorModel(ideal=True), cfg.tolerance)
    result, (model, data, info, _) = assemble(p, brick, structure, ideal)
    if result['status'] != 'complete':
        return {'break_force_N': 0., 'status': result['status']}
    top = result['total'] - 1
    start, _ = body_pose(data, info, top)
    body = info['bodies'][top]
    f = 0.
    while f < limit:
        f += rate*model.opt.timestep
        data.xfrc_applied[body, :3] = np.asarray(direction)*f
        mujoco.mj_step(model, data)
        if np.linalg.norm(body_pose(data, info, top)[0] - start) > cfg.tolerance.drift:
            return {'break_force_N': f, 'status': 'moved'}
    return {'break_force_N': limit, 'status': 'held'}


def as_dict(cfg: TrialConfig):
    return {'physics': asdict(cfg.physics), 'error': asdict(cfg.error), 'tolerance': asdict(cfg.tolerance)}
