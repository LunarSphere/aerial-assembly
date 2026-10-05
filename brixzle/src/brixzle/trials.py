"""MuJoCo trials: coarse-placement drops, sequential assembly, and push tests.

No drone dynamics: a brick is released as a free body at a sampled pose error
above its lattice seat. Metrics are simulated values under uncalibrated
contact parameters, not hardware success rates.
"""
from dataclasses import asdict, dataclass, field
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from aerial_assembly.config import Physics

from . import lattice as L
from .params import BrickParams
from .scene import MM, activate, body_pose, build_scene, park


@dataclass(frozen=True)
class ErrorModel:
    """Coarse drone placement: aim at the cavity, rely on geometry."""
    sigma_xy: float = 4.0        # mm
    clip_xy: float = 12.0        # mm
    aim_bias: float = 3.0        # mm toward the side the tooth stroke comes from
    sigma_yaw: float = 5.0       # deg
    sigma_tilt: float = 3.0      # deg (roll, pitch)
    clearance: tuple = (5., 30.)  # mm above the highest-reach release height
    sigma_vel: float = 0.03      # m/s residual velocity
    ideal: bool = False
    # Aim at the seat implied by the supporter's *measured* pose (global camera)
    # instead of the ideal lattice pose; matters once a cantilever sags.
    track_supporter: bool = False
    exact_courses: int = 0       # courses below this are placed without error (anchored start)

    def sample(self, rng):
        if self.ideal:
            return dict(dx=self.aim_bias, dy=0., yaw=0., roll=0., pitch=0., clear=self.clearance[0],
                        vel=[0., 0., 0.])
        dx, dy = np.clip(rng.normal(0, self.sigma_xy, 2), -self.clip_xy, self.clip_xy)
        return dict(dx=float(dx + self.aim_bias), dy=float(dy),
                    yaw=float(rng.normal(0, self.sigma_yaw)),
                    roll=float(rng.normal(0, self.sigma_tilt)), pitch=float(rng.normal(0, self.sigma_tilt)),
                    clear=float(rng.uniform(*self.clearance)),
                    vel=rng.normal(0, self.sigma_vel, 3).tolist())


@dataclass(frozen=True)
class Tolerance:
    seat_xz: float = 1.0       # mm
    seat_y: float = 2.5        # mm
    seat_angle: float = 3.0    # deg
    drift: float = 2.0         # mm moved by a placed brick during one stage => collapse
    lin_speed: float = 0.002   # m/s settle threshold
    ang_speed: float = 0.05    # rad/s
    settle_hold: float = 0.08  # s
    max_time: float = 1.5      # s per brick


@dataclass
class TrialConfig:
    physics: Physics = field(default_factory=lambda: Physics(timestep=0.001, friction=0.35,
                                                             contact_timeconst=0.004, iterations=50))
    error: ErrorModel = field(default_factory=ErrorModel)
    tolerance: Tolerance = field(default_factory=Tolerance)


def release_pose(p: BrickParams, placement, sample, seat_pose=None):
    """Seat pose displaced by the sampled error, lifted so nothing overlaps at release.

    ``seat_pose`` (pos, quat) overrides the ideal lattice seat, e.g. the seat
    implied by a sagging supporter's measured pose.
    """
    pos, quat = seat_pose if seat_pose is not None else L.pose(p, placement)
    d = L.insertion_axis(p, placement)
    side = -np.sign(d[0])  # the stroke arrives from this x side
    lift = p.amplitude + p.tooth_L + p.slot_extra + sample['clear']
    rel = pos + np.array([side*sample['dx'], sample['dy'], lift])
    seat = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    err = Rotation.from_euler('xyz', [sample['roll'], sample['pitch'], sample['yaw']], degrees=True)
    q = (err*seat).as_quat()
    return rel, np.array([q[3], q[0], q[1], q[2]])


def _rot(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def expected_seat(p, data, info, placement, ref):
    """Seat of ``placement`` carried along with supporter ref=(k, placement_k)'s actual pose."""
    k, rp = ref
    tpos, tquat = L.pose(p, placement)
    rpos, rquat = body_pose(data, info, k)
    ipos, iquat = L.pose(p, rp)
    delta = _rot(rquat)*_rot(iquat).inv()
    q = (delta*_rot(tquat)).as_quat()
    return rpos + delta.apply(tpos - ipos), np.array([q[3], q[0], q[1], q[2]])


def pose_error(p, data, info, j, placement, ref=None):
    """Position (mm) and angle (deg) error of brick j against its lattice seat.

    With ``ref = (k, ref_placement)`` the seat is taken relative to brick k's
    actual pose, so a brick that seats correctly on a sagging supporter counts
    as seated; sag is reported separately.
    """
    pos, quat = body_pose(data, info, j)
    tpos, tquat = L.pose(p, placement)
    target_r = _rot(tquat)
    if ref is not None:
        k, ref_placement = ref
        rpos, rquat = body_pose(data, info, k)
        ipos, iquat = L.pose(p, ref_placement)
        delta = _rot(rquat)*_rot(iquat).inv()
        tpos = rpos + delta.apply(tpos - ipos)
        target_r = delta*target_r
    r = _rot(quat)
    angle = math.degrees((target_r.inv()*r).magnitude())
    return pos - tpos, angle


def classify(p, err, angle, tol: Tolerance, on_floor):
    dx, dy, dz = err
    if on_floor:
        return 'fell'
    if abs(dx) <= tol.seat_xz and abs(dz) <= tol.seat_xz and abs(dy) <= tol.seat_y and angle <= tol.seat_angle:
        return 'seated'
    if abs(abs(dx) - 2*p.U*round(abs(dx)/(2*p.U))) <= 2*tol.seat_xz and abs(dx) > p.U and angle <= tol.seat_angle:
        return 'wrong_valley'
    return 'jammed'


def _brick_geoms(model, info):
    owner = {}
    for j, b in enumerate(info['bodies']):
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == b:
                owner[g] = j
    return owner


def run_until_settled(model, data, info, j, tol: Tolerance, owner, watch=(), frames=None, fps=60):
    """Step until brick j (and watched bricks) are still; return impact metrics."""
    force = np.zeros(6)
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
        total, touching = 0., False
        for i in range(data.ncon):
            c = data.contact[i]
            if owner.get(c.geom1) != j and owner.get(c.geom2) != j:
                continue
            touching = True
            mujoco.mj_contactForce(model, data, i, force)
            total += abs(force[0])
            pen_max = max(pen_max, -c.dist)
            if info['floor'] in (c.geom1, c.geom2):
                floor_hit = True
        f_max = max(f_max, total)
        if touching and not in_contact:
            onsets += 1
        in_contact = touching
        speeds = [np.linalg.norm(data.qvel[info['qvel'][m]:info['qvel'][m]+3]) for m in moving]
        spins = [np.linalg.norm(data.qvel[info['qvel'][m]+3:info['qvel'][m]+6]) for m in moving]
        if max(speeds) < tol.lin_speed and max(spins) < tol.ang_speed and in_contact:
            still_since = still_since if still_since is not None else data.time
            if data.time - still_since >= tol.settle_hold:
                break
        else:
            still_since = None
    settled = still_since is not None and data.time - still_since >= tol.settle_hold
    return {'f_max': f_max, 'onsets': onsets, 'floor': floor_hit, 'penetration_mm': pen_max/MM,
            'time': (still_since if settled else data.time) - t0, 'settled': settled}


def _bundles(brick):
    return {'A': brick} if 'collision' in brick else brick


def _outers(bundles):
    return {k: v['outer'] for k, v in bundles.items()}


def drop_trials(p: BrickParams, brick, cfg: TrialConfig, samples=32, seed=0, offsets=None, record=False,
                part='A'):
    """Drop one brick and judge its seat; Monte Carlo over the error model.

    part 'A': an A brick onto an anchored base at voxel 1. part 'B' ('ab' mode):
    a B brick onto an ideally seated A (B never lands on the base).
    ``offsets`` instead runs a deterministic capture sweep over X offsets (mm).
    """
    bundles = _bundles(brick)
    if part == 'B':
        support, target = L.Placement(0, 1), L.Placement(1, 1)
        body_parts, j = ['A', 'B'], 1
    else:
        support, target = None, L.Placement(0, 1)
        body_parts, j = ['A'], 0
    model, xml, info = build_scene(p, bundles, [(0, 4)], len(body_parts), cfg.physics, visual=record,
                                   body_parts=body_parts)
    data = mujoco.MjData(model)
    owner = _brick_geoms(model, info)
    rng = np.random.default_rng(seed)
    rows = []
    runs = offsets if offsets is not None else range(samples)
    for item in runs:
        mujoco.mj_resetData(model, data)
        if support is not None:
            spos, squat = L.pose(p, support)
            activate(model, data, info, 0, spos, squat)
            ideal = Tolerance(max_time=0.3)
            run_until_settled(model, data, info, 0, ideal, owner)
        sample = cfg.error.sample(rng)
        if offsets is not None:
            sample = dict(sample, dx=float(item), dy=0., yaw=0., roll=0., pitch=0., vel=[0, 0, 0],
                          clear=cfg.error.clearance[0])
        pos, quat = release_pose(p, target, sample)
        activate(model, data, info, j, pos, quat, linvel=sample['vel'])
        frames = [] if record else None
        m = run_until_settled(model, data, info, j, cfg.tolerance, owner, frames=frames,
                              watch=range(j))
        ref = (0, support) if support is not None else None
        err, angle = pose_error(p, data, info, j, target, ref)
        outcome = classify(p, err, angle, cfg.tolerance, m['floor'])
        rows.append({**sample, **m, 'outcome': outcome, 'err_mm': err.tolist(), 'angle_deg': angle})
        if record:
            rows[-1]['frames'] = frames
    return rows, xml


def summarize_drops(rows):
    n = len(rows)
    seated = [r for r in rows if r['outcome'] == 'seated']
    outcomes = {}
    for r in rows:
        outcomes[r['outcome']] = outcomes.get(r['outcome'], 0) + 1
    return {
        'n': n, 'p_success': len(seated)/n if n else 0.,
        'outcomes': outcomes,
        'f_max_N': float(np.percentile([r['f_max'] for r in rows], 95)) if rows else 0.,
        't_settle_s': float(np.mean([r['time'] for r in rows])) if rows else 0.,
        'collisions': float(np.mean([max(0, r['onsets'] - 1) + (r['penetration_mm'] > 1.0) for r in rows])),
        'max_penetration_mm': float(max(r['penetration_mm'] for r in rows)) if rows else 0.,
    }


def capture_radius(rows, level=0.9, bins=None):
    """Largest |offset from aim| band whose seated fraction stays >= level."""
    offs = np.array([abs(r['dx'] - r.get('aim', 0)) for r in rows])
    ok = np.array([r['outcome'] == 'seated' for r in rows])
    bins = bins if bins is not None else np.arange(0, offs.max() + 1.0, 1.0)
    radius = 0.
    for hi in bins[1:]:
        sel = offs <= hi
        if sel.sum() and ok[sel].mean() >= level:
            radius = float(hi)
        else:
            break
    return radius


def assemble(p: BrickParams, brick, structure, cfg: TrialConfig, seed=0, stop_on_fail=True,
             record=None):
    """Place bricks in planned order with sampled errors; detect failures and collapse.

    A new brick is judged against its seat on its actual supporter. Collapse is
    motion of already placed bricks while the new one settles (beyond
    ``tolerance.drift``), floor contact, or cumulative sag beyond half a course.
    ``record`` optionally receives qpos frames for every stage.
    """
    bundles = _bundles(brick)
    order = L.assembly_order(p, structure, _outers(bundles))
    graph = L.connectivity(structure)
    index = {b.name: j for j, b in enumerate(order)}
    model, xml, info = build_scene(p, bundles, structure.bases, len(order), cfg.physics,
                                   visual=record is not None, body_parts=[L.part(p, b) for b in order])
    data = mujoco.MjData(model)
    owner = _brick_geoms(model, info)
    rng = np.random.default_rng(seed)
    stages, placed = [], []
    status = 'complete'
    for j, b in enumerate(order):
        before = [body_pose(data, info, i) for i in range(j)]
        sample = cfg.error.sample(rng)
        if b.course < cfg.error.exact_courses:
            sample = dict(dx=0., dy=0., yaw=0., roll=0., pitch=0., clear=cfg.error.clearance[0], vel=[0., 0., 0.])
        supporters = [u for u in graph.predecessors(b.name) if u in index]
        ref = (index[supporters[0]], order[index[supporters[0]]]) if supporters else None
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
            moved = max(moved, float(np.linalg.norm(pos_i - before[i][0])), turn*p.U*math.pi/180)
            sag = max(sag, float(np.linalg.norm(pose_error(p, data, info, i, prev)[0])))
        collapsed = moved > cfg.tolerance.drift or sag > p.H0/2 or m['floor']
        stages.append({'brick': b.name, 'sample': sample, **m, 'outcome': outcome,
                       'err_mm': err.tolist(), 'angle_deg': angle,
                       'stage_motion_mm': moved, 'sag_mm': sag, 'collapsed': bool(collapsed)})
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
    """Ideal assembly, then a ramped horizontal force on the top brick until it moves."""
    ideal = TrialConfig(cfg.physics, ErrorModel(ideal=True, aim_bias=cfg.error.aim_bias), cfg.tolerance)
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
        pos, _ = body_pose(data, info, top)
        if np.linalg.norm(pos - start) > cfg.tolerance.drift:
            return {'break_force_N': f, 'status': 'moved'}
    return {'break_force_N': limit, 'status': 'held'}


def as_dict(cfg: TrialConfig):
    return {'physics': asdict(cfg.physics), 'error': asdict(cfg.error), 'tolerance': asdict(cfg.tolerance)}
