"""Drone assembly planner: order, fixture slots, and the per-brick motion primitives -> ``plan.json``.

The order is ``lattice.assembly_order`` (supporters first, stroke-aware, 2D path check).
Each brick gets a fixture slot and two segment lists for ``flight.mission``, both in a brick
frame (metres): ``pick`` in the camera-measured pose of the waiting brick, ``place`` in the
seat implied by the camera-measured supporter (``trials.expected_seat`` logic: seat =
supporter pose * nominal relative pose). In a brick frame the drone's grasp position is
``-carry`` (the carried brick's origin in the drone frame is ``carry``), so every primitive
is written once and holds for any slot or seat, including 180-degree-yawed ones.

Fork primitives (gripper family 'fork'):
  pick:  GOTO_PREPICK (transit height) -> ALIGN (tines on the bore axis, standing off in -y)
         -> ENGAGE (+y into the bores) -> payload on (ctrlMel.mass) -> LIFT (back out of the
         hook along -d, then up)
  place: TRANSIT -> PREPLACE (over the stroke start) -> DESCEND (vertical to the stroke start,
         then the tooth stroke along d, pressing ``press`` past the seat) -> RELEASE (payload
         off, unload the tines, withdraw -y) -> RETREAT (up)
Units: geometry in mm, plan output in m / s / rad.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from . import lattice as L, trials as T
from .drone_params import DroneConfig
from .fixture import FixtureParams, slot_pose, stand_home

MM = 1e-3


@dataclass(frozen=True)
class PlannerConfig:
    v_transit: float = 0.25        # m/s, average speed of go_to transits
    min_go_to_s: float = 1.5
    standoff_mm: float = 10.       # tine tips this far short of the brick's near face before ENGAGE
    v_engage: float = 0.05         # m/s
    v_lift: float = 0.03
    lift_mm: float = 40.
    v_descend: float = 0.06
    v_stroke: float = 0.02
    v_release: float = 0.04
    press_mm: float = 1.0          # stroke past the seat so the brick is pushed home
    unload_mm: float = 0.3         # lower the tines by the bore clearance before withdrawing
    pre_clear_mm: float = 10.      # PREPLACE height above the full-lift release height
    retreat_mm: float = 40.
    clear_view_mm: float = 130.    # move this far back in -y so the overhead camera sees the brick
    transit_margin_mm: float = 50.
    aim_low_mm: float = 1.5        # ALIGN this far below the bore axis; the guide's floor ramp lifts the tines
    align_s: float = 2.0
    settle_s: float = 0.3          # after each go_to
    align_wait_s: float = 1.0
    hold_s: float = 0.5
    takeoff_mm: float = 150.
    setpoint_hz: int = 20
    max_pick_retries: int = 2
    max_place_retries: int = 1
    yaw_in_transit: bool = False   # only matters when the fixture does not pre-rotate B bricks
    battery_reserve_s: float = 60.
    # Firmware Mellinger gains set at mission start (ctrlMel.*). The defaults have no roll/pitch
    # integral, so a payload COM offset of 1 mm leaves ~1.2 deg tilt and ~3 cm hover error.
    gains: tuple = (('ki_m_xy', 100000.), ('i_range_m_xy', 0.1))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def brick_axis(p, b: L.Placement):
    """Insertion axis (unit, 3D) in the brick's own frame."""
    d = L.insertion_axis(p, b)
    _, q = L.pose(p, b)
    return _rotz(-_quat_yaw(q)) @ np.array([d[0], 0., d[1]])


def assign_slots(p, order, fp: FixtureParams, n_slots):
    """Slot per brick: matching orientation (when pre-rotating), cycling so each slot is refilled after a pick."""
    want = [L.orientation(p, b) if fp.prerotate_B else 'A' for b in order]
    kinds = sorted(set(want))
    per = max(1, n_slots//len(kinds))
    orientations = [k for k in kinds for _ in range(per)]
    counters = {k: 0 for k in kinds}
    slots = []
    for o in want:
        idx = [i for i, so in enumerate(orientations) if so == o]
        slots.append(idx[counters[o] % len(idx)])
        counters[o] += 1
    return orientations, slots


def _seg(phase, op, to=None, duration=None, z_world=None, **kw):
    s = {'phase': phase, 'op': op}
    if to is not None:
        s['to'] = [float(v) for v in to]
    if duration is not None:
        s['duration'] = float(duration)
    if z_world is not None:
        s['z_world'] = float(z_world)
    s.update(kw)
    return s


def fork_pick(p, gripper, d_brick, cfg: PlannerConfig, transit_z, mass, hooked=False, guided=True):
    c = -np.asarray(gripper['carry'])*MM
    back = (p.D + gripper['params']['protrude'] + cfg.standoff_mm)*MM
    stroke = (p.tooth_L + p.slot_extra + 1.)*MM
    low = cfg.aim_low_mm*MM if guided else 0.
    align = c + [0, -back, -low]
    # A hooked brick must back out along its tooth axis; an unhooked cradle lifts straight up.
    unhook = c - stroke*d_brick if hooked else c + [0, 0, stroke]
    up = unhook + [0, 0, cfg.lift_mm*MM]
    segs = [
        _seg('GOTO_PREPICK', 'go_to', align, None, z_world=transit_z),
        _seg('ALIGN', 'go_to', align, cfg.align_s),
        _seg('ALIGN', 'wait', duration=cfg.align_wait_s),
        _seg('ENGAGE', 'line', c + [0, 0, -low], back/cfg.v_engage),
        _seg('ENGAGE', 'hold', duration=cfg.hold_s),
        _seg('LIFT', 'param', name='ctrlMel.mass', value=mass['loaded']),
        _seg('LIFT', 'hold', duration=cfg.hold_s),
        _seg('LIFT', 'line', unhook, stroke/cfg.v_lift),
        _seg('LIFT', 'line', up, cfg.lift_mm*MM/cfg.v_lift),
        _seg('LIFT', 'go_to', up + [0, -cfg.clear_view_mm*MM, 0], cfg.min_go_to_s),   # let the camera see the slot
    ]
    abort = [
        _seg('ABORT_PICK', 'param', name='ctrlMel.mass', value=mass['empty']),
        _seg('ABORT_PICK', 'line', c + [0, -back, 0], back/cfg.v_engage),
        _seg('ABORT_PICK', 'line', c + [0, -back, cfg.lift_mm*MM], cfg.lift_mm*MM/cfg.v_lift),
        _seg('ABORT_PICK', 'go_to', c + [0, -back - cfg.clear_view_mm*MM, cfg.lift_mm*MM], cfg.min_go_to_s),
    ]
    return segs, abort


def fork_place(p, gripper, d_brick, cfg: PlannerConfig, transit_z, mass):
    c = -np.asarray(gripper['carry'])*MM
    back = (p.D + gripper['params']['protrude'] + cfg.standoff_mm)*MM
    stroke = (p.tooth_L + p.slot_extra)*MM
    s0 = -stroke*d_brick
    lift = (p.amplitude + p.tooth_L + p.slot_extra + cfg.pre_clear_mm)*MM
    pre = c + s0 + [0, 0, max(cfg.pre_clear_mm*MM, lift - s0[2])]
    start = c + s0 + [0, 0, 2*MM]
    home = c + cfg.press_mm*MM*d_brick
    unloaded = home + [0, 0, -cfg.unload_mm*MM]
    out = unloaded + [0, -back, 0]
    segs = [
        _seg('TRANSIT', 'go_to', pre, None, z_world=transit_z),
        _seg('PREPLACE', 'go_to', pre, cfg.align_s),
        _seg('PREPLACE', 'wait', duration=cfg.align_wait_s),
        _seg('DESCEND', 'line', start, np.linalg.norm(pre - start)/cfg.v_descend),
        _seg('DESCEND', 'line', home, np.linalg.norm(home - start)/cfg.v_stroke),
        _seg('DESCEND', 'hold', duration=cfg.hold_s),
        _seg('RELEASE', 'param', name='ctrlMel.mass', value=mass['empty']),
        _seg('RELEASE', 'line', unloaded, 0.3),
        _seg('RELEASE', 'line', out, back/cfg.v_release),
        _seg('RETREAT', 'line', out + [0, 0, cfg.retreat_mm*MM], cfg.retreat_mm*MM/cfg.v_lift),
        _seg('RETREAT', 'go_to', out + [0, -cfg.clear_view_mm*MM, cfg.retreat_mm*MM], cfg.min_go_to_s),
    ]
    return segs


def _pose_dict(pos_mm, quat):
    return {'pos': [float(v)*MM for v in pos_mm], 'quat': [float(v) for v in quat]}


def _relative(p, b, ref):
    """Seat of b in ref's frame (m, wxyz), both nominal lattice poses."""
    from flight.frames import Pose
    tpos, tq = L.pose(p, b)
    rpos, rq = L.pose(p, ref)
    return Pose(tuple(tpos*MM), tuple(tq)).relative_to(Pose(tuple(rpos*MM), tuple(rq))).to_dict()


def structure_top_mm(p, bricks, brick_bundle):
    top = brick_bundle['mesh'].bounds[1][2] if 'mesh' in brick_bundle else brick_bundle['outer'].bounds[3]
    return max(b.course*p.H0 for b in bricks) + top


def _fill_durations(segs, frame_pos, prev_pos, cfg):
    """go_to segments without a duration get one from the distance to the previous target."""
    pos = prev_pos
    for s in segs:
        if 'to' not in s:
            continue
        target = frame_pos(s)
        if s['op'] == 'go_to' and 'duration' not in s:
            s['duration'] = max(cfg.min_go_to_s, float(np.linalg.norm(target - pos))/cfg.v_transit)
        pos = target
    return pos


def compile_plan(p, structure, brick_bundle, gripper, fp: FixtureParams, drone: DroneConfig,
                 cfg: PlannerConfig = PlannerConfig(), tol: T.Tolerance = T.Tolerance(), n_slots=4, bricks=None,
                 camera_sigma_mm=0.5, camera_sigma_deg=0.3):
    """Return (plan, layout). ``bricks`` limits the plan to the first n bricks of the order."""
    from flight.frames import Pose
    outer = {'A': brick_bundle['outer']}
    order = L.assembly_order(p, structure, outer)
    order = order[:bricks] if bricks else order
    graph = L.connectivity(structure)
    orientations, slot_of = assign_slots(p, order, fp, n_slots)
    payload_g = gripper['mass_g'] + brick_bundle['mass_g']
    empty = (drone.mass_g + gripper['mass_g'])*1e-3
    mass = {'empty': empty, 'loaded': empty + brick_bundle['mass_g']*1e-3}
    top = max(structure_top_mm(p, order, brick_bundle), fp.origin[2] + 40.)
    hang = -(np.asarray(gripper['carry'])[2] + (brick_bundle['mesh'].bounds[0][2] if 'mesh' in brick_bundle
                                                 else brick_bundle['outer'].bounds[1]))
    transit_z = (top + hang + cfg.transit_margin_mm)*MM
    home = stand_home(fp)*MM
    entries = []
    bases = {}
    for i, spec in enumerate(structure.bases):
        bases[f'base{i}'] = {'pos': [0., 0., 0.], 'quat': [1., 0., 0., 0.]}
    prev = np.array([home[0], home[1], home[2] + cfg.takeoff_mm*MM])
    per_brick = []
    placed = []
    for k, b in enumerate(order):
        slot = slot_of[k]
        spos, squat, _ = slot_pose(p, fp, slot, orientations[slot])
        slot_frame = Pose(tuple(spos*MM), tuple(squat))
        d_slot = brick_axis(p, L.Placement(0, 0))     # slot frame == a course-0 'A' seat, turned
        pick, abort = fork_pick(p, gripper, d_slot, cfg, transit_z, mass, hooked=fp.hooked, guided=fp.funnel)
        d_seat = brick_axis(p, b)
        place = fork_place(p, gripper, d_seat, cfg, transit_z, mass)
        seat_pos, seat_q = L.pose(p, b)
        seat_frame = Pose(tuple(seat_pos*MM), tuple(seat_q))
        supporters = sorted(u for u in graph.predecessors(b.name))
        placed_names = {x.name for x in placed}
        sup = next((u for u in supporters if u in placed_names), None)
        if sup is not None:
            ref = next(x for x in placed if x.name == sup)
            frame = {'supporter': sup, 'rel': _relative(p, b, ref), 'nominal': seat_frame.to_dict()}
        else:
            base = next(u for u in supporters if u.startswith('base'))
            frame = {'supporter': base, 'rel': seat_frame.to_dict(), 'nominal': seat_frame.to_dict()}

        def at(frame_pose):
            def f(s):
                pos = frame_pose.apply(s['to'])
                if 'z_world' in s:
                    pos[2] = s['z_world']
                return pos
            return f
        prev = _fill_durations(pick, at(slot_frame), prev, cfg)
        prev = _fill_durations(place, at(seat_frame), prev, cfg)
        t = sum(s.get('duration', 0.) + (cfg.settle_s if s['op'] == 'go_to' else 0.) for s in pick + place)
        per_brick.append(t)
        entries.append({'name': b.name, 'course': b.course, 'i0': b.i0, 'part': L.part(p, b),
                        'orientation': L.orientation(p, b), 'slot': slot,
                        'carry': (np.asarray(gripper['carry'])*MM).tolist(),
                        'pick': {'slot_nominal': slot_frame.to_dict(), 'segments': pick, 'abort': abort},
                        'place': {'frame': frame, 'segments': place}})
        placed.append(b)
    flight_time = drone.flight_time_s(payload_g)
    xs = [home[0], *(e['pick']['slot_nominal']['pos'][0] for e in entries),
          *(e['place']['frame']['nominal']['pos'][0] for e in entries)]
    ys = [home[1], *(e['pick']['slot_nominal']['pos'][1] for e in entries),
          *(e['place']['frame']['nominal']['pos'][1] for e in entries)]
    fence_margin = 0.25
    plan = {
        'version': 1,
        'name': f'{structure.name}:{len(order)}',
        'settings': {
            'setpoint_hz': cfg.setpoint_hz, 'transit_z': transit_z, 'takeoff_z': home[2] + cfg.takeoff_mm*MM,
            'takeoff_s': 2.0, 'land_s': 2.5, 'return_s': 3.0, 'settle_s': cfg.settle_s,
            'max_pick_retries': cfg.max_pick_retries, 'max_place_retries': cfg.max_place_retries,
            'ctrl_mass_kg': mass,
            # The mission judges seating through the noisy camera: widen the sim's tolerance by 3 sigma.
            'tolerance': {'seat_xz_m': (tol.seat_xz + 3*camera_sigma_mm)*MM,
                          'seat_y_m': (tol.seat_y + 3*camera_sigma_mm)*MM,
                          'seat_angle_deg': tol.seat_angle + 3*camera_sigma_deg,
                          'pick_gone_m': 0.005, 'recover_m': 0.03},
            'battery': {'flight_time_s': flight_time, 'per_brick_s': max(per_brick) if per_brick else 0.,
                        'reserve_s': cfg.battery_reserve_s},
            'geofence': {'min': [min(xs) - fence_margin, min(ys) - fence_margin, fp.floor_z*MM],
                         'max': [max(xs) + fence_margin, max(ys) + fence_margin, transit_z + 0.3]},
            'limits': {'v_max': 1.0, 'a_max': 2.0},
            'params': {f'ctrlMel.{k}': v for k, v in cfg.gains},
            'planner': cfg.to_dict(),
        },
        'home': {'pos': home.tolist(), 'yaw': 0.},
        'bases': bases,
        'bricks': entries,
    }
    layout = {'order': order, 'orientations': orientations, 'slot_of': slot_of, 'transit_z': transit_z,
              'payload_g': payload_g, 'per_brick_s': per_brick}
    return plan, layout
