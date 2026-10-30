"""Closed-loop simulated Crazyflie assembly of DF bricks (brixzle's Experiment 2 pipeline, ported).

Reused from brixzle: the drone model and controller, Lighthouse estimator, simulated cflib,
camera, mission (``flight.mission``), passive fork and its tine guides, planner settings and the
harness bookkeeping. DF-specific: the fixture (a course-0 base of receivers per slot), the pick
(lift straight out of the receivers) and the place (vertical descent into the receivers below).

Measured approach error: with ``error`` (a brixzle ErrorModel), each placement gets a constant
Lighthouse position bias drawn from it, set when the drone starts its transit to the seat and
cleared after it verifies the place. The drone then believes it is over the seat while it is
really off by the measured positioning error, which is what Exp 3 measured. Picks fly with the
default (registered) Lighthouse model: the fixture's tine guides capture +/-5 mm, far below the
measured error, so pickup under that error is a separate problem (``fixture.funnel_covers_error``).
Results are simulated with uncalibrated contact and aerodynamics; they are not hardware rates.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace

import mujoco
import numpy as np

from brixzle import fixture as Fx, gripper as Gr, planner as BP, scene
from brixzle.drone import Drone, firmware_mass
from brixzle.estimator import Lighthouse
from brixzle.flyassemble import FlyConfig, Harness, summarize  # noqa: F401
from brixzle.simcf import SimPoseSource, SimWorld
from flight.frames import Pose

from . import cad, geometry as G, lattice as L, structures as S, trials as T
from .params import DFParams

MM = 1e-3


def fork_view(p: DFParams):
    """The fields brixzle's fork, tine guides and gripper rules read from a brick's params."""
    return SimpleNamespace(D=2*p.extent, wall=p.wall_t, fork_d=p.fork_d, fork_clearance=p.fork_clearance,
                           gutter=0., U=p.U)


def slot_frame(p: DFParams, fp: Fx.FixtureParams, index):
    """Brick origin (mm, wxyz) seated in fixture slot ``index``: receiver floors at the slot origin z."""
    centre, quat, _ = Fx.slot_pose(p, fp, index, 'A')
    return centre + np.array([0., 0., p.foot_L]), quat


def build_fixture(p: DFParams, fp: Fx.FixtureParams, n_slots, channels):
    """Static pieces (world mm): a row of slots, each the base receivers of one course-0 brick plus
    tine guides in front of its bores, and the launch stand."""
    view = fork_view(p)
    guides = Fx.funnel_pieces(view, fp, channels) if fp.funnel else []
    cradle = G.base_pieces(p, [(p.voxel_x(i), 0., p.receiver_kind(i), 0.) for i in range(p.n_vox)])
    lift = np.array([0., 0., -p.foot_L])  # base floors (z = 0) sit at the foot tips
    slots, pieces = [], []
    for i in range(n_slots):
        pos, quat = slot_frame(p, fp, i)
        slots.append({'index': i, 'kind': 'DF', 'part': 'DF', 'pos': pos.tolist(), 'quat': quat.tolist()})
        pieces += [q + lift + pos for q in cradle] + [q + pos for q in guides]
    stand = []
    sx, sy = fp.stand_xy
    for side in (-1, 1):
        x = sx + side*35.36
        stand.append(Fx.cad_box(np.array([x - 6, sy - 60, fp.floor_z]), np.array([x + 6, sy + 60, fp.stand_z])))
    return {'slots': slots, 'collision': pieces, 'stand': stand}


def pick_segments(p: DFParams, gripper, cfg: BP.PlannerConfig, transit_z, mass, guided=True):
    """brixzle's fork pick with an unhooked cradle: the feet lift straight out of the receivers."""
    c = -np.asarray(gripper['carry'])*MM
    back = (2*p.extent + gripper['params']['protrude'] + cfg.standoff_mm)*MM
    stroke = (p.guide_h + 2.)*MM
    low = cfg.aim_low_mm*MM if guided else 0.
    align = c + [0, -back, -low]
    unhook = c + [0, 0, stroke]
    up = unhook + [0, 0, cfg.lift_mm*MM]
    seg = BP._seg
    segs = [
        seg('GOTO_PREPICK', 'go_to', align, None, z_world=transit_z),
        seg('ALIGN', 'go_to', align, cfg.align_s),
        seg('ALIGN', 'wait', duration=cfg.align_wait_s),
        seg('ENGAGE', 'line', c + [0, 0, -low], back/cfg.v_engage),
        seg('ENGAGE', 'hold', duration=cfg.hold_s),
        seg('LIFT', 'param', name='ctrlMel.mass', value=mass['loaded']),
        seg('LIFT', 'hold', duration=cfg.hold_s),
        seg('LIFT', 'line', unhook, stroke/cfg.v_lift),
        seg('LIFT', 'line', up, cfg.lift_mm*MM/cfg.v_lift),
        seg('LIFT', 'go_to', up + [0, -cfg.clear_view_mm*MM, 0], cfg.min_go_to_s),
    ]
    abort = [
        seg('ABORT_PICK', 'param', name='ctrlMel.mass', value=mass['empty']),
        seg('ABORT_PICK', 'line', c + [0, -back, 0], back/cfg.v_engage),
        seg('ABORT_PICK', 'line', c + [0, -back, cfg.lift_mm*MM], cfg.lift_mm*MM/cfg.v_lift),
        seg('ABORT_PICK', 'go_to', c + [0, -back - cfg.clear_view_mm*MM, cfg.lift_mm*MM], cfg.min_go_to_s),
    ]
    return segs, abort


def place_segments(p: DFParams, gripper, cfg: BP.PlannerConfig, transit_z, mass):
    """Over the seat at the release height, straight down into the receivers, press, release, retreat."""
    c = -np.asarray(gripper['carry'])*MM
    back = (2*p.extent + gripper['params']['protrude'] + cfg.standoff_mm)*MM
    pre = c + [0, 0, T.release_lift(p, cfg.pre_clear_mm)*MM]
    start = c + [0, 0, (p.guide_h + 2.)*MM]  # feet tips just above the receiver ridges
    home = c + [0, 0, -cfg.press_mm*MM]
    # The receivers stop the brick at its seat, so after pressing the tines sit low in the bores;
    # centre them on the bores before withdrawing (lowering further, as A/B does, binds them).
    unloaded = c
    out = unloaded + [0, -back, 0]
    seg = BP._seg
    return [
        seg('TRANSIT', 'go_to', pre, None, z_world=transit_z),
        seg('PREPLACE', 'go_to', pre, cfg.align_s),
        seg('PREPLACE', 'wait', duration=cfg.align_wait_s),
        seg('DESCEND', 'line', start, float(np.linalg.norm(pre - start))/cfg.v_descend),
        seg('DESCEND', 'line', home, float(np.linalg.norm(home - start))/cfg.v_stroke),
        seg('DESCEND', 'hold', duration=cfg.hold_s),
        seg('RELEASE', 'param', name='ctrlMel.mass', value=mass['empty']),
        seg('RELEASE', 'line', unloaded, 0.3),
        seg('RELEASE', 'line', out, back/cfg.v_release),
        seg('RETREAT', 'line', out + [0, 0, cfg.retreat_mm*MM], cfg.retreat_mm*MM/cfg.v_lift),
        seg('RETREAT', 'go_to', out + [0, -cfg.clear_view_mm*MM, cfg.retreat_mm*MM], cfg.min_go_to_s),
    ]


def _relative(p, b, ref):
    tpos, tq = L.pose(p, b)
    rpos, rq = L.pose(p, ref)
    return Pose(tuple(tpos*MM), tuple(tq)).relative_to(Pose(tuple(rpos*MM), tuple(rq))).to_dict()


def compile_plan(p: DFParams, structure, brick, gripper, cfg: FlyConfig, n_slots=2, bricks=None):
    """plan.json for a DF structure (same schema as brixzle.planner.compile_plan)."""
    fp, drone, pc, tol = cfg.fixture, cfg.drone, cfg.planner, cfg.tolerance
    order = L.assembly_order(p, structure)
    order = order[:bricks] if bricks else order
    empty = (drone.mass_g + gripper['mass_g'])*1e-3
    mass = {'empty': empty, 'loaded': empty + brick['mass_g']*1e-3}
    payload_g = gripper['mass_g'] + brick['mass_g']
    top = max(max(b.course*p.H0 + p.foot_L for b in order) + brick['mesh'].bounds[1][2], fp.origin[2] + 40.)
    hang = -(np.asarray(gripper['carry'])[2] + brick['mesh'].bounds[0][2])
    transit_z = (top + hang + pc.transit_margin_mm)*MM
    home = Fx.stand_home(fp)*MM
    prev = np.array([home[0], home[1], home[2] + pc.takeoff_mm*MM])
    entries, per_brick = [], []
    for k, b in enumerate(order):
        slot = k % n_slots
        spos, squat = slot_frame(p, fp, slot)
        sframe = Pose(tuple(spos*MM), tuple(squat))
        pick, abort = pick_segments(p, gripper, pc, transit_z, mass, guided=fp.funnel)
        place = place_segments(p, gripper, pc, transit_z, mass)
        seat_pos, seat_q = L.pose(p, b)
        seat = Pose(tuple(seat_pos*MM), tuple(seat_q))
        sup = L.supporter(p, structure, b)
        if sup is not None:
            frame = {'supporter': sup.name, 'rel': _relative(p, b, sup), 'nominal': seat.to_dict()}
        else:
            frame = {'supporter': 'base0', 'rel': seat.to_dict(), 'nominal': seat.to_dict()}

        def at(fr):
            def f(s):
                pos = fr.apply(s['to'])
                if 'z_world' in s:
                    pos[2] = s['z_world']
                return pos
            return f
        prev = BP._fill_durations(pick, at(sframe), prev, pc)
        prev = BP._fill_durations(place, at(seat), prev, pc)
        per_brick.append(sum(s.get('duration', 0.) + (pc.settle_s if s['op'] == 'go_to' else 0.) for s in pick + place))
        entries.append({'name': b.name, 'course': b.course, 'i0': b.i0, 'j0': b.j0, 'axis': b.axis, 'part': 'DF',
                        'orientation': 'A', 'slot': slot, 'slot_kind': 'DF',
                        'carry': (np.asarray(gripper['carry'])*MM).tolist(),
                        'pick': {'slot_nominal': sframe.to_dict(), 'segments': pick, 'abort': abort},
                        'place': {'frame': frame, 'segments': place}})
    xs = [home[0], *(e['pick']['slot_nominal']['pos'][0] for e in entries),
          *(e['place']['frame']['nominal']['pos'][0] for e in entries)]
    ys = [home[1], *(e['pick']['slot_nominal']['pos'][1] for e in entries),
          *(e['place']['frame']['nominal']['pos'][1] for e in entries)]
    cam_mm, cam_deg = cfg.camera.sigma_pos_mm, cfg.camera.sigma_angle_deg
    plan = {
        'version': 1, 'name': f'df-{structure.name}:{len(order)}',
        'settings': {
            'setpoint_hz': pc.setpoint_hz, 'transit_z': transit_z, 'takeoff_z': home[2] + pc.takeoff_mm*MM,
            'takeoff_s': 2.0, 'land_s': 2.5, 'return_s': 3.0, 'settle_s': pc.settle_s,
            'emergency_land_z': (hang + 10.)*MM,
            'max_pick_retries': pc.max_pick_retries, 'max_place_retries': pc.max_place_retries,
            'ctrl_mass_kg': mass,
            # DF seats are square: the same tolerance across and along the brick.
            'tolerance': {'seat_xz_m': (tol.seat_xz + 3*cam_mm)*MM, 'seat_y_m': (tol.seat_xz + 3*cam_mm)*MM,
                          'seat_angle_deg': tol.seat_angle + 3*cam_deg, 'pick_gone_m': 0.005, 'recover_m': 0.03},
            'battery': {'flight_time_s': drone.flight_time_s(payload_g), 'per_brick_s': max(per_brick),
                        'reserve_s': pc.battery_reserve_s},
            'geofence': {'min': [min(xs) - 0.3, min(ys) - 0.3, fp.floor_z*MM],
                         'max': [max(xs) + 0.3, max(ys) + 0.3, transit_z + 0.3]},
            'limits': {'v_max': 1.0, 'a_max': 2.0},
            'params': {f'ctrlMel.{k}': v for k, v in pc.gains},
            'planner': pc.to_dict(),
        },
        'home': {'pos': home.tolist(), 'yaw': 0.},
        'bases': {'base0': {'pos': [0., 0., 0.], 'quat': [1., 0., 0., 0.]}},
        'bricks': entries,
    }
    return plan, {'order': order, 'payload_g': payload_g, 'transit_z': transit_z, 'per_brick_s': per_brick}


@dataclass
class DFFly:
    cfg: FlyConfig = field(default_factory=FlyConfig)
    error: object = None  # brixzle ErrorModel for the placement bias, or None


class DFHarness(Harness):
    """brixzle's harness on a DF world; judging with DF seats; optional measured placement bias."""

    def __init__(self, p: DFParams, plan, brick, gripper, structure, cfg: FlyConfig, seed=0, record_fps=0,
                 error=None):
        self.p, self.plan, self.gripper, self.cfg = p, plan, gripper, cfg
        self.structure = structure
        self.rng = np.random.default_rng(seed)
        self.error, self.error_rng, self.bias_log = error, np.random.default_rng(seed + 7), {}
        names = [b['name'] for b in plan['bricks']]
        self.entries = {b['name']: b for b in plan['bricks']}
        self.placements = {b['name']: L.Placement(b['course'], b['i0'], b['j0'], b['axis']) for b in plan['bricks']}
        self.brick = {'DF': brick}
        n_slots = max(b['slot'] for b in plan['bricks']) + 1
        self.fixture = build_fixture(p, cfg.fixture, n_slots, brick['channels'])
        static = list(self.fixture['collision']) + list(self.fixture['stand'])
        static += G.base_pieces(p, L.base_receivers(p, structure, T._ring_cells(p, structure)))
        home = np.asarray(plan['home']['pos'])
        self.model, self.xml, self.info = scene.build_scene(
            None, self.brick, [], len(names), cfg.physics, visual=record_fps > 0, static=static,
            body_parts=['DF']*len(names), drone={'cfg': cfg.drone, 'gripper': gripper, 'pos': home},
            collision='collision_bored', sleep=True)
        self.data = mujoco.MjData(self.model)
        self.lighthouse = Lighthouse(cfg.lighthouse, self.rng, origin=np.asarray(cfg.fixture.origin)*MM)
        self.drone = Drone(self.model, self.data, cfg.drone, self.lighthouse,
                           ctrl_mass=firmware_mass(plan['settings']['ctrl_mass_kg']['empty']))
        self.drone.place(home, plan['home'].get('yaw', 0.))
        mujoco.mj_forward(self.model, self.data)
        self.world = SimWorld(self.model, self.data, self.info, self.drone, {n: j for j, n in enumerate(names)},
                              record_fps=record_fps, battery_s=plan['settings']['battery']['flight_time_s'])
        self.camera = SimPoseSource(self.world, cfg.camera, np.random.default_rng(seed + 1))
        self.owner = T._brick_geoms(self.model, self.info)
        self.records = {n: {'brick': n, 'phases': {}} for n in names}
        self.current = self.f_track = None
        self.before = {}
        self.world.hooks.append(self._tick)
        self.swaps = 0
        for slot in sorted({b['slot'] for b in plan['bricks']}):
            self._activate_in_slot(next(b['name'] for b in plan['bricks'] if b['slot'] == slot))
        self.world.advance(0.8)

    def _truth_rel_seat(self, name):
        j = self.world.brick_index[name]
        ref = self._ref(name)
        err, angle = T.pose_error(self.p, self.data, self.info, j, self.placements[name], ref)
        return err, angle, ref

    def _seat_rot(self, name):
        _, q = L.pose(self.p, self.placements[name])
        return T._rot(q).as_matrix()

    def _rel_yaw(self, name):
        j = self.world.brick_index[name]
        _, q = scene.body_pose(self.data, self.info, j)
        _, tq = L.pose(self.p, self.placements[name])
        return (T._rot(tq).inv()*T._rot(q)).as_euler('xyz', degrees=True).tolist()

    def on_event(self, e):
        name, phase = e.get('brick'), e.get('phase')
        if self.error is not None and name is not None:
            if phase == 'TRANSIT' and name not in self.bias_log:
                s = self.error.sample(self.error_rng)
                b0 = np.array([s['dx'], s['dy'], 0.])*MM
                self.lighthouse.b0 = b0
                self.bias_log[name] = (b0/MM).tolist()
            elif phase == 'VERIFY_PLACE':
                self.lighthouse.b0 = np.zeros(3)
        super().on_event(e)
        if name is None:
            return
        rec = self.records[name]
        if name in self.bias_log:
            rec['placement_bias_mm'] = self.bias_log[name]
        if phase == 'VERIFY_PLACE' and rec.get('verify'):
            v = rec['verify'][-1]
            v['true_outcome'] = T.classify(self.p, np.asarray(v['true_err_mm']), v['true_angle_deg'],
                                           self.cfg.tolerance, v['true_outcome'] == 'fell')


def make(p: DFParams, gp, structure_name, cfg: FlyConfig, bricks=None):
    brick = cad.build_brick(p)
    gripper = Gr.build_gripper(gp, brick, fork_view(p))
    structure = S.catalog(p)[structure_name]()
    n_slots = max(1, cfg.n_slots)
    plan, layout = compile_plan(p, structure, brick, gripper, cfg, n_slots=n_slots, bricks=bricks)
    plan['sim'] = {'params': p.to_dict(), 'gripper': gp.to_dict(), 'structure': structure_name, 'bricks': bricks,
                   'config': cfg.to_dict(), 'family': 'dfbrick'}
    return plan, layout, brick, gripper, structure


def fly_assemble(p: DFParams, gp, structure_name, cfg: FlyConfig, seed=0, bricks=None, error=None,
                 record_fps=0):
    plan, layout, brick, gripper, structure = make(p, gp, structure_name, cfg, bricks)
    h = DFHarness(p, plan, brick, gripper, structure, cfg, seed=seed, record_fps=record_fps, error=error)
    return h.run(), plan, h


def gripper_margins(p: DFParams, gp, brick, gripper, drone):
    return Gr.gripper_rules(gp, gripper, brick, fork_view(p), drone)


def _one(args):
    from brixzle.flyassemble import config_from_dict
    from brixzle.trials import ErrorModel
    p_dict, gp_dict, structure_name, cfg_dict, seed, bricks, err = args
    error = None
    if err is not None:
        err = dict(err)
        err['radii_mm'] = tuple(err.get('radii_mm', ()))
        err['clearance'] = tuple(err.get('clearance', (5., 30.)))
        error = ErrorModel(**err)
    res, plan, h = fly_assemble(DFParams.from_dict(p_dict), Gr.params_from_dict(gp_dict), structure_name,
                                config_from_dict(cfg_dict), seed=seed, bricks=bricks, error=error)
    res.pop('events', None)
    res['seed'] = seed
    return res


def run_seeds(p: DFParams, gp, structure_name, cfg: FlyConfig, seeds, bricks=None, workers=1, error=None):
    from dataclasses import asdict
    err = asdict(error) if error is not None else None
    jobs = [(p.to_dict(), gp.to_dict(), structure_name, cfg.to_dict(), s, bricks, err) for s in seeds]
    if workers <= 1:
        return [_one(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(workers) as pool:
        return pool.map(_one, jobs)
