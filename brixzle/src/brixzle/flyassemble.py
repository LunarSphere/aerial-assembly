"""Closed-loop drone assembly in simulation: build the world, run ``flight.mission``, log the truth.

The mission sees only the cflib API, the Lighthouse estimate and the noisy overhead camera;
this module additionally records ground truth per brick (positioning error before descent,
seat error at release, true outcome via ``trials.classify``, motion of placed bricks, peak
contact force) without feeding any of it back. Results are simulated, with uncalibrated
contact and aerodynamics; they are not hardware rates.
"""
from dataclasses import asdict, dataclass, field
import math

import mujoco
import numpy as np

from aerial_assembly.config import Physics
from flight.frames import Pose
from flight.mission import Mission

from . import cad, drone as D, fixture as Fx, gripper as Gr, lattice as L, planner as P, scene, structures as S, \
    trials as T
from .drone_params import DroneConfig
from .estimator import Lighthouse, LighthouseModel
from .params import BrickParams
from .simcf import CameraModel, SimClock, SimCrazyflie, SimLogConfig, SimPoseSource, SimSyncCrazyflie, SimWorld

MM = 1e-3


@dataclass
class FlyConfig:
    physics: Physics = field(default_factory=lambda: Physics(timestep=0.001, friction=0.35, contact_timeconst=0.004,
                                                             iterations=50))
    drone: DroneConfig = field(default_factory=DroneConfig)
    lighthouse: LighthouseModel = field(default_factory=LighthouseModel)
    camera: CameraModel = field(default_factory=CameraModel)
    fixture: Fx.FixtureParams = field(default_factory=Fx.FixtureParams)
    planner: P.PlannerConfig = field(default_factory=P.PlannerConfig)
    tolerance: T.Tolerance = field(default_factory=T.Tolerance)
    n_slots: int = 2
    keep_going: bool = False

    def to_dict(self):
        return {k: asdict(v) if hasattr(v, '__dataclass_fields__') else v for k, v in self.__dict__.items()}


def sim_section(p, gp, structure_name, cfg: FlyConfig, bricks):
    """What ``fly --sim`` needs to rebuild the same world from plan.json alone."""
    return {'params': p.to_dict(), 'gripper': gp.to_dict(), 'structure': structure_name, 'bricks': bricks,
            'config': cfg.to_dict()}


def config_from_dict(d):
    cfg = FlyConfig()
    kinds = {'physics': Physics, 'drone': DroneConfig, 'lighthouse': LighthouseModel, 'camera': CameraModel,
             'fixture': Fx.FixtureParams, 'planner': P.PlannerConfig, 'tolerance': T.Tolerance}
    for k, v in d.items():
        if k in kinds:
            cls = kinds[k]
            names = cls.__dataclass_fields__
            v = {a: (tuple(b) if isinstance(getattr(cls(), a, None), tuple) else b) for a, b in v.items() if a in names}
            setattr(cfg, k, cls(**v))
        elif k in ('n_slots', 'keep_going'):
            setattr(cfg, k, v)
    return cfg


def make_plan(p: BrickParams, gp, structure_name, cfg: FlyConfig, bricks=None):
    brick = cad.build_brick(p)
    gripper = Gr.build_gripper(gp, brick, p)
    structure = S.CATALOG[structure_name]()
    plan, layout = P.compile_plan(p, structure, brick, gripper, cfg.fixture, cfg.drone, cfg.planner, cfg.tolerance,
                                  n_slots=cfg.n_slots, bricks=bricks, camera_sigma_mm=cfg.camera.sigma_pos_mm,
                                  camera_sigma_deg=cfg.camera.sigma_angle_deg)
    plan['sim'] = sim_section(p, gp, structure_name, cfg, bricks)
    return plan, layout, brick, gripper, structure


class Harness:
    """Builds the MuJoCo world for a plan and records ground truth from mission events."""

    def __init__(self, p, plan, brick, gripper, structure, cfg: FlyConfig, seed=0, record_fps=0):
        self.p, self.plan, self.brick, self.gripper, self.cfg = p, plan, brick, gripper, cfg
        self.structure = structure
        self.rng = np.random.default_rng(seed)
        names = [b['name'] for b in plan['bricks']]
        self.entries = {b['name']: b for b in plan['bricks']}
        self.placements = {b['name']: L.Placement(b['course'], b['i0']) for b in plan['bricks']}
        orients = ['A']*max(cfg.n_slots, max(b['slot'] for b in plan['bricks']) + 1)
        for b in plan['bricks']:
            yaw = Pose.from_dict(b['pick']['slot_nominal']).yaw
            orients[b['slot']] = 'B' if abs(abs(yaw) - math.pi) < 1e-6 else 'A'
        self.fixture = Fx.build_fixture(p, cfg.fixture, orients, brick['channels'])
        home = np.asarray(plan['home']['pos'])
        static = list(self.fixture['collision']) + list(self.fixture['stand'])
        self.model, self.xml, self.info = scene.build_scene(
            p, brick, structure.bases, len(names), cfg.physics, visual=record_fps > 0, static=static,
            drone={'cfg': cfg.drone, 'gripper': gripper, 'pos': home}, collision='collision_bored', sleep=True)
        self.data = mujoco.MjData(self.model)
        origin = np.asarray(cfg.fixture.origin)*MM
        lh = Lighthouse(cfg.lighthouse, self.rng, origin=origin)
        self.drone = D.Drone(self.model, self.data, cfg.drone, lh,
                             ctrl_mass=D.firmware_mass(plan['settings']['ctrl_mass_kg']['empty']))
        self.drone.place(home, plan['home'].get('yaw', 0.))
        mujoco.mj_forward(self.model, self.data)
        self.world = SimWorld(self.model, self.data, self.info, self.drone, {n: j for j, n in enumerate(names)},
                              record_fps=record_fps, battery_s=plan['settings']['battery']['flight_time_s'])
        self.camera = SimPoseSource(self.world, cfg.camera, np.random.default_rng(seed + 1))
        self.owner = T._brick_geoms(self.model, self.info)
        self.records = {n: {'brick': n, 'phases': {}} for n in names}
        self.current = None
        self.f_track = None
        self.before = {}
        self.world.hooks.append(self._tick)
        self.swaps = 0
        # Fill every slot with its first brick and let them settle on the fixture.
        for slot in sorted({b['slot'] for b in plan['bricks']}):
            first = next(b['name'] for b in plan['bricks'] if b['slot'] == slot)
            self._activate_in_slot(first)
        self.world.advance(0.8)

    # ---- helpers ----------------------------------------------------------------------------------
    def _activate_in_slot(self, name):
        j = self.world.brick_index[name]
        nom = self.entries[name]['pick']['slot_nominal']
        scene.activate(self.model, self.data, self.info, j, np.asarray(nom['pos'])/MM, nom['quat'])
        self.world.active[j] = True

    def _slot_free(self, slot):
        for n, e in self.entries.items():
            j = self.world.brick_index[n]
            if e['slot'] == slot and self.world.active[j]:
                pos, _ = scene.body_pose(self.data, self.info, j)
                if np.linalg.norm(pos*MM - np.asarray(e['pick']['slot_nominal']['pos'])) < 0.03:
                    return False
        return True

    def _tick(self, world):
        if self.f_track is None:
            return
        j = self.world.brick_index[self.f_track]
        force = np.zeros(6)
        total = 0.
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.owner.get(c.geom1) == j or self.owner.get(c.geom2) == j:
                mujoco.mj_contactForce(self.model, self.data, i, force)
                total += abs(force[0])
        rec = self.records[self.f_track]
        rec['f_max_N'] = max(rec.get('f_max_N', 0.), total)

    def _truth_rel_seat(self, name):
        """True brick pose relative to its true seat (the actual supporter's pose * nominal relation)."""
        j = self.world.brick_index[name]
        b = self.placements[name]
        ref = self._ref(name)
        err, angle = T.pose_error(self.p, self.data, self.info, j, b, ref)
        return err, angle, ref

    def _ref(self, name):
        sup = self.entries[name]['place']['frame']['supporter']
        if sup in self.world.brick_index:
            return (self.world.brick_index[sup], self.placements[sup])
        return None

    def _drone_truth(self):
        pos, _, vel, _ = self.drone.true_state()
        est = self.drone.last_estimate
        return {'true_m': pos.tolist(), 'est_m': [float(v) for v in est[0]], 'vel_mps': vel.tolist()}

    # ---- mission events ---------------------------------------------------------------------------
    def on_event(self, e):
        name = e.get('brick')
        phase = e.get('phase')
        if name is None:
            return
        rec = self.records[name]
        rec['phases'].setdefault(phase, e['t'])
        if phase == 'PICK_REQUEST':
            j = self.world.brick_index[name]
            if not self.world.active[j]:
                slot = self.entries[name]['slot']
                if not self._slot_free(slot):
                    raise RuntimeError(f'slot {slot} still occupied when {name} is due')
                self._activate_in_slot(name)
                self.world.advance(0.5)
            self.current = name
            self.f_track = name
            self.before = {n: scene.body_pose(self.data, self.info, k) for n, k in self.world.brick_index.items()
                           if self.world.active[k] and n != name}
        elif phase == 'DESCEND' and 'preplace' not in rec:
            err, angle, _ = self._truth_rel_seat(name)
            seg = next(s for s in self.entries[name]['place']['segments'] if s['phase'] == 'PREPLACE' and 'to' in s)
            nominal = (np.asarray(seg['to']) + np.asarray(self.entries[name]['carry']))/MM
            j = self.world.brick_index[name]
            v = self.data.qvel[self.info['qvel'][j]:self.info['qvel'][j]+6]
            yaw = self._rel_yaw(name)
            rec['preplace'] = {'err_mm': (err - self._seat_rot(name) @ nominal).tolist(), 'angle_deg': angle,
                               'yaw_deg': yaw, 'vel_mps': v[:3].tolist(), 'drone': self._drone_truth()}
        elif phase == 'RELEASE' and 'release' not in rec:
            err, angle, _ = self._truth_rel_seat(name)
            rec['release'] = {'err_mm': err.tolist(), 'angle_deg': angle, 'drone': self._drone_truth()}
        elif phase == 'VERIFY_PLACE':
            err, angle, ref = self._truth_rel_seat(name)
            j = self.world.brick_index[name]
            floor = any(self.info['floor'] in (c.geom1, c.geom2) and self.owner.get(c.geom1, self.owner.get(c.geom2)) == j
                        for c in self.data.contact[:self.data.ncon])
            outcome = T.classify(self.p, err, angle, self.cfg.tolerance, floor)
            moved = 0.
            for n, (pos0, q0) in self.before.items():
                pos1, q1 = scene.body_pose(self.data, self.info, self.world.brick_index[n])
                turn = math.degrees((T._rot(q0).inv()*T._rot(q1)).magnitude())
                moved = max(moved, float(np.linalg.norm(pos1 - pos0)), turn*self.p.U*math.pi/180)
            rec.setdefault('verify', []).append({
                'mission_status': e.get('status'), 'mission_err_m': e.get('err_m'),
                'true_err_mm': err.tolist(), 'true_angle_deg': angle, 'true_outcome': outcome,
                'stage_motion_mm': moved, 'collapsed': bool(moved > self.cfg.tolerance.drift)})
            self.f_track = None

    def _seat_rot(self, name):
        _, q = L.pose(self.p, self.placements[name])
        return T._rot(q).as_matrix()

    def _rel_yaw(self, name):
        j = self.world.brick_index[name]
        _, q = scene.body_pose(self.data, self.info, j)
        _, tq = L.pose(self.p, self.placements[name])
        r = (T._rot(tq).inv()*T._rot(q)).as_euler('xyz', degrees=True)
        return r.tolist()

    def run(self, keep_going=None):
        keep = self.cfg.keep_going if keep_going is None else keep_going
        cf = SimCrazyflie(world=self.world)

        def swap():
            self.swaps += 1
            self.world.flight_s = 0.
        with SimSyncCrazyflie('sim://brixzle', cf=cf) as scf:
            mission = Mission(scf, self.plan, SimClock(self.world), self.camera, SimLogConfig,
                              on_event=self.on_event, on_battery_swap=swap, keep_going=keep)
            result = mission.run()
        return self.summary(result)

    def summary(self, result):
        bricks = []
        for rec in result.bricks:
            r = dict(self.records[rec['brick']], **rec)
            ph = r['phases']
            ts = sorted(ph.values())
            r['cycle_s'] = (ts[-1] - ts[0]) if len(ts) > 1 else 0.
            last = r.get('verify', [{}])[-1]
            r['true_outcome'] = last.get('true_outcome', 'not_placed')
            r['success'] = r['outcome'] == 'seated' and r['true_outcome'] == 'seated' and not last.get('collapsed')
            bricks.append(r)
        n_ok = sum(b['success'] for b in bricks)
        return {'plan': self.plan['name'], 'mission_status': result.status, 'placed': len(bricks),
                'total': len(self.plan['bricks']), 'successes': n_ok, 'bricks': bricks,
                'battery_swaps': result.battery_swaps, 'sim_time_s': self.world.time,
                'events': result.events, 'watchdog': self.world.fw.log}


def fly_assemble(p, gp, structure_name, cfg: FlyConfig, seed=0, bricks=None, record_fps=0):
    plan, layout, brick, gripper, structure = make_plan(p, gp, structure_name, cfg, bricks)
    h = Harness(p, plan, brick, gripper, structure, cfg, seed=seed, record_fps=record_fps)
    return h.run(), plan, h


def run_plan(plan, seed=0, record_fps=0):
    """Rebuild the world described by ``plan['sim']`` and fly the plan exactly as given."""
    sim = plan['sim']
    p = BrickParams.from_dict(sim['params'])
    gp = Gr.params_from_dict(sim['gripper'])
    cfg = config_from_dict(sim['config'])
    brick = cad.build_brick(p)
    gripper = Gr.build_gripper(gp, brick, p)
    structure = S.CATALOG[sim['structure']]()
    h = Harness(p, plan, brick, gripper, structure, cfg, seed=seed, record_fps=record_fps)
    return h.run(), h


def _one(args):
    p_dict, gp_dict, structure_name, cfg_dict, seed, bricks = args
    p = BrickParams.from_dict(p_dict)
    res, plan, h = fly_assemble(p, Gr.params_from_dict(gp_dict), structure_name, config_from_dict(cfg_dict),
                                seed=seed, bricks=bricks)
    res.pop('events', None)
    res['seed'] = seed
    return res


def run_seeds(p, gp, structure_name, cfg: FlyConfig, seeds, bricks=None, workers=1):
    jobs = [(p.to_dict(), gp.to_dict(), structure_name, cfg.to_dict(), s, bricks) for s in seeds]
    if workers <= 1:
        return [_one(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(workers) as pool:
        return pool.map(_one, jobs)


def _stats(values):
    a = np.asarray(values, float)
    if a.size == 0:
        return None
    return {'n': int(a.shape[0]), 'mean': a.mean(0).tolist(), 'std': a.std(0).tolist(),
            'p95_abs': np.percentile(np.abs(a), 95, axis=0).tolist()}


def summarize(results):
    """Pooled per-brick success and the measured positioning/release error distributions (mm, deg)."""
    bricks = [b for r in results for b in r['bricks']]
    pre = [b['preplace']['err_mm'] for b in bricks if 'preplace' in b]
    yaw = [b['preplace']['yaw_deg'] for b in bricks if 'preplace' in b]
    vel = [b['preplace']['vel_mps'] for b in bricks if 'preplace' in b]
    rel = [b['release']['err_mm'] for b in bricks if 'release' in b]
    drone_err = [np.subtract(b['preplace']['drone']['true_m'], b['preplace']['drone']['est_m'])/MM
                 for b in bricks if 'preplace' in b]
    outcomes = {}
    for b in bricks:
        key = b['outcome'] if b['outcome'] != 'seated' else f"seated/{b['true_outcome']}"
        outcomes[key] = outcomes.get(key, 0) + 1
    planned = sum(r['total'] for r in results)
    return {
        'runs': len(results), 'bricks_attempted': len(bricks), 'bricks_planned': planned,
        'p_brick_success': (sum(b['success'] for b in bricks)/len(bricks)) if bricks else 0.,
        'p_structure_complete': float(np.mean([r['successes'] == r['total'] for r in results])),
        'outcomes': outcomes,
        'preplace_err_mm': _stats(pre), 'preplace_rpy_deg': _stats(yaw), 'preplace_vel_mps': _stats(vel),
        'release_err_mm': _stats(rel), 'lighthouse_err_mm': _stats(drone_err),
        'cycle_s': _stats([b['cycle_s'] for b in bricks]),
        'f_max_N_p95': float(np.percentile([b.get('f_max_N', 0.) for b in bricks], 95)) if bricks else 0.,
        'battery_swaps': sum(r['battery_swaps'] for r in results),
        'mission_status': {r['seed']: r['mission_status'] for r in results},
    }
