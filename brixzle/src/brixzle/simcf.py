"""Simulated Crazyflie behind the cflib API: ``SimSyncCrazyflie``/``SimCrazyflie`` drive the MuJoCo drone.

The classes duck-type the parts of cflib 0.1.33 the mission uses (same method signatures;
``tests/test_flight_parity.py`` compares them with ``inspect.signature``) and emulate the
firmware behaviour that matters for the mission:

* high-level commander: firmware 7th-order pieces (``flight.poly7``), planner IDLE/FLYING/
  LANDING states, go_to from the current goal or (when idle) from the state estimate,
  ``stop()`` cuts the motors;
* low-level setpoints take priority over the high-level commander and stop it;
  ``send_notify_setpoint_stop`` relaxes priority and hands the estimate back to it;
* supervisor watchdog (``supervisor.c``): setpoint older than 0.5 s -> level out (roll = pitch
  = 0, hold z), older than 2 s -> motors off and locked until re-armed;
* logs return the Lighthouse estimate (yaw in degrees, as the firmware logs it), never truth;
* ``ctrlMel.mass`` (kg) maps onto crazyflow's controller mass via ``drone.firmware_mass``.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np

from cflib.utils.callbacks import Caller
from flight import poly7
from flight.frames import Pose

from . import drone as D
from .cf.rot import quat_yaw
from .scene import MM, body_pose

WDT_LEVEL_S = 0.5      # COMMANDER_WDT_TIMEOUT_STABILIZE
WDT_SHUTDOWN_S = 2.0   # COMMANDER_WDT_TIMEOUT_SHUTDOWN
PRIO_DISABLE, PRIO_HIGHLEVEL, PRIO_CRTP = 0, 1, 2

DEFAULT_PARAMS = {
    'stabilizer.estimator': '0', 'stabilizer.controller': '0', 'commander.enHighLevel': '0',
    'deck.bcLighthouse4': '1', 'deck.bcFlow2': '0', 'deck.bcFlow': '0', 'ctrlMel.mass': '0.0461',
    'kalman.resetEstimation': '0', 'kalman.initialZ': '0.0', 'supervisor.infdmp': '0',
}
# Firmware Mellinger defaults (controller_mellinger.c); kd_z follows crazyflow's 0.5, not the firmware's 0.4.
DEFAULT_PARAMS.update({f'ctrlMel.{k}': str(v) for k, v in {
    'kp_xy': 0.4, 'kd_xy': 0.2, 'ki_xy': 0.05, 'i_range_xy': 2.0, 'kp_z': 1.25, 'kd_z': 0.5, 'ki_z': 0.05,
    'i_range_z': 0.4, 'kR_xy': 70000, 'kw_xy': 20000, 'ki_m_xy': 0.0, 'i_range_m_xy': 1.0, 'kR_z': 60000,
    'kw_z': 12000, 'ki_m_z': 500, 'i_range_m_z': 1500, 'kd_omega_rp': 200}.items()})


class Firmware:
    """Commander, high-level planner and supervisor state of one simulated Crazyflie."""

    def __init__(self, world):
        self.w = world
        self.params = dict(DEFAULT_PARAMS)
        self.priority = PRIO_DISABLE
        self.planner = 'idle'              # idle | flying | landing
        self.piece = None
        self.hl_state = None               # TrajEval used when the planner is idle
        self.ll_setpoint = None
        self.last_setpoint_t = -math.inf
        self.armed = False
        self.locked = False
        self.motors_on = False
        self.log = []                      # (t, event) for tests

    # --- params -------------------------------------------------------------------------------
    def set_param(self, name, value):
        if name not in self.params:
            raise KeyError(f'{name} not in param TOC')
        self.params[name] = str(value)
        if name == 'ctrlMel.mass':
            self.w.drone.set_ctrl_mass(D.firmware_mass(float(value)))
        elif name.startswith('ctrlMel.') and name[8:] in D.Drone.GAINS:
            self.w.drone.set_gain(name[8:], float(value))

    @property
    def high_level_enabled(self):
        return int(float(self.params['commander.enHighLevel'])) == 1

    # --- state helpers ------------------------------------------------------------------------
    def estimate(self):
        est = self.w.drone.last_estimate
        if est is None:
            pos, quat, vel, w = self.w.drone.true_state()
            est = self.w.drone.lh.observe(pos, quat, vel, w)
        return est

    def tell_state(self):
        p, q, v, _ = self.estimate()
        self.hl_state = poly7.TrajEval(np.array(p), np.array(v), np.zeros(3), quat_yaw(q), 0.)

    def current_goal(self, t):
        if self.planner in ('flying', 'landing') and self.piece is not None:
            return self.piece.eval(t)
        return None

    # --- high-level commands ------------------------------------------------------------------
    def _hl_ok(self, what):
        if not self.high_level_enabled:
            self.log.append((self.w.time, f'{what} ignored: commander.enHighLevel = 0'))
            return False
        return True

    def takeoff(self, height, duration, yaw):
        if not self._hl_ok('takeoff') or self.planner != 'idle':
            return
        p, q, _, _ = self.estimate()
        yaw = quat_yaw(q) if yaw is None else yaw
        self.piece = poly7.plan_takeoff_or_landing(self.w.time, p, quat_yaw(q), height, yaw, duration)
        self.planner = 'flying'

    def land(self, height, duration, yaw):
        if not self._hl_ok('land') or self.planner == 'landing':
            return
        goal = self.current_goal(self.w.time)
        if goal is None:
            self.tell_state()
            goal = self.hl_state
        yaw = goal.yaw if yaw is None else yaw
        self.piece = poly7.plan_takeoff_or_landing(self.w.time, goal.pos, goal.yaw, height, yaw, duration)
        self.planner = 'landing'

    def go_to(self, x, y, z, yaw, duration, relative, linear):
        if not self._hl_ok('go_to'):
            return
        t = self.w.time
        goal = self.current_goal(t)
        if goal is None:
            if self.hl_state is None:
                self.tell_state()
            goal = self.hl_state
        self.piece = poly7.plan_go_to_from(t, goal, [x, y, z], yaw, duration, relative, linear)
        self.planner = 'flying'

    def hl_stop(self):
        self.planner = 'idle'
        self.piece = None
        self.motors_on = False
        self.w.drone.cmd = None
        self.log.append((self.w.time, 'stop'))

    # --- low-level commands -------------------------------------------------------------------
    def low_level(self, setpoint):
        self.priority = PRIO_CRTP
        self.planner = 'idle'          # crtpCommanderHighLevelStop
        self.piece = None
        self.ll_setpoint = setpoint
        self.last_setpoint_t = self.w.time
        if self.armed and not self.locked:
            self.motors_on = setpoint is not None

    def relax(self):
        self.tell_state()
        self.priority = PRIO_DISABLE

    # --- per control tick -----------------------------------------------------------------------
    def tick(self):
        t = self.w.time
        dr = self.w.drone
        if self.high_level_enabled and self.priority <= PRIO_HIGHLEVEL:
            goal = self.current_goal(t)
            if goal is not None:
                if self.planner == 'landing' and self.piece.finished(t):
                    self.planner = 'idle'
                self.priority = PRIO_HIGHLEVEL
                self.ll_setpoint = D.full_state(goal.pos, goal.vel, goal.acc, goal.yaw, (0, 0, goal.yaw_rate))
                self.last_setpoint_t = t
                if self.armed and not self.locked:
                    self.motors_on = True
        if not (self.armed and not self.locked and self.motors_on) or self.ll_setpoint is None:
            dr.cmd, dr.level = None, False
            return
        age = t - self.last_setpoint_t
        if age > WDT_SHUTDOWN_S:
            self.locked, self.motors_on = True, False
            dr.cmd, dr.level = None, False
            self.log.append((t, 'watchdog: motors off'))
        elif age > WDT_LEVEL_S:
            if not dr.level:
                self.log.append((t, 'watchdog: level out'))
            dr.cmd, dr.level = self.ll_setpoint, True
        else:
            dr.cmd, dr.level = self.ll_setpoint, False

    def arm(self, do_arm):
        controller = int(float(self.params['stabilizer.controller']))
        if do_arm and controller != 2:
            raise RuntimeError('The simulator implements the Mellinger controller only; set '
                               'stabilizer.controller = 2 before arming')
        self.armed = bool(do_arm)
        if do_arm:
            self.locked = False
        else:
            self.motors_on = False


# ---- cflib duck types ---------------------------------------------------------------------------
class SimHighLevelCommander:
    def __init__(self, fw):
        self._fw = fw

    def takeoff(self, absolute_height_m, duration_s, group_mask=0, yaw=0.0):
        self._fw.takeoff(absolute_height_m, duration_s, yaw)

    def land(self, absolute_height_m, duration_s, group_mask=0, yaw=0.0):
        self._fw.land(absolute_height_m, duration_s, yaw)

    def stop(self, group_mask=0):
        self._fw.hl_stop()

    def go_to(self, x, y, z, yaw, duration_s, relative=False, linear=False, group_mask=0):
        self._fw.go_to(x, y, z, yaw, duration_s, relative, linear)

    def define_trajectory(self, trajectory_id, offset, n_pieces, type=0):
        raise NotImplementedError('Uploaded trajectories (cf.mem) are not simulated; the mission uses go_to')

    def start_trajectory(self, trajectory_id, time_scale=1.0, relative_position=False, relative_yaw=False,
                         reversed=False, group_mask=0):
        raise NotImplementedError('Uploaded trajectories (cf.mem) are not simulated; the mission uses go_to')


class SimCommander:
    def __init__(self, fw):
        self._fw = fw

    def send_position_setpoint(self, x, y, z, yaw):
        self._fw.low_level(D.full_state([x, y, z], yaw=math.radians(yaw)))

    def send_full_state_setpoint(self, pos, vel, acc, orientation, rollrate, pitchrate, yawrate):
        sp = D.full_state(pos, vel, acc, 0., np.radians([rollrate, pitchrate, yawrate]))
        sp[9:13] = orientation         # cflib order [qx, qy, qz, qw] == crazyflow xyzw
        self._fw.low_level(sp)

    def send_notify_setpoint_stop(self, remain_valid_milliseconds=0):
        self._fw.relax()

    def send_stop_setpoint(self):
        self._fw.low_level(None)


class _Toc:
    def __init__(self, params):
        self._params = params

    def get_element_by_complete_name(self, complete_name):
        return complete_name if complete_name in self._params else None


class SimParam:
    def __init__(self, fw):
        self._fw = fw
        self._callbacks = []
        self.toc = _Toc(fw.params)

    def set_value(self, complete_name, value):
        self._fw.set_param(complete_name, value)
        group, name = complete_name.split('.', 1)
        for g, n, cb in list(self._callbacks):
            if (g is None or g == group) and (n is None or n == name):
                cb(complete_name, str(value))

    def get_value(self, complete_name, timeout=60):
        return self._fw.params[complete_name]

    def add_update_callback(self, group=None, name=None, cb=None):
        self._callbacks.append((group, name, cb))

    def remove_update_callback(self, group, name=None, cb=None):
        self._callbacks = [c for c in self._callbacks if c != (group, name, cb)]


class SimLogConfig:
    def __init__(self, name, period_in_ms):
        self.name, self.period_in_ms = name, period_in_ms
        self.variables = []
        self.data_received_cb = Caller()
        self.error_cb = Caller()
        self.started_cb = Caller()
        self.added_cb = Caller()
        self._log = None
        self.started = False
        self._next = 0.

    def add_variable(self, name, fetch_as=None):
        self.variables.append(name)

    def start(self):
        self.started = True
        self._next = self._log.world.time if self._log else 0.

    def stop(self):
        self.started = False

    def delete(self):
        self.started = False
        if self._log is not None:
            self._log.configs.remove(self)


class SimLog:
    def __init__(self, world):
        self.world = world
        self.configs = []

    def add_config(self, logconf):
        logconf._log = self
        self.configs.append(logconf)

    def tick(self):
        t = self.world.time
        for c in self.configs:
            if c.started and t >= c._next - 1e-9:
                c._next = t + c.period_in_ms/1000
                values = self.world.log_values()
                c.data_received_cb.call(int(t*1000), {v: values.get(v, 0.) for v in c.variables}, c)


class SimSupervisor:
    def __init__(self, fw):
        self._fw = fw

    def send_arming_request(self, do_arm: bool):
        self._fw.arm(do_arm)

    @property
    def is_armed(self):
        return self._fw.armed


class SimCrazyflie:
    """cflib ``Crazyflie`` stand-in; ``world`` is the extra, keyword-only binding to the simulator."""

    def __init__(self, link=None, ro_cache=None, rw_cache=None, *, world=None):
        if world is None:
            raise ValueError('SimCrazyflie needs world=SimWorld(...)')
        self.world = world
        self.fw = world.fw
        self.high_level_commander = SimHighLevelCommander(self.fw)
        self.commander = SimCommander(self.fw)
        self.param = SimParam(self.fw)
        self.log = world.log
        self.supervisor = SimSupervisor(self.fw)

    def is_connected(self):
        return True


class SimSyncCrazyflie:
    def __init__(self, link_uri, cf=None):
        if cf is None:
            raise ValueError('SimSyncCrazyflie needs cf=SimCrazyflie(world=...)')
        self._link_uri = link_uri
        self.cf = cf

    def __enter__(self):
        self.open_link()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_link()

    def open_link(self):
        pass

    def close_link(self):
        pass

    def wait_for_params(self):
        pass

    def is_params_updated(self):
        return True

    def is_link_open(self):
        return True


# ---- world and clock ----------------------------------------------------------------------------
class SimClock:
    def __init__(self, world):
        self.world = world

    def time(self):
        return self.world.time

    def sleep(self, dt):
        self.world.advance(dt)


@dataclass(frozen=True)
class CameraModel:
    sigma_pos_mm: float = 0.5       # assumption: overhead RealSense pose noise per axis
    sigma_angle_deg: float = 0.3    # assumption
    occlusion_radius_mm: float = 70.  # bricks under the drone's footprint are hidden from above


class SimPoseSource:
    """Overhead camera: noisy poses (Lighthouse frame) of every visible brick, bricks under the drone hidden."""

    def __init__(self, world, model: CameraModel, rng):
        self.w, self.m, self.rng = world, model, rng

    def poses(self):
        from .cf.rot import quat_to_mat, mat_to_quat, rotvec_to_mat
        from .quat import wxyz_to_xyzw, xyzw_to_wxyz
        out = {}
        dpos = self.w.drone.true_state()[0]/MM
        for name, j in self.w.brick_index.items():
            if not self.w.active[j]:
                continue
            pos, quat = body_pose(self.w.data, self.w.info, j)
            if np.hypot(*(pos[:2] - dpos[:2])) < self.m.occlusion_radius_mm and pos[2] < dpos[2]:
                continue
            noisy = pos + self.rng.normal(0, self.m.sigma_pos_mm, 3)
            R = rotvec_to_mat(np.radians(self.rng.normal(0, self.m.sigma_angle_deg, 3))) @ quat_to_mat(wxyz_to_xyzw(quat))
            out[name] = Pose(tuple(noisy*MM), tuple(xyzw_to_wxyz(mat_to_quat(R))))
        return out


class SimWorld:
    """MuJoCo model + drone + firmware emulation; ``advance(dt)`` is the only way time moves."""

    def __init__(self, model, data, info, drone, brick_index, record_fps=0, battery_s=600.):
        self.model, self.data, self.info, self.drone = model, data, info, drone
        self.brick_index = brick_index           # name -> pool body index
        self.active = [False]*len(info['bodies'])
        self.fw = Firmware(self)
        self.log = SimLog(self)
        self.frames = [] if record_fps else None
        self.record_every = max(1, round(1/(record_fps*model.opt.timestep))) if record_fps else 0
        self.state_every = drone.state_every
        self.step = 0
        self.battery_s = battery_s
        self.flight_s = 0.
        self.hooks = []                          # callables(world) every control tick

    @property
    def time(self):
        return self.data.time

    def log_values(self):
        p, q, _, _ = self.fw.estimate()
        frac = min(1., self.flight_s/self.battery_s)
        return {'stateEstimate.x': float(p[0]), 'stateEstimate.y': float(p[1]), 'stateEstimate.z': float(p[2]),
                'stateEstimate.yaw': math.degrees(quat_yaw(q)), 'pm.vbat': 4.15 - 0.55*frac,
                'supervisor.info': float(self.fw.armed) + 2*float(self.fw.locked)}

    def advance(self, dt):
        n = max(0, int(round(dt/self.model.opt.timestep)))
        for _ in range(n):
            if self.step % self.state_every == 0:
                self.fw.tick()
                self.log.tick()
                for h in self.hooks:
                    h(self)
            self.drone.before_step()
            mujoco.mj_step(self.model, self.data)
            if self.fw.motors_on:
                self.flight_s += self.model.opt.timestep
            if self.frames is not None and self.step % self.record_every == 0:
                self.frames.append(self.data.qpos.copy())
            self.step += 1
