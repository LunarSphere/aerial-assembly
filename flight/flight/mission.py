"""Pick-and-place mission: executes a compiled ``plan.json`` through the cflib API only.

Per brick: GOTO_PREPICK -> ALIGN -> ENGAGE -> LIFT -> VERIFY_PICK -> TRANSIT -> PREPLACE ->
DESCEND -> RELEASE -> RETREAT -> VERIFY_PLACE -> (next | RETRY | ABORT). The plan gives each
phase as segments in a brick frame (the camera-measured pick pose, or the seat implied by
the camera-measured supporter); this module resolves frames at run time and issues:

* ``high_level_commander.takeoff/go_to/land/stop`` for free flight (firmware 7th-order pieces),
* streamed ``commander.send_full_state_setpoint`` (>= ``setpoint_hz``) for slow contact moves,
  then ``send_notify_setpoint_stop`` before handing back to the high-level commander,
* ``param.set_value`` for estimator/controller selection and ``ctrlMel.mass`` (payload).

Time only advances through ``clock.sleep``. The grasp is never sensed directly: it is
inferred from the overhead camera (``BrickPoseSource``).
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .frames import Pose

PLAN_VERSION = 1
LOG_VARS = ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z', 'stateEstimate.yaw', 'pm.vbat')
SETUP_PARAMS = (('stabilizer.estimator', 2), ('stabilizer.controller', 2), ('commander.enHighLevel', 1))


class MissionAbort(RuntimeError):
    pass


def load_plan(plan):
    if plan.get('version') != PLAN_VERSION:
        raise ValueError(f'Unsupported plan version {plan.get("version")!r}')
    for b in plan['bricks']:
        for key in ('name', 'pick', 'place', 'carry'):
            if key not in b:
                raise ValueError(f'Brick entry missing {key!r}')
    return plan


def _quat_yaw(yaw):
    return [0., 0., math.sin(yaw/2), math.cos(yaw/2)]   # cflib orientation order [qx, qy, qz, qw]


@dataclass
class MissionResult:
    status: str = 'complete'
    bricks: list = field(default_factory=list)
    events: list = field(default_factory=list)
    battery_swaps: int = 0


class Mission:
    """Run with ``Mission(scf, plan, clock, poses, LogConfig).run()``.

    ``LogConfig`` is the backend's class (cflib's, or the simulator's duck type).
    ``on_event(dict)`` sees every phase boundary; ``on_battery_swap()`` is called while landed.
    """

    def __init__(self, scf, plan, clock, poses, LogConfig, on_event=None, on_battery_swap=None, keep_going=False,
                 arm=True):
        self.scf, self.cf = scf, scf.cf
        self.plan = load_plan(plan)
        self.s = plan['settings']
        self.clock, self.poses, self.LogConfig = clock, poses, LogConfig
        self.on_event = on_event or (lambda e: None)
        self.on_battery_swap = on_battery_swap or (lambda: None)
        self.keep_going, self.arm = keep_going, arm
        self.result = MissionResult()
        self.state = {}
        self.cmd_pos, self.cmd_yaw = None, 0.
        self.low_level = False
        self.airborne_since = None
        self.flying = False

    # ---- plumbing ---------------------------------------------------------------------------
    def event(self, **e):
        e = {'t': self.clock.time(), **e}
        self.result.events.append(e)
        self.on_event(e)

    def _on_log(self, timestamp, data, logconf):
        self.state.update(data)

    def start_logging(self):
        conf = self.LogConfig(name='Mission', period_in_ms=50)
        for v in LOG_VARS:
            conf.add_variable(v, 'float')
        self.cf.log.add_config(conf)
        conf.data_received_cb.add_callback(self._on_log)
        conf.start()
        self.log_conf = conf

    def estimate(self):
        return np.array([self.state.get(f'stateEstimate.{a}', np.nan) for a in 'xyz'])

    def set_param(self, name, value):
        self.cf.param.set_value(name, str(value))

    def setup(self):
        for name, value in SETUP_PARAMS:
            self.set_param(name, value)
        for name, value in self.s.get('params', {}).items():
            self.set_param(name, value)
        self.set_param('ctrlMel.mass', self.s['ctrl_mass_kg']['empty'])
        self.start_logging()
        self.clock.sleep(0.2)

    # ---- motion primitives ------------------------------------------------------------------
    def _to_high_level(self):
        if self.low_level:
            self.cf.commander.send_notify_setpoint_stop()
            self.low_level = False

    def go_to(self, pos, yaw, duration):
        self._to_high_level()
        self.cf.high_level_commander.go_to(float(pos[0]), float(pos[1]), float(pos[2]), float(yaw), float(duration))
        self.clock.sleep(duration + self.s.get('settle_s', 0.3))
        self.cmd_pos, self.cmd_yaw = np.asarray(pos, float), yaw

    def line(self, pos, yaw, duration):
        """Stream a straight constant-speed setpoint ramp (velocity feed-forward) at setpoint_hz."""
        rate = self.s['setpoint_hz']
        p0 = self.estimate() if self.cmd_pos is None else self.cmd_pos
        p1 = np.asarray(pos, float)
        n = max(1, math.ceil(duration*rate))
        vel = (p1 - p0)/max(duration, 1e-9)
        for k in range(1, n + 1):
            p = p0 + (p1 - p0)*k/n
            v = vel if k < n else np.zeros(3)
            self.cf.commander.send_full_state_setpoint(p.tolist(), v.tolist(), [0., 0., 0.], _quat_yaw(yaw), 0., 0., 0.)
            self.low_level = True
            self.clock.sleep(1/rate)
        self.cmd_pos, self.cmd_yaw = p1, yaw

    def hold(self, duration):
        self.line(self.cmd_pos, self.cmd_yaw, duration)

    def run_segments(self, frame: Pose, segments, brick):
        for seg in segments:
            op = seg['op']
            self.event(brick=brick, phase=seg['phase'], op=op)
            if op == 'param':
                self.set_param(seg['name'], seg['value'])
                continue
            if op == 'wait':
                if self.low_level:
                    self.hold(seg['duration'])
                else:
                    self.clock.sleep(seg['duration'])
                continue
            if op == 'hold':
                self.hold(seg['duration'])
                continue
            pos = frame.apply(seg['to'])
            if 'z_world' in seg:
                pos[2] = seg['z_world']
            yaw = frame.yaw + seg.get('yaw', 0.)
            if op == 'go_to':
                self.go_to(pos, yaw, seg['duration'])
            elif op == 'line':
                self.line(pos, yaw, seg['duration'])
            else:
                raise ValueError(f'Unknown segment op {op!r}')

    # ---- frames from the camera -------------------------------------------------------------
    def seat_frame(self, spec, poses):
        """Seat pose: the measured supporter's pose composed with the planned relative pose."""
        sup = spec['supporter']
        if sup in self.plan.get('bases', {}):
            ref = Pose.from_dict(self.plan['bases'][sup])
        elif sup in poses:
            ref = poses[sup]
        else:
            return Pose.from_dict(spec['nominal'])
        return ref.compose(Pose.from_dict(spec['rel']))

    def pick_frame(self, brick, poses):
        if brick['name'] in poses:
            return poses[brick['name']]
        raise MissionAbort(f'camera does not see brick {brick["name"]}')

    def seat_error(self, brick, poses):
        """(dx, dy, dz) in the seat frame (m) and angle (deg) of the measured brick."""
        seat = self.seat_frame(brick['place']['frame'], poses)
        b = poses.get(brick['name'])
        if b is None:
            return None, None
        local = seat.inverse().apply(b.pos)
        return local, seat.angle_to(b)

    # ---- the per-brick state machine ---------------------------------------------------------
    def verify_pick(self, brick, before):
        """Camera-only grasp check: the slot is empty and the brick is not seen anywhere else.

        From above, a carried brick is hidden under the drone, so "not visible" after a pick
        means "on the gripper"; a brick still seen near its slot (or elsewhere) is a miss.
        """
        tol = self.s['tolerance']
        now = self.poses.poses().get(brick['name'])
        if now is None:
            return True, {'state': 'hidden_under_drone'}
        moved = float(np.linalg.norm(np.subtract(now.pos, before.pos)))
        return False, {'state': 'in_slot' if moved < tol['pick_gone_m'] else 'displaced', 'moved_m': moved}

    def verify_place(self, brick):
        tol = self.s['tolerance']
        poses = self.poses.poses()
        err, angle = self.seat_error(brick, poses)
        if err is None:
            return 'missing', None, None
        seated = (abs(err[0]) <= tol['seat_xz_m'] and abs(err[2]) <= tol['seat_xz_m']
                  and abs(err[1]) <= tol['seat_y_m'] and angle <= tol['seat_angle_deg'])
        if seated:
            return 'seated', err, angle
        near = float(np.linalg.norm(err)) <= tol['recover_m']
        return ('recoverable' if near else 'lost'), err, angle

    def pick(self, brick, retries):
        for attempt in range(retries + 1):
            self.event(brick=brick['name'], phase='PICK_REQUEST', attempt=attempt)
            poses = self.poses.poses()
            before = self.pick_frame(brick, poses)
            self.run_segments(before, brick['pick']['segments'], brick['name'])
            ok, info = self.verify_pick(brick, before)
            self.event(brick=brick['name'], phase='VERIFY_PICK', ok=ok, attempt=attempt, info=info)
            if ok:
                return True
            # Back away from wherever the brick ended up before retrying.
            self.run_segments(before, brick['pick'].get('abort', []), brick['name'])
        return False

    def place_brick(self, brick):
        name = brick['name']
        record = {'brick': name, 'pick_attempts': 0, 'place_attempts': 0}
        if not self.pick(brick, self.s['max_pick_retries']):
            record['outcome'] = 'pick_failed'
            return record
        for attempt in range(self.s['max_place_retries'] + 1):
            record['place_attempts'] = attempt + 1
            poses = self.poses.poses()
            seat = self.seat_frame(brick['place']['frame'], poses)
            self.event(brick=name, phase='PREPLACE_TARGET', seat=seat.to_dict())
            self.run_segments(seat, brick['place']['segments'], name)
            status, err, angle = self.verify_place(brick)
            self.event(brick=name, phase='VERIFY_PLACE', status=status,
                       err_m=None if err is None else list(map(float, err)), angle_deg=angle)
            if status == 'seated':
                record['outcome'] = 'seated'
                return record
            if status != 'recoverable' or attempt == self.s['max_place_retries']:
                break
            if not self.pick(brick, 0):        # re-grasp the brick where the camera sees it
                break
        record['outcome'] = 'place_failed'
        return record

    # ---- whole mission ----------------------------------------------------------------------
    def takeoff(self):
        if self.arm:
            self.cf.supervisor.send_arming_request(True)
            self.clock.sleep(1.0)
        self.event(phase='TAKEOFF')
        self.cf.high_level_commander.takeoff(self.s['takeoff_z'], self.s['takeoff_s'])
        self.clock.sleep(self.s['takeoff_s'] + 0.5)
        home = self.plan['home']
        self.cmd_pos, self.cmd_yaw = np.array([*home['pos'][:2], self.s['takeoff_z']]), home.get('yaw', 0.)
        self.flying = True
        self.airborne_since = self.clock.time()

    def land(self):
        self._to_high_level()
        home = self.plan['home']
        self.event(phase='LAND')
        hz = self.s['transit_z']
        if self.cmd_pos is not None and self.cmd_pos[2] < hz:
            self.go_to([*self.cmd_pos[:2], hz], self.cmd_yaw, 1.5)
        self.go_to([home['pos'][0], home['pos'][1], hz], home.get('yaw', 0.), self.s['return_s'])
        self.go_to([home['pos'][0], home['pos'][1], self.s['takeoff_z']], home.get('yaw', 0.), 1.5)
        self.cf.high_level_commander.land(home['pos'][2], self.s['land_s'])
        self.clock.sleep(self.s['land_s'] + 0.5)
        self.cf.high_level_commander.stop()
        self.flying = False
        if self.arm:
            self.cf.supervisor.send_arming_request(False)

    def battery_ok(self, brick):
        b = self.s['battery']
        used = self.clock.time() - self.airborne_since
        return used + b['per_brick_s'] + b['reserve_s'] <= b['flight_time_s']

    def run(self):
        self.setup()
        self.takeoff()
        try:
            for brick in self.plan['bricks']:
                if not self.battery_ok(brick):
                    self.land()
                    self.result.battery_swaps += 1
                    self.event(phase='BATTERY_SWAP')
                    self.on_battery_swap()
                    self.takeoff()
                rec = self.place_brick(brick)
                self.result.bricks.append(rec)
                if rec['outcome'] != 'seated':
                    self.result.status = f'{rec["outcome"]}:{brick["name"]}'
                    if not self.keep_going:
                        break
        except MissionAbort as e:
            self.result.status = f'abort:{e}'
        except BaseException:
            # Ctrl+C or any unexpected error: land where we are, then stop the motors.
            self.result.status = 'interrupted'
            if self.flying:
                self.flying = False
                self.emergency_land()
            raise
        if self.flying:
            self.land()
        return self.result

    def emergency_land(self):
        """Ctrl+C / exception path: hand back to the high-level commander, land in place, stop."""
        try:
            self._to_high_level()
            self.cf.high_level_commander.land(self.s.get('emergency_land_z', 0.08), self.s['land_s'])
            self.clock.sleep(self.s['land_s'] + 0.5)
        finally:
            self.cf.high_level_commander.stop()
            if self.arm:
                self.cf.supervisor.send_arming_request(False)
