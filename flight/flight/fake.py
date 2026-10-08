"""Recording stand-ins for the cflib objects, for dry runs and parity tests (no radio, no physics).

``RecordingCrazyflie`` records every mission call as (method, args) and reports a perfect
state estimate that follows the commanded setpoints (high-level pieces from ``poly7``).
``ScriptedPoseSource`` plays the camera for a mission where every pick and place succeeds.
"""
import math

import numpy as np

from cflib.utils.callbacks import Caller

from . import poly7
from .frames import Pose


def _f(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_f(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return round(float(v), 9) if isinstance(v, (int, float, np.floating, np.integer)) else v


class Recorder:
    def __init__(self, clock=None):
        self.clock = clock
        self.calls = []

    def add(self, name, *args):
        self.calls.append((name, *[_f(a) for a in args]))


def _norm(v):
    if hasattr(v, 'variables') and hasattr(v, 'period_in_ms'):
        return ['LogConfig', v.name, v.period_in_ms, list(v.variables)]
    if callable(v):
        return 'callback'
    return v


class RecordingProxy:
    """Wraps a cflib-like object; records every public method call with its full bound arguments
    (defaults applied, in signature order) under ``prefix.method``, then forwards it."""

    def __init__(self, target, prefix, recorder):
        self._target, self._prefix, self._rec = target, prefix, recorder

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if name.startswith('_') or not callable(attr):
            return attr
        import inspect
        sig = inspect.signature(attr)

        def call(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            self._rec.add(f'{self._prefix}.{name}', *[_norm(v) for v in bound.arguments.values()])
            return attr(*args, **kwargs)
        return call


SUBOBJECTS = (('high_level_commander', 'hl'), ('commander', 'cmd'), ('param', 'param'), ('log', 'log'),
              ('supervisor', 'supervisor'))


def record(cf, recorder):
    """Replace cf's API sub-objects by recording proxies (in place); returns cf."""
    for attr, prefix in SUBOBJECTS:
        setattr(cf, attr, RecordingProxy(getattr(cf, attr), prefix, recorder))
    return cf


class _HL:
    def __init__(self, cf):
        self._cf = cf

    def takeoff(self, absolute_height_m, duration_s, group_mask=0, yaw=0.0):
        p = self._cf.position.copy()
        self._cf.piece = poly7.plan_takeoff_or_landing(self._cf.clock.time(), p, 0., absolute_height_m, yaw or 0.,
                                                       duration_s)

    def land(self, absolute_height_m, duration_s, group_mask=0, yaw=0.0):
        self._cf.piece = poly7.plan_takeoff_or_landing(self._cf.clock.time(), self._cf.current(), 0.,
                                                       absolute_height_m, yaw or 0., duration_s)

    def go_to(self, x, y, z, yaw, duration_s, relative=False, linear=False, group_mask=0):
        cur = poly7.TrajEval(self._cf.current(), np.zeros(3), np.zeros(3), 0., 0.)
        self._cf.piece = poly7.plan_go_to_from(self._cf.clock.time(), cur, [x, y, z], yaw, duration_s, relative, linear)

    def stop(self, group_mask=0):
        self._cf.piece = None

    def define_trajectory(self, trajectory_id, offset, n_pieces, type=0):
        pass

    def start_trajectory(self, trajectory_id, time_scale=1.0, relative_position=False, relative_yaw=False,
                         reversed=False, group_mask=0):
        pass


class _Commander:
    def __init__(self, cf):
        self._cf = cf

    def send_position_setpoint(self, x, y, z, yaw):
        self._cf.piece, self._cf.position = None, np.array([x, y, z], float)

    def send_full_state_setpoint(self, pos, vel, acc, orientation, rollrate, pitchrate, yawrate):
        self._cf.piece, self._cf.position = None, np.array(pos, float)

    def send_notify_setpoint_stop(self, remain_valid_milliseconds=0):
        pass

    def send_stop_setpoint(self):
        pass


class _Param:
    def __init__(self, cf):
        self._cf = cf
        self.values = {'deck.bcLighthouse4': '1'}
        self._cbs = []

    def set_value(self, complete_name, value):
        self.values[complete_name] = str(value)
        for g, n, cb in self._cbs:
            if (g is None or complete_name.startswith(g + '.')) and (n is None or complete_name.endswith('.' + n)):
                cb(complete_name, str(value))

    def get_value(self, complete_name, timeout=60):
        return self.values.get(complete_name, '0')

    def add_update_callback(self, group=None, name=None, cb=None):
        self._cbs.append((group, name, cb))

    def remove_update_callback(self, group, name=None, cb=None):
        self._cbs = [c for c in self._cbs if c != (group, name, cb)]


class _Toc:
    def __init__(self, param):
        self._param = param

    def get_element_by_complete_name(self, complete_name):
        return complete_name if complete_name in self._param.values or complete_name.startswith(
            ('stabilizer.', 'commander.', 'ctrlMel.', 'kalman.')) else None


class _Supervisor:
    def __init__(self, cf):
        self._cf = cf
        self.is_armed = False

    def send_arming_request(self, do_arm: bool):
        self.is_armed = bool(do_arm)


class FakeLogConfig:
    def __init__(self, name, period_in_ms):
        self.name, self.period_in_ms = name, period_in_ms
        self.variables = []
        self.data_received_cb = Caller()
        self.error_cb = Caller()
        self.cf = None

    def add_variable(self, name, fetch_as=None):
        self.variables.append(name)

    def start(self):
        pass

    def stop(self):
        pass

    def delete(self):
        pass


class _Log:
    def __init__(self, cf):
        self._cf = cf
        self.configs = []

    def add_config(self, logconf):
        logconf.cf = self._cf
        self.configs.append(logconf)


class RecordingCrazyflie:
    """Kinematic, recording ``Crazyflie``: state estimate = commanded setpoint."""

    def __init__(self, link=None, ro_cache=None, rw_cache=None, *, clock=None, home=(0., 0., 0.)):
        self.clock = clock
        self.rec = Recorder(clock)
        self.position = np.asarray(home, float)
        self.piece = None
        self.high_level_commander = _HL(self)
        self.commander = _Commander(self)
        self.param = _Param(self)
        self.param.toc = _Toc(self.param)
        self.supervisor = _Supervisor(self)
        self.log = _Log(self)
        record(self, self.rec)

    def current(self):
        if self.piece is not None:
            self.position = self.piece.eval(self.clock.time()).pos
        return self.position.copy()

    def is_connected(self):
        return True

    def publish(self):
        """Push one log sample to every started config (call after each clock advance)."""
        p = self.current()
        data = {'stateEstimate.x': p[0], 'stateEstimate.y': p[1], 'stateEstimate.z': p[2], 'stateEstimate.yaw': 0.,
                'pm.vbat': 4.1, 'kalman.varPX': 1e-4, 'kalman.varPY': 1e-4, 'kalman.varPZ': 1e-4}
        for c in self.log._target.configs:
            c.data_received_cb.call(int(self.clock.time()*1000), {v: data.get(v, 0.) for v in c.variables}, c)


class PublishingClock:
    """Manual clock that pushes the fake's log after every sleep."""

    def __init__(self, t0=0.):
        self.t = t0
        self.cf = None

    def time(self):
        return self.t

    def sleep(self, dt):
        self.t += max(0., dt)
        if self.cf is not None:
            self.cf.publish()


class FakeSyncCrazyflie:
    def __init__(self, link_uri, cf=None):
        self._link_uri, self.cf = link_uri, cf

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def open_link(self):
        pass

    def close_link(self):
        pass

    def wait_for_params(self):
        pass

    def is_params_updated(self):
        return True


class ScriptedPoseSource:
    """Camera for an all-success mission: fixture bricks at their slots until picked, hidden while
    carried, at the nominal seat (relative to the nominal supporter) after placing."""

    def __init__(self, plan):
        self.plan = plan
        self.state = {b['name']: 'slot' for b in plan['bricks']}
        self.current = None

    def on_event(self, e):
        name, phase = e.get('brick'), e.get('phase')
        if name is None:
            return
        if phase == 'LIFT':
            self.state[name] = 'carried'
        elif phase == 'RETREAT':
            self.state[name] = 'placed'

    def poses(self):
        out = {}
        for b in self.plan['bricks']:
            st = self.state[b['name']]
            if st == 'slot':
                out[b['name']] = Pose.from_dict(b['pick']['slot_nominal'])
            elif st == 'placed':
                out[b['name']] = Pose.from_dict(b['place']['frame']['nominal'])
        return out
