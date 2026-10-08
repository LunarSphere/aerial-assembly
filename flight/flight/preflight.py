"""Hardware preflight, in the style of ``crazyfly/hover.py``: acknowledged param writes,
deck detection, Kalman reset and variance convergence, battery, and the take-off pose.

Differences from hover.py: the Lighthouse deck (``deck.bcLighthouse4``) replaces the Flow
deck, the Mellinger controller (2) replaces PID so the real drone runs the controller the
simulator ports from crazyflow, the high-level commander is enabled, and ``pm.vbat`` is
checked. Time goes through the injected clock.
"""
from collections import deque
from contextlib import contextmanager
import math
from queue import Empty, Queue

import numpy as np

PREFLIGHT_TIMEOUT = 30.0
POSITION_NAMES = ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z')
VARIANCE_NAMES = ('kalman.varPX', 'kalman.varPY', 'kalman.varPZ')
VBAT_MIN = 3.8   # V, 1S LiPo before take-off


class PreflightError(RuntimeError):
    pass


def wait_for_parameters(scf, clock, timeout=PREFLIGHT_TIMEOUT):
    deadline = clock.time() + timeout
    while not scf.is_params_updated():
        if clock.time() >= deadline:
            raise TimeoutError('Parameter download timed out; no arming requested')
        clock.sleep(0.1)


def set_parameter(cf, name, value, timeout=5.0):
    """Wait for the firmware's write acknowledgement, not a cached get_value()."""
    replies = Queue()
    group, field = name.split('.', 1)

    def updated(_name, received):
        replies.put(received)

    cf.param.add_update_callback(group=group, name=field, cb=updated)
    try:
        cf.param.set_value(name, str(value))
        try:
            received = replies.get(timeout=timeout)   # acknowledgements arrive on cflib's thread
        except Empty as exc:
            raise TimeoutError(f'No acknowledgement for {name}') from exc
        if not math.isclose(float(received), float(value), rel_tol=1e-6, abs_tol=1e-9):
            raise PreflightError(f'{name}: requested {value}, received {received}')
    finally:
        cf.param.remove_update_callback(group=group, name=field, cb=updated)


@contextmanager
def samples(cf, LogConfig, clock, names, period_ms=200):
    queue = Queue()
    config = LogConfig(name='Brixzle preflight', period_in_ms=period_ms)
    for n in names:
        config.add_variable(n, 'float')

    def received(_timestamp, data, _config):
        queue.put((clock.time(), data))

    def error(_config, message):
        queue.put((clock.time(), RuntimeError(f'Telemetry: {message}')))

    config.data_received_cb.add_callback(received)
    config.error_cb.add_callback(error)
    cf.log.add_config(config)
    try:
        config.start()
        yield queue
    finally:
        try:
            if cf.is_connected():
                config.delete()
        finally:
            config.data_received_cb.remove_callback(received)
            config.error_cb.remove_callback(error)


def wait_for_position(cf, queue, clock, timeout=PREFLIGHT_TIMEOUT):
    """Bitcraze variance criterion (spread < 0.001 over 10 samples) plus a stationary position."""
    deadline = clock.time() + timeout
    history = deque(maxlen=10)
    while clock.time() < deadline:
        if not cf.is_connected():
            raise ConnectionError('Disconnected during preflight')
        try:
            received_at, data = queue.get_nowait()
        except Empty:
            clock.sleep(0.05)
            continue
        if isinstance(data, Exception):
            raise data
        values = tuple(float(data[n]) for n in VARIANCE_NAMES + POSITION_NAMES)
        if not all(math.isfinite(v) for v in values) or min(values[:3]) < 0:
            raise PreflightError('Invalid estimator telemetry; no arming requested')
        history.append(values)
        if len(history) == 10:
            cols = list(zip(*history))
            if all(max(c) - min(c) < 0.001 for c in cols[:3]) and all(max(c) - min(c) < 0.02 for c in cols[3:]):
                return np.array([sum(c)/len(c) for c in cols[3:]])
    raise TimeoutError('No stable Lighthouse position within 30 s; no arming requested')


def battery(cf, queue, clock, n=5, timeout=5.0):
    values, deadline = [], clock.time() + timeout
    while len(values) < n and clock.time() < deadline:
        try:
            _, data = queue.get_nowait()
        except Empty:
            clock.sleep(0.05)
            continue
        if isinstance(data, Exception):
            raise data
        values.append(float(data['pm.vbat']))
    if not values:
        raise TimeoutError('No battery telemetry')
    return sum(values)/len(values)


def run_preflight(scf, plan, LogConfig, clock, vbat_min=VBAT_MIN, home_tolerance=0.15):
    """Return a report dict; raise PreflightError (no arming) on any failed check."""
    cf = scf.cf
    wait_for_parameters(scf, clock)
    if cf.param.toc.get_element_by_complete_name('deck.bcLighthouse4') is None or \
            int(float(cf.param.get_value('deck.bcLighthouse4'))) != 1:
        raise PreflightError('No Lighthouse deck detected (deck.bcLighthouse4); no arming requested')
    set_parameter(cf, 'stabilizer.estimator', 2)    # Kalman (required by the Lighthouse deck)
    set_parameter(cf, 'stabilizer.controller', 2)   # Mellinger, as simulated (crazyflow port)
    set_parameter(cf, 'commander.enHighLevel', 1)
    for name, value in plan['settings'].get('params', {}).items():
        set_parameter(cf, name, value)
    set_parameter(cf, 'kalman.resetEstimation', 1)
    clock.sleep(0.1)
    set_parameter(cf, 'kalman.resetEstimation', 0)
    with samples(cf, LogConfig, clock, VARIANCE_NAMES + POSITION_NAMES, period_ms=500) as q:
        pos = wait_for_position(cf, q, clock)
    with samples(cf, LogConfig, clock, ('pm.vbat',)) as q:
        vbat = battery(cf, q, clock)
    if vbat < vbat_min:
        raise PreflightError(f'Battery {vbat:.2f} V < {vbat_min} V; no arming requested')
    home = np.asarray(plan['home']['pos'])
    if np.linalg.norm(pos[:2] - home[:2]) > home_tolerance or abs(pos[2] - home[2]) > home_tolerance:
        raise PreflightError(f'Drone at {np.round(pos, 3).tolist()} is not on the stand at '
                             f'{np.round(home, 3).tolist()}; no arming requested')
    lo, hi = np.asarray(plan['settings']['geofence']['min']), np.asarray(plan['settings']['geofence']['max'])
    if np.any(pos < lo) or np.any(pos > hi):
        raise PreflightError('Drone outside the geofence; no arming requested')
    return {'position_m': pos.tolist(), 'vbat_V': vbat}
