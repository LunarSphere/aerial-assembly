#!/usr/bin/env python3
"""Flow deck takeoff, five-second hover, forward flight, and landing."""

import argparse
from collections import deque
from contextlib import contextmanager
import math
from queue import Empty, Queue
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

PREFLIGHT_TIMEOUT = 30.0
RAMP_SECONDS = 3.0
SETPOINT_PERIOD = 0.05
POSITION_NAMES = tuple(f"stateEstimate.{axis}" for axis in "xyz")
VARIANCE_NAMES = tuple(f"kalman.varP{axis}" for axis in "XYZ")


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def parse_args(argv=None, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--uri", default="radio://0/80/2M",
                        help="radio URI (default: radio://0/80/2M)")
    parser.add_argument("--height", type=positive_float, default=0.30,
                        help="rise above launch position in metres (default: 0.30)")
    parser.add_argument("--duration", type=positive_float, default=5.0,
                        help="hover duration in seconds (default: 5)")
    parser.add_argument("--distance", type=positive_float, default=0.25,
                        help="forward distance in metres (default: 0.25)")
    parser.add_argument("--speed", type=positive_float, default=0.2,
                        help="forward speed in metres/second (default: 0.2)")
    parser.add_argument("--check-only", action="store_true",
                        help="run this script's preflight without arming")
    return parser.parse_args(argv)


def connected_sleep(cf, seconds):
    deadline = time.monotonic() + seconds
    while True:
        if not cf.is_connected():
            raise ConnectionError("Radio link lost; scripted landing is unavailable")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def wait_for_parameters(scf):
    deadline = time.monotonic() + PREFLIGHT_TIMEOUT
    while not scf.is_params_updated():
        if time.monotonic() >= deadline:
            raise TimeoutError("Parameter download timed out; no arming requested")
        connected_sleep(scf.cf, 0.1)


def set_parameter(cf, name, value):
    """Wait for the firmware's write acknowledgement, not a cached get_value()."""
    replies = Queue()
    group, field = name.split(".", 1)

    def updated(_name, received):
        replies.put(received)

    cf.param.add_update_callback(group=group, name=field, cb=updated)
    try:
        cf.param.set_value(name, str(value))
        try:
            received = replies.get(timeout=5.0)
        except Empty as exc:
            raise TimeoutError(f"No acknowledgement for {name}") from exc
        if float(received) != float(value):
            raise RuntimeError(f"{name}: requested {value}, received {received}")
    finally:
        cf.param.remove_update_callback(group=group, name=field, cb=updated)


@contextmanager
def position_samples(cf):
    """Six floats fit in one 24-byte log payload; callbacks never block."""
    samples = Queue()
    config = LogConfig(name="Hover preflight", period_in_ms=500)
    for name in VARIANCE_NAMES + POSITION_NAMES:
        config.add_variable(name, "float")

    def data_received(_timestamp, data, _config):
        samples.put((time.monotonic(), data))

    def log_error(_config, message):
        samples.put((time.monotonic(), RuntimeError(f"Telemetry: {message}")))

    config.data_received_cb.add_callback(data_received)
    config.error_cb.add_callback(log_error)
    cf.log.add_config(config)
    try:
        config.start()
        yield samples
    finally:
        try:
            if cf.is_connected():
                config.delete()
        finally:
            config.data_received_cb.remove_callback(data_received)
            config.error_cb.remove_callback(log_error)


def wait_for_position(cf, samples, timeout=PREFLIGHT_TIMEOUT):
    deadline = time.monotonic() + timeout
    history = deque(maxlen=10)
    while time.monotonic() < deadline:
        if not cf.is_connected():
            raise ConnectionError("Disconnected during preflight")
        try:
            received_at, data = samples.get(timeout=min(0.2, max(0, deadline - time.monotonic())))
        except Empty:
            continue
        if isinstance(data, Exception):
            raise data
        if time.monotonic() - received_at > 1.0:
            history.clear()
            continue
        values = tuple(float(data[name]) for name in VARIANCE_NAMES + POSITION_NAMES)
        if not all(math.isfinite(value) for value in values) or min(values[:3]) < 0:
            raise RuntimeError("Invalid estimator telemetry; no arming requested")
        if history and received_at - history[-1][0] > 1.0:
            history.clear()
        history.append((received_at, values))
        if len(history) == 10:
            columns = list(zip(*(sample[1] for sample in history)))
            # Bitcraze's variance convergence criterion, plus stationary position.
            variance_stable = all(max(c) - min(c) < 0.001 for c in columns[:3])
            position_stable = all(max(c) - min(c) < 0.02 for c in columns[3:])
            if variance_stable and position_stable:
                return tuple(sum(c) / len(c) for c in columns[3:])
    raise TimeoutError("No stable, fresh position within 30 seconds; no arming requested")


def prepare_flight(scf, initial_z=None):
    """Configure the Flow deck flight without waiting for position telemetry."""
    cf = scf.cf
    wait_for_parameters(scf)
    flow_detected = False
    for name in ("deck.bcFlow2", "deck.bcFlow"):
        if cf.param.toc.get_element_by_complete_name(name) is not None:
            flow_detected |= int(cf.param.get_value(name)) == 1
    if not flow_detected:
        raise RuntimeError("No Flow deck detected; no arming requested")
    set_parameter(cf, "stabilizer.estimator", 2)  # Kalman
    set_parameter(cf, "stabilizer.controller", 1)  # PID
    set_parameter(cf, "commander.enHighLevel", 0)
    if initial_z is not None:
        set_parameter(cf, "kalman.initialZ", initial_z)
    set_parameter(cf, "kalman.resetEstimation", 1)
    connected_sleep(cf, 0.1)
    set_parameter(cf, "kalman.resetEstimation", 0)


def preflight(scf):
    prepare_flight(scf)
    cf = scf.cf
    print("Waiting for stable position; keep the Crazyflie still on the floor.")
    with position_samples(cf) as samples:
        return wait_for_position(cf, samples)


def stream_hover(cf, duration, start_z, end_z, vx=0.0, on_height=None):
    """Stream body-frame velocity and estimator-frame Z at 20 Hz.

    Like MotionCommander's forward(), distance is speed times duration, not
    an independently measured endpoint. No background sender survives cleanup.
    """
    started = time.monotonic()
    while True:
        if not cf.is_connected():
            raise ConnectionError("Radio link lost; scripted landing is unavailable")
        elapsed = time.monotonic() - started
        fraction = min(elapsed / duration, 1.0)
        z = start_z + (end_z - start_z) * fraction
        if on_height is not None:
            on_height(z)
        # The final packet stops horizontal travel before transitioning phases.
        cf.commander.send_hover_setpoint(vx if fraction < 1 else 0.0, 0.0, 0.0, z)
        if fraction >= 1:
            return
        time.sleep(min(SETPOINT_PERIOD, duration - elapsed))


def fly(cf, ground_z, height, duration, distance=0.25, speed=0.2):
    commander = cf.commander
    arm_requested = False
    takeoff_requested = False
    commanded_z = ground_z

    def record_height(z):
        nonlocal commanded_z
        commanded_z = z

    try:
        arm_requested = True
        cf.supervisor.send_arming_request(True)
        connected_sleep(cf, 1.0)
        # Modern firmware exposes arming state; older versions use cflib's fallback.
        if cf.platform.get_protocol_version() >= 12 and not cf.supervisor.is_armed:
            raise RuntimeError("Firmware did not confirm arming")
        print(f"Taking off: {height:.2f} m above launch position.")
        takeoff_requested = True  # Also clean up if sending takeoff raises.
        target_z = ground_z + height
        stream_hover(cf, RAMP_SECONDS, ground_z, target_z, on_height=record_height)
        print(f"Hovering for {duration:g} seconds.")
        stream_hover(cf, duration, target_z, target_z)
        print(f"Flying forward {distance:g} m at {speed:g} m/s.")
        stream_hover(cf, distance / speed, target_z, target_z, vx=speed)
        stream_hover(cf, 0.5, target_z, target_z)  # Brake before descending.
    finally:
        if arm_requested:
            if cf.is_connected():
                try:
                    if takeoff_requested:
                        print("Landing.")
                        stream_hover(cf, RAMP_SECONDS, commanded_z, ground_z)
                        stream_hover(cf, 0.5, ground_z, ground_z)
                finally:
                    if cf.is_connected():
                        try:
                            commander.send_stop_setpoint()
                        finally:
                            cf.supervisor.send_arming_request(False)
                            time.sleep(0.1)  # Allow final packets to leave before disconnecting.
            else:
                print("Link lost: cannot command landing or motor stop.", file=sys.stderr)


def run(args):
    cflib.crtp.init_drivers()
    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./.cache")) as scf:
        x, y, z = preflight(scf)
        print(f"Stable launch position: x={x:.3f}, y={y:.3f}, z={z:.3f} m")
        print("Flow deck and estimator checks passed; use a well-lit, textured floor.")
        if args.check_only:
            print("Check complete; no arming or flight commands sent.")
            return
        fly(scf.cf, z, args.height, args.duration, args.distance, args.speed)
        print("Sequence complete; stop and disarm commands sent.")


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted; flight cleanup attempted if needed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Hover failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
