"""Stationary hover and shared supervised flight execution."""
from contextlib import ExitStack
from collections import deque
import math
import multiprocessing
from pathlib import Path
from queue import Empty, Full
import time

import numpy as np

from .. import poly7
from ..clock import RealClock
from ..frames import Pose
from ..preflight import run_preflight, samples, set_parameter
from . import artifacts
from .lighthouse import check_live_geometry
from .rig import corners

POSITION_NAMES = ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z')
HEALTH_NAMES = ('pm.vbat', 'lighthouse.bsActive', 'stateEstimate.roll', 'stateEstimate.pitch', 'stateEstimate.yaw')


def battery_policy(demo=False):
    # This CF21B reports pm.lowVoltage=3.35 and criticalLowVoltage=3.0.
    # Keep a 0.1 V margin over critical and debounce ordinary load transients.
    return dict(low_v=3.35, critical_v=3.1, low_duration_s=1.) if demo else dict(
        low_v=3.5, critical_v=3.5, low_duration_s=0.)


class BatteryMonitor:
    def __init__(self, policy):
        self.policy = policy
        self.low_since = self.previous_at = None

    def update(self, voltage, received_at):
        if not math.isfinite(voltage) or voltage <= 0:
            raise RuntimeError('Invalid battery telemetry')
        if voltage < self.policy['critical_v']:
            raise RuntimeError(f'Battery {voltage:.2f} V below {self.policy["critical_v"]:g} V critical threshold')
        if self.previous_at is not None and received_at-self.previous_at > .5:
            self.low_since = None
        self.previous_at = received_at
        if voltage >= self.policy['low_v']:
            self.low_since = None
        else:
            if self.low_since is None:
                self.low_since = received_at
            if received_at-self.low_since >= self.policy['low_duration_s']:
                raise RuntimeError(f'Battery below {self.policy["low_v"]:g} V for '
                                   f'{self.policy["low_duration_s"]:g} seconds; ending hover check')


def hover_height(rig):
    return max(max(t['pose']['pos'][2] for t in rig['tags']) + .20 - rig['envelope_body_m']['min'][2],
               rig['home']['pos'][2]+.05)


def route(rig, demo=False, visits=None, hover_s=10.):
    """Hover over launch by default; optional visits reuse the same trajectory limits."""
    if not math.isfinite(hover_s) or not 0 < hover_s <= 60:
        raise ValueError('Hover duration must be greater than zero and at most 60 seconds')
    lo, hi = (np.asarray(rig['envelope_body_m'][k]) for k in ('min', 'max'))
    fmin, fmax = (np.asarray(rig['geofence'][k]) for k in ('min', 'max'))
    height = hover_height(rig)
    home = np.asarray(rig['home']['pos'], float)
    reference_offset = np.asarray(rig['body_to_sample_m'])
    ez = rig['emergency_land_body_z_m']
    if ez >= height or ez + lo[2] - .02 < fmin[2] or ez + hi[2] + .02 > fmax[2]:
        raise ValueError('Emergency landing height must be below hover and fit the workspace envelope')
    targets = [dict(name='takeoff', target_body_m=[*home[:2], height],
                    dwell_s=hover_s if visits is None else 2.)]
    if visits is not None:
        targets.extend(visits)
        targets.append(dict(name='return', target_body_m=[*home[:2], height], dwell_s=0.))
    targets.append(dict(name='land', target_body_m=home.tolist(), dwell_s=0.))
    previous, segments = home, []
    for visit in targets:
        name, target, dwell = visit['name'], np.asarray(visit['target_body_m'], float), visit['dwell_s']
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError(f'{name}: invalid waypoint')
        # Zero-yaw route with an extra attitude margin; no interior obstacles are modelled.
        if any(np.any(p + lo - .02 < fmin) or np.any(p + hi + .02 > fmax) for p in (previous, target)):
            raise ValueError(f'{name}: drone/gripper envelope exceeds clear workspace (including 2 cm margin)')
        # Firmware seventh-order rest-to-rest profile. Increase duration until analytic sampled limits pass.
        duration = max(3., float(np.linalg.norm(target - previous)) / (.09 if demo else .04))
        while True:
            piece = poly7.plan_7th_order_no_jerk(0., duration, previous, 0., np.zeros(3), 0., np.zeros(3),
                                                 target, 0., np.zeros(3), 0., np.zeros(3))
            speed, accel = poly7.limits(piece)
            if speed <= (.2 if demo else .1) and accel <= (.3 if demo else .2):
                break
            duration *= 1.1
        segments.append(dict(visit, target_m=(target + reference_offset).tolist(),
                             target_body_m=target.tolist(), duration_s=duration,
                             peak_speed_mps=speed, peak_accel_mps2=accel))
        previous = target
    if sum(s['duration_s'] + s['dwell_s'] for s in segments) > 180:
        raise ValueError('Hover route exceeds the 180 s experiment budget; shorten the layout')
    return dict(version=1, kind='hover-check' if visits is None else 'tag-visit',
                segments=segments, clearance_m=.2, envelope_margin_m=.02, demo=demo,
                tracking_tolerance_m=.10 if demo else .05,
                duration_s=sum(s['duration_s'] + s['dwell_s'] for s in segments))


class Health:
    def __init__(self, cf, clock, position_queue, health_queue, rig, emit, demo=False):
        self.cf, self.clock, self.pq, self.hq = cf, clock, position_queue, health_queue
        self.rig, self.emit = rig, emit
        self.position = self.health = None
        self.position_at = self.health_at = -math.inf
        self.mask = sum(1 << k for k in rig['base_stations'])
        self.demo = demo
        self.last_station_at = -math.inf
        self.battery_monitor = BatteryMonitor(battery_policy(demo))

    def check(self, require_battery=True):
        if not self.cf.is_connected():
            raise ConnectionError('Radio link lost')
        for queue, label in ((self.pq, 'position'), (self.hq, 'health')):
            while True:
                try:
                    received, data = queue.get_nowait()
                except Empty:
                    break
                if isinstance(data, Exception):
                    raise data
                if not all(math.isfinite(float(v)) for v in data.values()):
                    raise RuntimeError('Nonfinite flight telemetry')
                setattr(self, label, data)
                setattr(self, label + '_at', received)
                self.emit(dict(event='telemetry', host_time=received, values=data))
                if label == 'health' and require_battery and self.clock.time()-received <= .5:
                    self.battery_monitor.update(float(data['pm.vbat']), received)
        if self.clock.time() - min(self.position_at, self.health_at) > .5:
            raise RuntimeError('Flight telemetry stale for more than 0.5 s')
        active = int(self.health['lighthouse.bsActive']) & self.mask
        if active:
            self.last_station_at = self.health_at
        if not self.demo and active != self.mask:
            raise RuntimeError('Both Lighthouse stations must remain active')
        if self.demo and self.clock.time()-self.last_station_at > .5:
            raise RuntimeError('No Lighthouse stations have provided tracking for 0.5 seconds')
        angle_limit = 20 if self.demo else 10
        if max(abs(self.health[k]) for k in ('stateEstimate.roll', 'stateEstimate.pitch')) > angle_limit:
            raise RuntimeError(f'Attitude exceeds {angle_limit} degrees')
        if abs((self.health['stateEstimate.yaw'] + 180) % 360 - 180) > angle_limit:
            raise RuntimeError(f'Yaw differs from zero heading by more than {angle_limit} degrees')
        position = np.array([self.position[k] for k in POSITION_NAMES])
        from scipy.spatial.transform import Rotation
        attitude = Rotation.from_euler('xyz', [self.health['stateEstimate.roll'], -self.health['stateEstimate.pitch'],
                                                self.health['stateEstimate.yaw']], degrees=True)
        body = position - attitude.apply(self.rig['body_to_sample_m'])
        low, high = (np.asarray(self.rig['geofence'][k]) for k in ('min', 'max'))
        # Rotate all eight measured envelope corners, including the gripper.
        elo, ehi = (np.asarray(self.rig['envelope_body_m'][k]) for k in ('min', 'max'))
        rotated = attitude.apply([[x, y, z] for x in (elo[0], ehi[0])
                                  for y in (elo[1], ehi[1]) for z in (elo[2], ehi[2])])
        if np.any(body + np.min(rotated, axis=0) - .02 < low) or np.any(body + np.max(rotated, axis=0) + .02 > high):
            raise RuntimeError('Drone/gripper envelope left the configured workspace')
        return position


def publish_latest(queue, value):
    try:
        queue.put_nowait(value)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(value)
        except Full:
            pass


def camera_worker(args, rig, calibration, queue, stop, out):
    """Camera IO/vision lives in a process so a blocked camera cannot block flight cleanup."""
    from .camera import cv, open_camera, project_points, intrinsics_compatibility
    from .vision import detect
    try:
        camera = open_camera(args)
        try:
            if camera.profile != calibration['profile'] or args.camera_mode != calibration['camera_mode']:
                raise ValueError('Camera profile/mode differs from accepted calibration')
            saved_intrinsics = artifacts.read(Path(args.calibration)/'intrinsics.json')
            if artifacts.fingerprint(saved_intrinsics) != calibration['intrinsics_fingerprint']:
                raise ValueError('Saved intrinsics file changed after calibration')
            compatibility = intrinsics_compatibility(saved_intrinsics, camera.intrinsics)
            if getattr(args, 'demo', False):
                compatibility['geometry_uses'] = 'live_intrinsics_with_refreshed_camera_pose'
            artifacts.write(Path(out)/'intrinsics-check.json', dict(version=1, **compatibility,
                saved=saved_intrinsics, live=camera.intrinsics,
                using_live_intrinsics_for_refresh=getattr(args, 'demo', False)))
            if not compatibility['compatible'] and not getattr(args, 'demo', False):
                raise ValueError(f'Camera lens mapping changed by {compatibility["max_pixel_drift"]:.3f} px '
                                 f'(limit {compatibility["allowed_pixel_drift"]:.2f} px); recalibrate. '
                                 f'See {Path(out)/"intrinsics-check.json"}')
            # Strict mode retains the fitted optical model. Demo mode refits pose
            # from the fixed map using current intrinsics, including thermal drift.
            if not getattr(args, 'demo', False):
                camera.intrinsics = saved_intrinsics
            c = cv()
            registration = None
            if getattr(args, 'demo', False):
                from .registration import refresh_pose
                captures = []
                deadline = time.monotonic()+15
                while len(captures) < 10 and time.monotonic() < deadline:
                    frame = camera.frame()
                    found = detect(frame.color)
                    known = {t['id'] for t in rig['tags']} & set(found)
                    if len(known) < 3:
                        captures.clear()
                        continue
                    if captures:
                        common = set(captures[0]) & known
                        if len(common) < 3 or max(np.linalg.norm(found[k]-captures[0][k], axis=1).max() for k in common) > 2.:
                            captures.clear()
                    captures.append({k: found[k] for k in known})
                if len(captures) < 10:
                    raise ValueError('Could not collect a stable view of three fixed tags for camera refresh')
                ids = set.intersection(*(set(v) for v in captures))
                observations = {k: np.median([v[k] for v in captures], axis=0) for k in ids}
                registration = refresh_pose(rig, observations, camera.intrinsics, calibration['T_lighthouse_camera'])
                artifacts.write(Path(out)/'camera-registration.json', registration)
                calibration = dict(calibration, T_lighthouse_camera=registration['T_lighthouse_camera'])
            inv = Pose.from_dict(calibration['T_lighthouse_camera']).inverse()
            detector = None
            if calibration.get('detector_model') and not getattr(args, 'camera_check_only', False):
                from .drone import NanoDetector, observe
                detector = NanoDetector(calibration['detector_model'])
            from scipy.spatial.transform import Rotation
            from ..frames import wxyz_to_xyzw
            rv = Rotation.from_quat(wxyz_to_xyzw(inv.quat)).as_rotvec()
            k, d = np.array(camera.intrinsics['matrix']), np.array(camera.intrinsics['distortion'])
            saved, last_saved = 0, -math.inf
            while not stop.is_set():
                frame = camera.frame()
                found = detect(frame.color)
                errors = []
                for tag in rig['tags']:
                    if tag['id'] in found:
                        projected = project_points(corners(tag), rv, np.array(inv.pos), camera.intrinsics)
                        errors.extend(np.linalg.norm(projected - found[tag['id']], axis=1).tolist())
                status = dict(time=frame.timestamp, visible_ids=sorted(found),
                              registration=registration,
                              alignment_rms_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else None)
                if len(errors) < 8 or status['alignment_rms_px'] > 2.:
                    status['error'] = (f'Camera/tag projection mismatch ({status["alignment_rms_px"]} px); '
                                       'use --demo to refresh the camera pose against fixed tags')
                if detector is not None:
                    try:
                        # Demo flight uses Lighthouse position; depth on propellers/background
                        # and a learned box centre are not reliable precision measurements.
                        if getattr(args, 'demo', False):
                            from .camera import Frame
                            visual_frame = Frame(frame.color, frame.timestamp, metadata=frame.metadata)
                        else:
                            visual_frame = frame
                        observation = observe(visual_frame, camera.intrinsics, detector)
                    except (ValueError, RuntimeError) as e:
                        if not getattr(args, 'demo', False):
                            raise
                        status['vision_warning'] = str(e)
                        observation = None
                    if observation is not None:
                        status['drone'] = observation
                    # RGB supplies a bearing; depth supplies an estimated 3D position.
                    # This is a visual body centre, not a measured Lighthouse sensor centre.
                    if observation is not None and observation['point_camera_m'] is not None:
                        position = inv.inverse().apply(observation['point_camera_m'])
                        status['drone_world_m'] = position.tolist()
                        status['tag_offsets_m'] = {str(t['id']): (position-np.array(t['pose']['pos'])).tolist()
                                                   for t in rig['tags']}
                    if observation is not None:
                        box = np.rint(observation['bbox_xyxy']).astype(int)
                        c.rectangle(frame.color, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
                if frame.timestamp - last_saved > 1:
                    filename = f'hover-{saved:04d}.png'
                    if not c.imwrite(str(Path(out) / filename), frame.color):
                        raise OSError('Could not save hover camera image')
                    status['image'] = filename
                    saved, last_saved = saved + 1, frame.timestamp
                # Nonblocking latest-value channel: never stall camera or flight on consumer speed.
                publish_latest(queue, status)
                stop.wait(.1)
        finally:
            camera.close()
    except BaseException as e:
        publish_latest(queue, dict(time=time.monotonic(), error=str(e)))


class CameraMonitor:
    def __init__(self, args, rig, calibration, out, emit):
        ctx = multiprocessing.get_context('spawn')
        self.queue, self.stop = ctx.Queue(maxsize=4), ctx.Event()
        self.process = ctx.Process(target=camera_worker,
                                   args=(args, rig, calibration, self.queue, self.stop, str(out)), daemon=True)
        self.latest, self.emit = None, emit
        self.calibration = calibration
        self.demo = getattr(args, 'demo', False)
        self.in_flight = False
        self.last_warning = None
        self.max_age_s = 1.
        self.position_history = deque()

    def __enter__(self):
        self.process.start()
        try:
            self.latest = self.queue.get(timeout=30)
            self.emit(dict(event='camera', **self.latest))
            self.check()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def check(self, expected_position=None):
        now = time.monotonic()
        if expected_position is not None:
            self.position_history.append((now, np.asarray(expected_position, float)))
            while self.position_history and now-self.position_history[0][0] > 3:
                self.position_history.popleft()
        while True:
            try:
                self.latest = self.queue.get_nowait()
                self.emit(dict(event='camera', **self.latest))
            except Empty:
                break
        issue = None
        if self.latest is None or time.monotonic() - self.latest['time'] > self.max_age_s:
            issue = f'Camera observations stale for more than {self.max_age_s:g} seconds'
        elif 'error' in self.latest:
            issue = self.latest['error']
        if issue:
            if self.demo and self.in_flight:
                if issue != self.last_warning:
                    self.emit(dict(event='vision_warning', reason=issue, flight_control='lighthouse'))
                    self.last_warning = issue
                return
            raise RuntimeError(issue)
        if expected_position is not None and 'drone' in self.latest:
            # Inference completes after capture. Comparing an older image to
            # the current pose would create a speed-dependent tracking error.
            times = np.array([v[0] for v in self.position_history])
            positions = np.array([v[1] for v in self.position_history])
            expected_position = np.array([np.interp(self.latest['time'], times, positions[:, axis]) for axis in range(3)])
            if 'drone_world_m' in self.latest:
                error = float(np.linalg.norm(np.array(self.latest['drone_world_m'])-expected_position))
                self.emit(dict(event='visual_tracking_error', error_m=error, metric='body_centre_3d'))
                if error > .05 and not self.demo:
                    raise RuntimeError('Visual drone position differs from Lighthouse by more than 5 cm')
            else:
                transform = (self.latest.get('registration') or self.calibration)['T_lighthouse_camera']
                expected = Pose.from_dict(transform).inverse().apply(expected_position)
                ray = np.array(self.latest['drone']['ray_camera'])
                error = float(np.linalg.norm(np.cross(expected, ray))/np.linalg.norm(ray))
                self.emit(dict(event='visual_tracking_error', error_m=error, metric='bearing_lateral'))
                if (expected[2] <= 0 or error > .05) and not self.demo:
                    raise RuntimeError('Visual drone bearing differs from Lighthouse by more than 5 cm laterally')

    def __exit__(self, *exc):
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.queue.close()


def execute(cf, plan, rig, health, camera, clock, emit):
    """Called only after explicit --arm and preflight. Injectable for hardware-free tests."""
    airborne = False
    try:
        health.check()
        camera.check()
        cf.supervisor.send_arming_request(True)
        clock.sleep(1.)
        if cf.platform.get_protocol_version() >= 12 and not cf.supervisor.is_armed:
            raise RuntimeError('Firmware did not confirm arming')
        camera.in_flight = True
        for segment in plan['segments']:
            health.check()
            camera.check()
            target = segment['target_m']
            emit(dict(event='target', **segment))
            if segment['name'] == 'takeoff':
                airborne = True  # cleanup also covers partial command transmission
                cf.high_level_commander.takeoff(target[2], segment['duration_s'])
            elif segment['name'] == 'land':
                cf.high_level_commander.land(target[2], segment['duration_s'])
            else:
                cf.high_level_commander.go_to(*target, 0., segment['duration_s'])
            end = clock.time() + segment['duration_s'] + .5 + segment['dwell_s']
            settled_at = clock.time() + segment['duration_s'] + .5
            while clock.time() < end:
                clock.sleep(.05)
                position = health.check()
                camera.check(position)
                if clock.time() >= settled_at:
                    error = float(np.linalg.norm(position - np.array(target)))
                    emit(dict(event='tracking_error', target=segment['name'], error_m=error))
                    tolerance = plan.get('tracking_tolerance_m', .05)
                    if error > tolerance:
                        raise RuntimeError(f'Settled tracking error exceeds {tolerance*100:g} cm')
            if segment['name'] == 'land':
                airborne = False
    finally:
        if cf.is_connected():
            try:
                if airborne:
                    emit(dict(event='emergency_landing'))
                    cf.high_level_commander.land(rig['emergency_land_body_z_m'] + rig['body_to_sample_m'][2], 4.)
                    clock.sleep(4.5)
            finally:
                try:
                    cf.high_level_commander.stop()
                finally:
                    cf.supervisor.send_arming_request(False)


def run(args, rig, ask=input):
    allow_unverified = getattr(args, 'allow_unverified', False)
    calibration = artifacts.accepted(args.calibration, 'camera', rig, allow_unverified=allow_unverified)
    lh = artifacts.accepted(args.lighthouse, 'lighthouse', rig, allow_unverified=allow_unverified)
    unverified = not (calibration.get('accepted') is True and lh.get('accepted') is True)
    if calibration['lighthouse_fingerprint'] != lh['geometry_fingerprint']:
        raise ValueError('Camera calibration belongs to a different Lighthouse geometry')
    solution = artifacts.read(Path(args.lighthouse) / 'solution.json')
    if unverified and not (solution.get('fit_valid') is True and solution.get('verification_skipped') is True):
        raise ValueError('Unverified override requires a successfully fitted, explicitly skipped Lighthouse solution')
    if artifacts.fingerprint(solution['geometry']) != lh['geometry_fingerprint']:
        raise ValueError('Saved Lighthouse geometry has changed')
    demo = getattr(args, 'demo', False)
    camera_only = getattr(args, 'camera_check_only', False)
    command = getattr(args, 'command', 'hover-check')
    controller = getattr(args, 'controller', None) or ('pid' if demo else 'mellinger')
    live = args.arm or args.check_only or camera_only
    output = artifacts.session(args.out, command, rig)
    artifacts.report(output, unverified_calibration=unverified, allow_unverified=allow_unverified, demo=demo,
                     visual_position_advisory=demo,
                     controller=controller,
                     flight_battery_policy=battery_policy(demo),
                     camera_reprojection_rms_px=calibration.get('reprojection_rms_px'),
                     camera_reprojection_limit_px=calibration.get('reprojection_limit_px', 1.),
                     camera_observation_max_age_s=1.)
    def make_plan(pose_calibration, intrinsics, source):
        if command == 'tag-visit':
            from .visits import route as tag_route, preview
            plan = tag_route(rig, pose_calibration, intrinsics,
                             offset_m=getattr(args, 'tag_offset_m', .15), demo=demo)
            plan['camera_pose_source'] = source
            preview(plan, output/'route-preview.png')
            summary = ', '.join(f'{s["tag_id"]}: {s["inward_offset_m"]*100:.0f} cm'
                                for s in plan['segments'] if 'tag_id' in s)
            print(f'Tag visits toward the centre (tag: offset): {summary}.', flush=True)
        else:
            plan = route(rig, demo=demo, hover_s=getattr(args, 'hover_seconds', 10.))
        plan['controller'] = controller
        artifacts.write(output / 'route.json', plan)
        return plan

    if not live:
        make_plan(calibration, artifacts.read(Path(args.calibration)/'intrinsics.json'), 'saved_calibration')
        artifacts.report(output, dry_run=True, accepted=True)
        print(f'DRY RUN: no devices opened. Route: {output / "route.json"}')
        return output
    if not args.uri and not camera_only:
        raise ValueError('--uri is required for --arm or --check-only')
    from ..hardware import connect
    from cflib.crazyflie.log import LogConfig
    clock = RealClock()
    events = (output / 'events.jsonl').open('w', buffering=1)
    import json
    def emit(value):
        events.write(json.dumps(dict(t=clock.time(), **value), allow_nan=False) + '\n')
    try:
        with ExitStack() as stack:
            stack.callback(events.close)
            camera = stack.enter_context(CameraMonitor(args, rig, calibration, output, emit))
            if demo and camera.latest.get('registration'):
                registration = camera.latest['registration']
                print(f'Camera registered from {len(registration["tags"])} fixed tags: '
                      f'{registration["before_rms_px"]:.2f} -> {registration["reprojection_rms_px"]:.2f} px RMS.')
            refreshed = demo and bool(camera.latest.get('registration'))
            pose_calibration = camera.latest['registration'] if refreshed else calibration
            intrinsics = (artifacts.read(output/'intrinsics-check.json')['live'] if refreshed
                          else artifacts.read(Path(args.calibration)/'intrinsics.json'))
            plan = make_plan(pose_calibration, intrinsics, 'live_registration' if refreshed else 'saved_calibration')
            if camera_only:
                artifacts.report(output, accepted=True, camera_check_only=True,
                                 camera_status=camera.latest, radio_connected=False, flight_complete=False)
                print('Camera check passed; no radio connection or arming.')
                return output
            scf = stack.enter_context(connect(args.uri))
            # Check provenance on hardware, not merely accepted files on disk.
            from ..preflight import wait_for_parameters
            wait_for_parameters(scf, clock)
            factory = artifacts.read(Path(args.lighthouse) / 'factory-calibration.json')['stations']
            if unverified:
                if scf.cf.platform.get_protocol_version() >= 12 and scf.cf.supervisor.is_armed:
                    raise RuntimeError('Geometry upload requires the drone to be disarmed')
                from .lighthouse import upload
                emit(dict(event='upload_unverified_geometry', geometry_fingerprint=lh['geometry_fingerprint']))
                upload_report = upload(scf.cf, solution, factory)
                artifacts.report(output, unverified_geometry_uploaded=True, geometry_readback_verified=True,
                                 lighthouse_upload=upload_report)
            else:
                check_live_geometry(scf.cf, solution['geometry'], factory)
            reference_home = dict(rig['home'], pos=(np.array(rig['home']['pos']) + rig['body_to_sample_m']).tolist())
            preflight_plan = dict(home=reference_home, settings=dict(geofence=rig['geofence'],
                params={'lighthouse.method': 1} if demo else {}))
            report = run_preflight(scf, preflight_plan, LogConfig, clock, home_tolerance=.10 if demo else .03,
                                   controller=1 if controller == 'pid' else 2)
            if controller == 'mellinger':
                # Restore firmware defaults; use the rig's estimated loaded mass.
                from cflib.crazyflie.param import ParamTocElement
                controller_params = scf.cf.param.toc.toc.get('ctrlMel', {})
                if 'mass' not in controller_params:
                    raise RuntimeError('Firmware does not expose the required Mellinger mass parameter')
                for field, element in controller_params.items():
                    if field != 'mass' and element.access == ParamTocElement.RW_ACCESS:
                        name = 'ctrlMel.' + field
                        set_parameter(scf.cf, name, wait_callback_default(scf.cf, name))
                set_parameter(scf.cf, 'ctrlMel.mass', rig['mass_kg'])
            pq = stack.enter_context(samples(scf.cf, LogConfig, clock, POSITION_NAMES, period_ms=50,
                                             log_name='Cal position'))
            hq = stack.enter_context(samples(scf.cf, LogConfig, clock, HEALTH_NAMES, period_ms=100,
                                             log_name='Cal health'))
            health = Health(scf.cf, clock, pq, hq, rig, emit, demo=demo)
            clock.sleep(.3)
            position = health.check()
            camera.check(position)
            artifacts.report(output, preflight=report)
            if args.check_only:
                artifacts.report(output, accepted=True, check_only=True)
                print('Preflight passed; motors remained disarmed.')
                return output
            print(f'Starting {command} with {controller.upper()}; planned duration '
                  f'{plan["duration_s"]:.0f} s plus settling. Keep the flight area clear.', flush=True)
            execute(scf.cf, plan, rig, health, camera, clock, emit)
        if command == 'tag-visit':
            visibility = visit_visibility(output/'events.jsonl', plan)
            artifacts.write(output/'tag-visibility.json', visibility)
            question = f'Did the drone hover beside all {len(rig["tags"])} tags with the tags visible?'
        else:
            question = 'Did the drone hold steadily over the launch point and land?'
        artifacts.report(output, flight_complete=True, operator_observation='pending')
        passed = ask(f'Flight complete. {question} [yes/no]: ').strip().lower()
        artifacts.report(output, accepted=passed == 'yes', operator_observation=passed,
                         flight_complete=True, independent_drone_pose_measured=False)
    except BaseException as e:
        artifacts.report(output, accepted=False, reason=str(e))
        raise
    finally:
        events.close()
    return output


def visit_visibility(events_path, plan):
    """Count actual tag detections during each settled dwell, using capture times."""
    import json
    events = [json.loads(line) for line in Path(events_path).read_text().splitlines()]
    result = {}
    known = {s['tag_id'] for s in plan['segments'] if 'tag_id' in s}
    for event in events:
        if event.get('event') != 'target' or 'tag_id' not in event:
            continue
        start = event['t']+event['duration_s']+.5
        end = start+event['dwell_s']
        frames = {e['time']: e for e in events if e.get('event') == 'camera'
                  and start <= e['time'] <= end and 'visible_ids' in e}
        result[str(event['tag_id'])] = dict(frames=len(frames),
            target_visible_frames=sum(event['tag_id'] in e['visible_ids'] for e in frames.values()),
            all_tags_visible_frames=sum(known.issubset(e['visible_ids']) for e in frames.values()))
    return dict(version=1, visits=result, measurement='AprilTag detections during settled dwells')


def wait_callback_default(cf, name):
    from .lighthouse import wait_callback
    # Firmware supplies parameter defaults; cflib invokes callback(name, default_value).
    value = wait_callback(lambda cb: cf.param.get_default_value(name, lambda _name, value: cb(value)), timeout=5.)
    if value is None:
        raise RuntimeError(f'Firmware could not provide default value for {name}')
    return value
