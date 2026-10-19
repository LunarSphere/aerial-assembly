"""Synthetic and mocked-hardware calibration checks. Never open cameras or radios."""
import copy
import math
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flight.calibration import artifacts, camera, cli, hover, lighthouse, rig, vision, layout
from flight.clock import ManualClock
from flight.frames import Pose, xyzw_to_wxyz
from flight import fake, preflight


def test_rgbd_launch_return_uses_metric_distance():
    from flight.calibration.setup import check_launch_return
    origin = dict(center_px=[858., 435.], point_camera_m=[.52, .16, 1.58])
    home = dict(center_px=[860., 429.], point_camera_m=[.526, .147, 1.583])
    assert check_launch_return(origin, home)['accepted']
    home['point_camera_m'] = [.52, .16, 1.61]
    with pytest.raises(ValueError, match='within 2 cm'):
        check_launch_return(origin, home)


def test_camera_refresh_recovers_pose_without_changing_tag_map(survey, view):
    from flight.calibration.registration import refresh_pose
    expected, intrinsics, observations = view
    prior = Pose((.47, .33, 1.5), tuple(xyzw_to_wxyz(
        Rotation.from_euler('xyz', [174, 3, 13], degrees=True).as_quat())))
    original = copy.deepcopy(survey)
    result = refresh_pose(survey, observations, intrinsics, prior.to_dict())
    recovered = Pose.from_dict(result['T_lighthouse_camera'])
    np.testing.assert_allclose(recovered.pos, expected.pos, atol=1e-6)
    assert recovered.angle_to(expected) < 1e-4
    assert result['max_heldout_rms_px'] < 1e-4
    assert survey == original
    assert result['independent_verification'] is False


def test_camera_refresh_rejects_moved_tag_and_missing_map(survey, view):
    from flight.calibration.registration import refresh_pose
    pose, intrinsics, observations = view
    with pytest.raises(ValueError, match='three fixed'):
        refresh_pose(survey, {k: observations[k] for k in (0, 1)}, intrinsics, pose.to_dict())
    moved = copy.deepcopy(observations)
    moved[4] += [30, -20]
    with pytest.raises(ValueError, match='inconsistent'):
        refresh_pose(survey, moved, intrinsics, pose.to_dict())


@pytest.fixture
def survey():
    positions = [[0, 0, -.06], [1, 0, -.06], [.2, .7, -.06],
                 [.7, .2, -.06], [.8, .7, -.06], [-.1, .5, -.06]]
    return rig.validate(dict(version=1, family='25h9', sample_offset_m=[0, 0, .06],
        body_to_sample_m=[0, 0, .012], home=dict(pos=[-.2, -.2, -.012], yaw=0),
        emergency_land_body_z_m=-.02, mass_kg=.055,
        envelope_body_m=dict(min=[-.085, -.085, -.045], max=[.085, .085, .035]),
        geofence=dict(min=[-.5, -.5, -.3], max=[1.3, 1., .8]), base_stations=[0, 1],
        tags=[dict(id=i, role=rig.ROLES[i], size_m=.08, pose=dict(pos=p, quat=[1, 0, 0, 0]))
              for i, p in enumerate(positions)]))


@pytest.fixture
def view(survey):
    # Slightly tilted overhead camera, looking down (+camera Z into the work surface).
    rotation = Rotation.from_euler('xyz', [174, 3, 4], degrees=True)
    transform = Pose((.45, .35, 1.5), tuple(xyzw_to_wxyz(rotation.as_quat())))
    inverse = transform.inverse()
    intrinsics = dict(version=1, profile=dict(backend='opencv', device='0', width=1280, height=720),
                      matrix=[[850, 0, 640], [0, 850, 360], [0, 0, 1]], distortion=[.02, -.01, .001, 0, 0])
    c = camera.cv()
    observations = {}
    rv = Rotation.from_matrix(rotation.as_matrix().T).as_rotvec()
    for tag in survey['tags']:
        observations[tag['id']] = c.projectPoints(rig.corners(tag), rv, np.array(inverse.pos),
            np.array(intrinsics['matrix'], float), np.array(intrinsics['distortion'], float))[0].reshape(4, 2)
    return transform, intrinsics, observations


def test_camera_pose_and_independent_check_tags(survey, view):
    expected, intrinsics, observations = view
    result = vision.solve(survey, observations, intrinsics)
    assert result['accepted']
    got = Pose.from_dict(result['T_lighthouse_camera'])
    np.testing.assert_allclose(got.pos, expected.pos, atol=1e-6)
    assert got.angle_to(expected) < 1e-4
    assert result['max_check_error_m'] < 1e-6


def test_heldout_tag_survey_error_is_not_fitted_away(survey, view):
    _, intrinsics, observations = view
    survey['tags'][4]['pose']['pos'][0] += .02
    result = vision.solve(survey, observations, intrinsics)
    assert not result['accepted']
    assert result['max_check_error_m'] > .019
    assert result['reprojection_rms_px'] < 1e-5


def test_camera_missing_tag_and_intrinsics_rejected(survey, view):
    _, intrinsics, observations = view
    del observations[5]
    with pytest.raises(ValueError, match='Missing'):
        vision.solve(survey, observations, intrinsics)
    with pytest.raises(ValueError, match='differs'):
        camera.validate_intrinsics(intrinsics, dict(intrinsics['profile'], width=640))
    with pytest.raises(ValueError, match='Invalid'):
        camera.validate_intrinsics(dict(intrinsics, matrix=np.zeros((3, 3)).tolist()))


def test_rgbd_rejects_invalid_depth(survey, view):
    expected, intrinsics, observations = view
    frame = camera.Frame(np.zeros((720, 1280, 3), np.uint8), 0., np.full((720, 1280), np.nan))
    report = vision.depth_report(frame, observations, survey, intrinsics, expected.to_dict())
    assert len(report) == 6
    assert not any(item['valid'] for item in report)


def test_tag_corner_order_and_dictionary_are_detected(survey):
    c = camera.cv()
    dictionary = c.aruco.getPredefinedDictionary(c.aruco.DICT_APRILTAG_25h9)
    canvas = np.full((500, 500), 255, np.uint8)
    canvas[100:400, 100:400] = c.aruco.generateImageMarker(dictionary, 2, 300)
    found = vision.detect(canvas)
    assert list(found) == [2]
    np.testing.assert_allclose(found[2], [[100, 100], [399, 100], [399, 399], [100, 399]], atol=1)


def test_rig_validation(survey):
    survey['tags'][1]['pose']['pos'][0] = .999
    with pytest.raises(ValueError, match='exactly'):
        rig.validate(survey)


def test_rig_units_and_missing_offset(survey):
    survey['mass_kg'] = 55
    with pytest.raises(ValueError, match='kg'):
        rig.validate(survey)


def health_data(clock, position, mask=3, vbat=4.):
    q1, q2 = Queue(), Queue()
    q1.put((clock.time(), dict(zip(hover.POSITION_NAMES, position))))
    q2.put((clock.time(), {'pm.vbat': vbat, 'lighthouse.bsActive': mask,
                          'stateEstimate.roll': 0., 'stateEstimate.pitch': 0., 'stateEstimate.yaw': 0.}))
    return q1, q2


def test_health_stale_stations_and_battery(survey):
    clock = ManualClock()
    cf = SimpleNamespace(is_connected=lambda: True)
    q1, q2 = health_data(clock, [0, 0, .2])
    monitor = hover.Health(cf, clock, q1, q2, survey, lambda _: None)
    monitor.check()
    clock.sleep(.51)
    with pytest.raises(RuntimeError, match='stale'):
        monitor.check()
    for mask, vbat, text in [(1, 4, 'stations'), (3, 3.4, 'Battery')]:
        q1, q2 = health_data(clock, [0, 0, .2], mask, vbat)
        with pytest.raises(RuntimeError, match=text):
            hover.Health(cf, clock, q1, q2, survey, lambda _: None).check()


def test_demo_health_single_station_and_total_loss_timeout(survey):
    clock = ManualClock()
    cf = SimpleNamespace(is_connected=lambda: True)
    q1, q2 = health_data(clock, [0, 0, .2], mask=1)
    monitor = hover.Health(cf, clock, q1, q2, survey, lambda _: None, demo=True)
    monitor.check()
    clock.sleep(.2)
    monitor.pq, monitor.hq = health_data(clock, [0, 0, .2], mask=0)
    monitor.check()
    clock.sleep(.31)
    monitor.pq, monitor.hq = health_data(clock, [0, 0, .2], mask=0)
    with pytest.raises(RuntimeError, match='No Lighthouse stations'):
        monitor.check()


def test_demo_battery_sag_recovers_but_sustained_low_voltage_lands():
    monitor = hover.BatteryMonitor(hover.battery_policy(True))
    for t, voltage in [(0., 4.1), (.1, 3.48), (.2, 3.3), (.5, 3.3), (.6, 3.6),
                       (.8, 3.3), (1.1, 3.3), (1.4, 3.3), (1.7, 3.3)]:
        monitor.update(voltage, t)
    with pytest.raises(RuntimeError, match='for 1 seconds'):
        monitor.update(3.3, 1.9)


def test_demo_critical_battery_voltage_is_immediate():
    monitor = hover.BatteryMonitor(hover.battery_policy(True))
    monitor.update(4.1, 0.)
    with pytest.raises(RuntimeError, match='critical'):
        monitor.update(3.09, .1)


def test_battery_debounce_does_not_bridge_telemetry_gap():
    monitor = hover.BatteryMonitor(hover.battery_policy(True))
    monitor.update(3.3, 0.)
    monitor.update(3.3, 2.)
    assert monitor.low_since == 2.


def test_demo_camera_fault_is_advisory_only_after_arming():
    import time
    monitor = object.__new__(hover.CameraMonitor)
    monitor.latest = dict(time=time.monotonic(), error='Camera unavailable')
    monitor.queue = Queue()
    monitor.max_age_s = 1.
    monitor.demo = True
    monitor.in_flight = False
    monitor.last_warning = None
    events = []
    monitor.emit = events.append
    with pytest.raises(RuntimeError, match='Camera unavailable'):
        monitor.check()
    monitor.in_flight = True
    monitor.check()
    monitor.check()
    assert len(events) == 1
    assert events[0]['event'] == 'vision_warning'


def synthetic_lighthouse_samples():
    from cflib.localization.lighthouse_types import LhDeck4SensorPositions
    from cflib.localization.lighthouse_bs_vector import LighthouseBsVector
    # BS +X points into the volume. Its local +Z need not be world +Z.
    origins = [np.array([-1.5, -1., 2.]), np.array([2., 1.5, 2.2])]
    target = np.array([.4, .3, .2])
    rotations = []
    for origin in origins:
        x = target - origin
        x /= np.linalg.norm(x)
        y = np.cross([0, 0, 1.], x)
        y /= np.linalg.norm(y)
        rotations.append(np.column_stack([x, y, np.cross(x, y)]))
    points = [('origin', [0, 0, 0]), ('x', [1, 0, 0]), ('xy', [.2, .7, 0])]
    rng = np.random.default_rng(40)
    points += [('xyz', rng.uniform([-.1, -.1, .1], [1., .7, .8])) for _ in range(8)]
    points += [('verification', p) for p in [[.1, .1, .1], [.8, .1, .3], [.1, .6, .5], [.8, .6, .6], [.5, .3, .25]]]
    raw = dict(version=1, base_stations=[0, 1], samples=[])
    for role, point in points:
        attitude = Rotation.from_euler('xyz', rng.uniform(-.25, .25, 3)) if role == 'xyz' else Rotation.identity()
        sensors = attitude.apply(LhDeck4SensorPositions.positions) + point
        angles = {}
        for i, (origin, r) in enumerate(zip(origins, rotations)):
            angles[str(i)] = [list(LighthouseBsVector.from_cart(r.T @ (v - origin)).lh_v1_angle_pair) for v in sensors]
        raw['samples'].append(dict(role=role, angles=angles))
    return raw


def test_real_cflib_solver_with_synthetic_sweeps(tmp_path):
    artifacts.write(tmp_path / 'samples.json', synthetic_lighthouse_samples())
    solution = lighthouse.solve_bounded(tmp_path / 'samples.json', tmp_path)
    assert solution['accepted'], solution
    assert solution['max_verification_error_m'] < 1e-6
    assert set(solution['geometry']) == {'0', '1'}


@pytest.mark.parametrize('error,passes', [(.000399, True), (.0004, True), (.000401, False)])
def test_lighthouse_error_boundary_in_complete_solve(error, passes):
    from cflib.localization.lighthouse_geo_estimation_manager import LhGeoEstimationManager
    from cflib.localization.lighthouse_cf_pose_sample import LhCfPoseSampleType
    original = LhGeoEstimationManager.estimate_geometry
    def controlled(data):
        solution = original(data)
        for sample in solution.samples:
            if sample.sample_type == LhCfPoseSampleType.VERIFICATION:
                sample.error_distance = error
        return solution
    with patch.object(LhGeoEstimationManager, 'estimate_geometry', side_effect=controlled):
        assert lighthouse.solve(synthetic_lighthouse_samples())['accepted'] == passes


def test_missing_station_and_verification():
    raw = synthetic_lighthouse_samples()
    del raw['samples'][0]['angles']['1']
    with pytest.raises(ValueError, match='two configured'):
        lighthouse.solve(raw)
    raw = synthetic_lighthouse_samples()
    raw['samples'] = [s for s in raw['samples'] if s['role'] != 'verification']
    with pytest.raises(ValueError, match='verification'):
        lighthouse.solve(raw)


def test_solver_timeout_preserves_diagnostic(tmp_path):
    import subprocess
    with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('solver', .01)):
        with pytest.raises(TimeoutError):
            lighthouse.solve_bounded(tmp_path / 'samples.json', tmp_path)
    assert (tmp_path / 'solver.log').exists()


def test_artifact_rejection_and_mismatch(tmp_path, survey):
    path = artifacts.session(tmp_path / 'session', 'lighthouse', survey)
    with pytest.raises(ValueError, match='accepted'):
        artifacts.accepted(path, 'lighthouse', survey)
    artifacts.report(path, accepted=True)
    survey['mass_kg'] += .001
    with pytest.raises(ValueError, match='Survey changed'):
        artifacts.accepted(path, 'lighthouse', survey)
    with pytest.raises(ValueError):
        artifacts.fingerprint({'bad': math.nan})


def test_hover_interrupt_lands_stops_and_disarms(survey):
    clock = ManualClock()
    cf = fake.RecordingCrazyflie(clock=clock, home=[-.2, -.2, 0.])
    cf.platform = SimpleNamespace(get_protocol_version=lambda: 11)
    monitor = SimpleNamespace(check=MagicMock(side_effect=[np.array([0, 0, .2]), np.array([0, 0, .2]), KeyboardInterrupt()]))
    with pytest.raises(KeyboardInterrupt):
        hover.execute(cf, hover.route(survey), survey, monitor, SimpleNamespace(check=lambda: None), clock, lambda _: None)
    names = [c[0] for c in cf.rec.calls]
    assert 'hl.land' in names
    assert names[-2:] == ['hl.stop', 'supervisor.send_arming_request']
    assert cf.rec.calls[-1][1] is False


def accepted_sessions(tmp_path, survey):
    lh = artifacts.session(tmp_path / 'lh', 'lighthouse', survey)
    geometry = {'0': dict(origin=[0, 0, 2], rotation=np.eye(3).tolist())}
    artifacts.write(lh / 'solution.json', dict(version=1, geometry=geometry))
    key = artifacts.fingerprint(geometry)
    artifacts.report(lh, accepted=True, upload_verified=True, geometry_fingerprint=key)
    ca = artifacts.session(tmp_path / 'ca', 'camera', survey)
    artifacts.report(ca, accepted=True, lighthouse_fingerprint=key)
    return lh, ca


def test_shared_preflight_rejects_stale_and_nonfinite_battery():
    clock, q = ManualClock(10.), Queue()
    for _ in range(5):
        q.put((0., {'pm.vbat': 4.}))
    with pytest.raises(TimeoutError, match='fresh'):
        preflight.battery(None, q, clock, timeout=.2)
    q.put((clock.time(), {'pm.vbat': math.nan}))
    with pytest.raises(preflight.PreflightError, match='Invalid'):
        preflight.battery(None, q, clock)


def test_no_arm_and_check_only_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.parser().parse_args(['hover-check', '--rig', 'x', '--lighthouse', 'y', '--calibration', 'z',
                                 '--out', 'o', '--arm', '--check-only'])


@pytest.mark.parametrize('demo', [False, True])
def test_complete_hover_against_kinematic_crazyflie(survey, demo):
    clock = ManualClock()
    home_reference = np.array(survey['home']['pos']) + survey['body_to_sample_m']
    cf = fake.RecordingCrazyflie(clock=clock, home=home_reference)
    cf.platform = SimpleNamespace(get_protocol_version=lambda: 12)
    monitor = SimpleNamespace(check=lambda: cf.current())
    events = []
    plan = hover.route(survey, demo=demo)
    hover.execute(cf, plan, survey, monitor, SimpleNamespace(check=lambda *args: None), clock, events.append)
    assert [e['name'] for e in events if e['event'] == 'target'] == [s['name'] for s in plan['segments']]
    assert not cf.supervisor.is_armed
    assert max(e['error_m'] for e in events if e['event'] == 'tracking_error') < 1e-6


def test_camera_fault_during_hover_triggers_landing(survey):
    clock = ManualClock()
    cf = fake.RecordingCrazyflie(clock=clock, home=[0, 0, 0])
    cf.platform = SimpleNamespace(get_protocol_version=lambda: 12)
    camera_monitor = SimpleNamespace(check=MagicMock(side_effect=[None, None, RuntimeError('camera stalled')]))
    with pytest.raises(RuntimeError, match='stalled'):
        hover.execute(cf, hover.route(survey), survey, SimpleNamespace(check=cf.current), camera_monitor,
                      clock, lambda _: None)
    assert any(c[0] == 'hl.land' for c in cf.rec.calls)
    assert not cf.supervisor.is_armed


def test_camera_queue_keeps_recent_observations():
    q = Queue(maxsize=2)
    for i in range(20):
        hover.publish_latest(q, i)
    assert [q.get_nowait(), q.get_nowait()] == [18, 19]


@pytest.mark.parametrize('capacity', [2, 4, 16])
@pytest.mark.parametrize('write_success', [True, False])
def test_upload_respects_firmware_capacity_and_checks_real_write_result(capacity, write_success):
    # Exercise cflib's real writer, including its padding of unused slots.
    from cflib.utils.callbacks import Caller
    pending, writes, persists = [], [], []
    caller = Caller()
    cf = SimpleNamespace(loc=SimpleNamespace(receivedLocationPacket=caller, LH_PERSIST_DATA=11))
    def persist(geos, calibs):
        persists.append((geos, calibs))
        pending.append(lambda: caller.call(SimpleNamespace(type=11)))
    cf.loc.send_lh_persist_data_packet = persist
    class Helper:
        def __init__(self, cf):
            pass
        def read_all_geos(self, cb):
            cb(dict.fromkeys(range(capacity)))
        read_all_calibs = read_all_geos
        def write_geos(self, values, cb):
            writes.append(values)
            pending.append(lambda: cb(write_success and set(values).issubset(range(capacity))))
    def complete(start, timeout=30.):
        results = []
        start(results.append)
        while pending:
            pending.pop(0)()
        assert len(results) == 1
        return results[0]
    solution = {'geometry': {str(k): dict(origin=[k, 0, 2], rotation=np.eye(3).tolist()) for k in (0, 1)}}
    with (patch('cflib.crazyflie.mem.LighthouseMemHelper', Helper),
          patch('cflib.localization.lighthouse_config_manager.LighthouseMemHelper', Helper),
          patch.object(lighthouse, 'wait_callback', complete),
          patch.object(lighthouse, 'check_live_geometry') as verify):
        if write_success:
            result = lighthouse.upload(cf, solution, {})
            assert result['supported_station_slots'] == capacity
            verify.assert_called_once_with(cf, solution['geometry'], {})
        else:
            with pytest.raises(RuntimeError, match='persistence failed'):
                lighthouse.upload(cf, solution, {})
            verify.assert_not_called()
    assert len(writes) == 1 and set(writes[0]) == set(range(capacity))
    assert all(writes[0][k].valid == (k < 2) for k in range(capacity))
    assert persists == [(list(range(capacity)), [])]  # factory calibration untouched
    assert caller.callbacks == []


@pytest.mark.parametrize('slots,calib_slots,requested', [([], [], 0), ([0, 2], [0, 2], 0),
    ([0, 1], [0], 0), ([0, 1], [0, 1], 4)])
def test_upload_rejects_uncertain_capacity_and_unsupported_channel(slots, calib_slots, requested):
    helper = MagicMock()
    helper.read_all_geos.side_effect = lambda cb: cb(dict.fromkeys(slots))
    helper.read_all_calibs.side_effect = lambda cb: cb(dict.fromkeys(calib_slots))
    with (patch('cflib.crazyflie.mem.LighthouseMemHelper', return_value=helper),
          patch('cflib.localization.lighthouse_config_manager.LighthouseConfigWriter') as writer):
        with pytest.raises((RuntimeError, ValueError), match='capacity'):
            lighthouse.upload(object(), {'geometry': {str(requested): {}}}, {})
        writer.assert_not_called()


def test_rejected_lighthouse_run_preserves_samples_and_never_uploads(tmp_path, survey):
    from cflib.crazyflie.mem import LighthouseBsCalibration
    cal = LighthouseBsCalibration()
    cal.valid = True
    cal.uid = 1
    cf = MagicMock()
    cf.mem.get_mems.return_value = [MagicMock()]
    cf.param.get_value.side_effect = lambda name: {'deck.bcLighthouse4': '1', 'lighthouse.systemType': '2'}[name]
    scf = SimpleNamespace(cf=cf, is_params_updated=lambda: True)
    context = MagicMock()
    context.__enter__.return_value = scf
    solution = dict(version=1, accepted=False, max_verification_error_m=.001, coverage_ok=True)
    args = SimpleNamespace(out=str(tmp_path / 'failed'), uri='radio://fake')
    with (patch('flight.hardware.connect', return_value=context),
          patch.object(lighthouse, 'wait_factory_calibration'),
          patch.object(lighthouse, 'wait_callback', return_value={0: cal, 1: cal}),
          patch.object(lighthouse, 'collect', return_value={'angles': {}}),
          patch.object(lighthouse, 'solve_bounded', return_value=solution),
          patch.object(lighthouse, 'upload') as upload):
        with pytest.raises(ValueError, match='rejected'):
            lighthouse.calibrate(args, survey, ask=lambda _: '')
        upload.assert_not_called()
    assert len(artifacts.read(tmp_path / 'failed' / 'samples.json')['samples']) == 16
    assert not artifacts.read(tmp_path / 'failed' / 'report.json')['accepted']
    assert all(call.args == (False,) for call in cf.supervisor.send_arming_request.call_args_list)


def test_callback_timeout_and_no_success_callback():
    with pytest.raises(TimeoutError):
        lighthouse.wait_callback(lambda cb: None, timeout=.001)


def test_nonconverged_and_repeated_verification_rejected():
    from cflib.localization.lighthouse_geo_estimation_manager import LhGeoEstimationManager
    original = LhGeoEstimationManager.estimate_geometry
    def fail_convergence(data):
        result = original(data)
        result.has_converged = False
        return result
    with patch.object(LhGeoEstimationManager, 'estimate_geometry', side_effect=fail_convergence):
        assert not lighthouse.solve(synthetic_lighthouse_samples())['accepted']
    raw = synthetic_lighthouse_samples()
    records = [s for s in raw['samples'] if s['role'] != 'verification']
    repeated = next(s for s in raw['samples'] if s['role'] == 'verification')
    raw['samples'] = records + [copy.deepcopy(repeated) for _ in range(5)]
    result = lighthouse.solve(raw)
    assert not result['coverage_ok']
    assert not result['accepted']


def test_live_geometry_mismatch_rejected():
    cf = MagicMock()
    cf.mem.get_mems.return_value = [MagicMock()]
    expected = {'0': dict(origin=[0, 0, 2], rotation=np.eye(3).tolist())}
    actual = lighthouse.geometry_objects({'0': dict(origin=[.1, 0, 2], rotation=np.eye(3).tolist())})
    with patch.object(lighthouse, 'wait_callback', return_value=actual):
        with pytest.raises(ValueError, match='differs'):
            lighthouse.check_live_geometry(cf, expected, {})


def test_camera_calibration_session_preserves_capture_and_provenance(tmp_path, survey, view):
    _, intrinsics, observations = view
    lh, _ = accepted_sessions(tmp_path, survey)
    source = SimpleNamespace(profile=intrinsics['profile'], intrinsics=intrinsics, close=MagicMock(),
        frame=lambda: camera.Frame(np.zeros((720, 1280, 3), np.uint8), __import__('time').monotonic()))
    args = SimpleNamespace(lighthouse=str(lh), out=str(tmp_path / 'camera-capture'))
    with (patch.object(vision, 'open_camera', return_value=source),
          patch.object(vision, 'detect', return_value=observations),
          patch.object(vision, 'depth_report', return_value=[dict(valid=True)])):
        output = vision.calibrate(args, survey)
    report = artifacts.read(output / 'report.json')
    assert report['accepted']
    assert report['lighthouse_fingerprint'] == artifacts.read(lh / 'report.json')['geometry_fingerprint']
    assert len(list(output.glob('color-*.png'))) == 20
    assert len(artifacts.read(output / 'observations.json')['frames']) == 20
    source.close.assert_called_once()


def test_guided_survey_does_not_prompt_when_output_already_exists(tmp_path):
    path = tmp_path / 'rig.json'
    path.write_text('keep this survey')
    with pytest.raises(ValueError, match='already exists'):
        rig.survey(path, ask=lambda _: pytest.fail('should not prompt before checking output'))
    assert path.read_text() == 'keep this survey'


def test_automatic_layout_recovers_metric_tag_positions_and_rotations(survey, view):
    transform, intrinsics, observations = view
    depth = np.full((720, 1280), np.nan)
    normal = Rotation.from_quat(np.roll(transform.quat, -1)).as_matrix()[2]
    distance = survey['tags'][0]['pose']['pos'][2] - transform.pos[2]
    for quad in observations.values():
        mask = np.zeros(depth.shape, np.uint8)
        camera.cv().fillConvexPoly(mask, np.rint(quad).astype(np.int32), 1)
        yy, xx = np.nonzero(mask)
        rays = camera.pixel_rays(np.column_stack([xx, yy]), intrinsics)
        depth[yy, xx] = distance/(rays @ normal)
    result = layout.estimate(observations, intrinsics, depth)
    assert result['reference_distance_m'] == pytest.approx(1., abs=1e-6)
    assert not result['accepted_for_flight']
    for tag in result['tags']:
        expected = np.array(survey['tags'][tag['id']]['pose']['pos'])
        expected[2] = 0
        np.testing.assert_allclose(tag['pose']['pos'], expected, atol=1e-6)
        assert tag['size_m'] == pytest.approx(.08, abs=1e-6)
        assert not tag['independently_verified']


def test_layout_estimates_cannot_certify_themselves(survey, view):
    _, intrinsics, observations = view
    for tag in survey['tags']:
        tag['independently_verified'] = False
        tag['survey_source'] = 'rgbd_estimate'
    result = vision.solve(survey, observations, intrinsics)
    assert result['max_check_error_m'] < 1e-6
    assert not result['independent_tag_verification']
    assert not result['accepted']


def test_inverse_brown_projection_round_trip_and_pose(survey, view):
    transform, intrinsics, _ = view
    intrinsics = dict(intrinsics, distortion_model='distortion.inverse_brown_conrady')
    inverse = transform.inverse()
    rv = Rotation.from_quat(np.roll(inverse.quat, -1)).as_rotvec()
    observations = {t['id']: camera.project_points(rig.corners(t), rv, np.array(inverse.pos), intrinsics)
                    for t in survey['tags']}
    result = vision.solve(survey, observations, intrinsics)
    assert result['accepted']
    np.testing.assert_allclose(result['T_lighthouse_camera']['pos'], transform.pos, atol=1e-6)
    pixels = np.array([[100., 100.], [640., 360.], [1100., 600.]])
    rays = camera.pixel_rays(pixels, intrinsics)
    recovered = camera.project_points(rays, np.zeros(3), np.zeros(3), intrinsics)
    np.testing.assert_allclose(recovered, pixels, atol=1e-6)


def test_invalid_layout_inputs(survey, view):
    _, intrinsics, observations = view
    with pytest.raises(ValueError, match='aligned'):
        layout.estimate(observations, intrinsics, np.ones((10, 10)))
    del observations[0]
    with pytest.raises(ValueError, match='Need tags'):
        layout.estimate(observations, intrinsics, np.ones((720, 1280)))


def test_survey_import_prefills_five_tag_estimates(tmp_path, capsys):
    source = artifacts.read(Path(__file__).parents[1] / 'configs' / 'tag-layout-overhead.json')
    # No per-tag position, orientation or size answers: all are supplied by the camera layout.
    answers = iter(['60', '', '', '', '', '12', '', '45', '55',
                    '85', '85', '85', '85', '45', '35', '-50', '130', '-120', '50', '86', '1 2'])
    prompts = []
    def answer(prompt):
        prompts.append(prompt)
        return next(answers)
    result = rig.survey(tmp_path / 'rig.json', ask=answer,
                        layout_path=Path(__file__).parents[1] / 'configs' / 'tag-layout-overhead.json')
    assert len(result['tags']) == 5
    assert not any('rotation' in p or 'Tag 2: forward' in p for p in prompts)
    assert result['tags'][1]['pose']['pos'] == [1., 0., -.06]
    expected = np.array(source['tags'][2]['pose']['pos'][:2]) / source['reference_distance_m']
    np.testing.assert_allclose(result['tags'][2]['pose']['pos'][:2], expected)
    assert not any(t['independently_verified'] for t in result['tags'])
    assert 'estimated 0-to-1 distance' in capsys.readouterr().out


def test_session_replaces_previous_session_but_not_other_directories(tmp_path):
    out = tmp_path / 'run'
    artifacts.session(out, 'setup', {})
    (out / 'stale.png').write_text('old')
    artifacts.session(out, 'setup', {})
    assert not (out / 'stale.png').exists()
    other = tmp_path / 'precious'
    other.mkdir()
    (other / 'data.txt').write_text('keep')
    with pytest.raises(ValueError, match='not a previous'):
        artifacts.session(other, 'setup', {})
    assert (other / 'data.txt').read_text() == 'keep'
