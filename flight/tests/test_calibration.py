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
from flight.frames import Pose, wxyz_to_xyzw, xyzw_to_wxyz
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


def wide_lens(intrinsics):
    # The view fixture's lens frames the tags, not a drone hovering over the outer ones.
    return dict(intrinsics, matrix=[[450, 0, 640], [0, 450, 360], [0, 0, 1]])


def test_centre_route_hovers_deck_over_every_tag_centre(survey, view, tmp_path):
    from flight.calibration import visits
    transform, intrinsics, _ = view
    intrinsics = wide_lens(intrinsics)
    calibration = dict(T_lighthouse_camera=transform.to_dict())
    plan = visits.centre_route(survey, calibration, intrinsics, demo=True)
    tags = [s for s in plan['segments'] if 'tag_id' in s]
    assert sorted(s['tag_id'] for s in tags) == [t['id'] for t in survey['tags']]
    by_id = {t['id']: t for t in survey['tags']}
    for segment in tags:
        np.testing.assert_allclose(segment['target_m'][:2], by_id[segment['tag_id']]['pose']['pos'][:2], atol=1e-12)
        assert segment['target_body_m'][2] == pytest.approx(hover.hover_height(survey))
    home = np.array(survey['home']['pos'][:2])
    nearest = min(survey['tags'], key=lambda t: np.linalg.norm(np.array(t['pose']['pos'][:2])-home))
    assert tags[0]['tag_id'] == nearest['id']
    assert plan['kind'] == 'approach-accuracy' and plan['centre_source'] == 'rig'
    assert plan['duration_s'] <= 180
    surveyed = {0: [.01, -.02, -.06]}
    moved = visits.centre_route(survey, calibration, intrinsics, centres=surveyed, demo=True)
    segment = next(s for s in moved['segments'] if s.get('tag_id') == 0)
    np.testing.assert_allclose(segment['target_m'][:2], [.01, -.02])
    assert moved['centre_source'] == 'survey'
    visits.preview(plan, tmp_path/'preview.png')
    assert (tmp_path/'preview.png').stat().st_size > 0
    with pytest.raises(ValueError, match='dwell'):
        visits.centre_route(survey, calibration, intrinsics, dwell_s=0.)


def test_tag_survey_recovers_centres_and_flags_disagreement(survey, view):
    from flight.calibration import accuracy
    transform, intrinsics, observations = view
    result = accuracy.survey_tags(observations, transform.to_dict(), intrinsics, survey)
    for tag, rig_tag in zip(result['tags'], survey['tags']):
        np.testing.assert_allclose(tag['centre_m'], rig_tag['pose']['pos'], atol=1e-6)
    assert result['max_delta_xy_m'] < 1e-6 and result['warnings'] == []
    shifted = {**observations, 4: observations[4]+[20., 0.]}
    result = accuracy.survey_tags(shifted, transform.to_dict(), intrinsics, survey)
    assert [w.startswith('Tag 4') for w in result['warnings']] == [True]
    with pytest.raises(ValueError, match='missing \\[5\\]'):
        accuracy.survey_tags({k: v for k, v in observations.items() if k != 5}, transform.to_dict(), intrinsics, survey)


def camera_event(t, xy, height=.2, origin=(.45, .35, 1.5), depth_noise=0.):
    deck, origin = np.array([*xy, height]), np.array(origin)
    direction = (deck-origin)/np.linalg.norm(deck-origin)
    measured = deck+direction*depth_noise
    return dict(event='camera', t=t, time=t, alignment_rms_px=.4, drone_world_m=measured.tolist(),
                drone_ray_world_m=dict(origin=origin.tolist(), direction=direction.tolist()))


def test_deck_xy_uses_ray_at_median_depth_height():
    from flight.calibration import accuracy
    # Off-axis deck: depth noise along the ray moves the raw depth point sideways.
    frames = [camera_event(i, [1.0, .7], depth_noise=n) for i, n in enumerate([-.03, 0., 0., .005, .01])]
    result = accuracy.deck_xy(frames)
    np.testing.assert_allclose(result['xy_m'], [1.0, .7], atol=1e-12)
    assert result['deck_height_m'] == pytest.approx(.2)
    assert np.linalg.norm(np.subtract(result['depth_xy_m'], [1.0, .7])) > 1e-3
    assert result['frames'] == 5


def telemetry_event(t, xy):
    return dict(event='telemetry', t=t, host_time=t,
                values={'stateEstimate.x': xy[0], 'stateEstimate.y': xy[1], 'stateEstimate.z': .2})


def target_event(t, name, target, tag=None, duration=2., dwell=5.):
    event = dict(event='target', t=t, name=name, target_m=[*target, .2], duration_s=duration, dwell_s=dwell)
    return event if tag is None else dict(event, tag_id=tag)


def synthetic_trial_events():
    return [target_event(0., 'takeoff', [0, 0], dwell=2.),
            telemetry_event(4., [1., 1.]),
            target_event(5., 'tag-0', [0, 0], tag=0),          # settled window 7.5–12.5 s
            telemetry_event(8., [.01, 0.]), telemetry_event(11., [.01, .02]),
            camera_event(8., [.03, -.04]), camera_event(10., [.03, -.04]),
            camera_event(13.5, [.5, .5]),                       # transit: outside the window
            target_event(13., 'tag-1', [1, 0], tag=1),         # settled window 15.5–20.5 s
            telemetry_event(16., [1., 0.]),
            target_event(21., 'return', [0, 0], dwell=0.),
            target_event(25., 'land', [0, 0], dwell=0.)]


SYNTHETIC_CENTRES = {0: [0., 0., -.06], 1: [1., 0., -.06]}


def test_approach_errors_from_settled_dwells():
    from flight.calibration import accuracy
    rows = accuracy.approaches(synthetic_trial_events(), SYNTHETIC_CENTRES, trial=1)
    assert [r['tag_id'] for r in rows] == [0, 1]
    first, second = rows
    np.testing.assert_allclose(first['control_error_xy_m'], [.01, .01])
    np.testing.assert_allclose(first['positioning_error_xy_m'], [.03, -.04], atol=1e-12)
    assert first['positioning_error_m'] == pytest.approx(.05)
    assert (first['lighthouse_samples'], first['camera_frames']) == (2, 2)
    assert second['control_error_m'] == pytest.approx(0.)
    assert second['positioning_error_xy_m'] is None and second['camera_frames'] == 0
    # A flight that stops during a dwell contributes no approach for it.
    assert accuracy.approaches(synthetic_trial_events()[:6], SYNTHETIC_CENTRES, trial=1) == []


def test_summary_statistics_match_hand_calculation():
    from flight.calibration import accuracy
    rows = [dict(tag_id=0, control_error_xy_m=[.03, .04], positioning_error_xy_m=None),
            dict(tag_id=1, control_error_xy_m=[0., 0.], positioning_error_xy_m=[.01, 0.])]
    summary = accuracy.summarize(rows)
    control = summary['control']['overall']
    assert control['n'] == 2
    assert control['mean_error_m'] == pytest.approx(.025)
    assert control['rmse_m'] == pytest.approx(math.sqrt(.0025/2))
    assert control['std_error_m'] == pytest.approx(np.std([.05, 0.], ddof=1))
    np.testing.assert_allclose(control['bias_xy_m'], [.015, .02])
    assert summary['control']['per_tag']['0']['mean_error_m'] == pytest.approx(.05)
    assert summary['positioning']['missing'] == 1
    assert summary['positioning']['per_tag']['0'] == dict(n=0)
    assert summary['positioning']['overall']['max_error_m'] == pytest.approx(.01)


def synthetic_session(root):
    import json
    root.mkdir()
    artifacts.write(root/'session.json', dict(version=1, kind='approach-accuracy', dry_run=False, trials=[
        dict(index=1, dir='trial-01', accepted=True), dict(index=2, dir='trial-02', accepted=False)]))
    artifacts.write(root/'survey.json', dict(version=1, kind='tag-survey', registration=None,
        tags=[dict(id=k, centre_m=v) for k, v in SYNTHETIC_CENTRES.items()]))
    (root/'trial-01').mkdir()
    (root/'trial-01'/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in synthetic_trial_events()))
    (root/'trial-02').mkdir()  # failed trial: no events, excluded
    return root


def test_metrics_and_plot_regenerate_offline(tmp_path):
    pytest.importorskip('matplotlib')
    from flight.calibration import accuracy
    root = synthetic_session(tmp_path/'session')
    metrics = accuracy.write_metrics(root)
    assert metrics['trials_included'] == ['trial-01'] and metrics['trials_excluded'] == ['trial-02']
    assert artifacts.read(root/'metrics.json') == metrics
    assert len(metrics['approaches']) == 2
    assert (root/'approach-errors.png').stat().st_size > 0
    (root/'metrics.json').unlink()
    (root/'approach-errors.png').unlink()
    assert cli.main(['approach-metrics', str(root)]) == 0
    assert artifacts.read(root/'metrics.json') == metrics
    assert (root/'approach-errors.png').exists()


def test_exp3_figure_import_keeps_survey_image_and_plots_approach_errors(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    import importlib.util
    script = Path(__file__).resolve().parents[2]/'figures'/'exp3-approach-accuracy'/'make_figures.py'
    spec = importlib.util.spec_from_file_location('exp3_make_figures', script)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    monkeypatch.setattr(figures, 'DATA', tmp_path/'data')
    root = synthetic_session(tmp_path/'session')
    (root/'survey.png').write_bytes(b'png')
    (root/'approach-errors.png').write_bytes(b'derived')
    for trial in ('trial-01', 'trial-02'):
        artifacts.write(root/trial/'report.json', dict(version=1))
    (root/'trial-02'/'events.jsonl').write_text('')
    figures.import_session(root)
    assert (tmp_path/'data'/'survey.png').read_bytes() == b'png'
    assert not (tmp_path/'data'/'approach-errors.png').exists()  # derived; rebuilt from data/
    _, approaches, _ = figures.load()
    figures.plot_approach_errors(approaches, tmp_path/'approach-errors.png')
    assert (tmp_path/'approach-errors.png').stat().st_size > 0


def calibrated_files(tmp_path, survey, view):
    transform, intrinsics, _ = view
    lh, ca = accepted_sessions(tmp_path, survey)
    artifacts.report(ca, T_lighthouse_camera=transform.to_dict(), profile=intrinsics['profile'])
    artifacts.write(ca/'intrinsics.json', wide_lens(intrinsics))
    artifacts.write(tmp_path/'rig.json', survey)
    return ['--rig', str(tmp_path/'rig.json'), '--lighthouse', str(lh), '--calibration', str(ca)]


def test_approach_accuracy_dry_run_opens_no_devices_and_never_overwrites(tmp_path, survey, view):
    files = calibrated_files(tmp_path, survey, view)
    out = tmp_path/'exp3'
    with patch('flight.calibration.camera.open_camera', side_effect=AssertionError('camera opened')):
        assert cli.main(['approach-accuracy', *files, '--demo', '--out', str(out)]) == 0
    plan = artifacts.read(out/'preview'/'route.json')
    assert plan['mode'] == 'over_centre' and plan['centre_source'] == 'rig'
    assert (out/'preview'/'route-preview.png').exists()
    assert artifacts.read(out/'session.json')['dry_run'] is True
    (out/'keep.txt').write_text('keep')
    assert cli.main(['approach-accuracy', *files, '--demo', '--out', str(out)]) == 1
    assert (out/'keep.txt').read_text() == 'keep'
    assert cli.main(['approach-accuracy', *files, '--demo']) == 1  # needs --out or --resume


def test_resume_appends_trials_without_touching_earlier_runs(tmp_path, survey):
    from flight.calibration import accuracy
    root = tmp_path/'exp3'
    root.mkdir()
    settings = dict(demo=True, controller='pid', dwell_s=5.)
    artifacts.write(root/'session.json', dict(version=1, kind='approach-accuracy', dry_run=False,
        rig_fingerprint=artifacts.fingerprint(survey), settings=settings,
        trials=[dict(index=i, dir=f'trial-{i:02d}', accepted=True) for i in (1, 2)]))
    artifacts.write(root/'survey.json', dict(version=1, registration=dict(T_lighthouse_camera='fixed'),
        tags=[dict(id=t['id'], centre_m=t['pose']['pos']) for t in survey['tags']]))
    for i in (1, 2):
        (root/f'trial-{i:02d}').mkdir()
        (root/f'trial-{i:02d}'/'marker').write_text(str(i))
    flown = []

    def fake_run(args, rig, ask):
        flown.append(args)
        output = artifacts.session(args.out, args.command, rig)
        artifacts.report(output, accepted=True, flight_complete=True)
        return output

    args = SimpleNamespace(command='approach-accuracy', trials=3, arm=True, check_only=False, camera_check_only=False,
                           resume=str(root), out=None, demo=True, controller=None, dwell_s=5., save_metrics=False)
    with patch.object(hover, 'run', fake_run):
        assert accuracy.run_session(args, survey, ask=lambda _: '') == root
    assert [Path(a.out).name for a in flown] == ['trial-03']
    assert not hasattr(flown[0], 'fixed_registration')
    assert set(flown[0].tag_centres) == {t['id'] for t in survey['tags']}
    assert [(root/f'trial-{i:02d}'/'marker').read_text() for i in (1, 2)] == ['1', '2']
    assert [t['dir'] for t in artifacts.read(root/'session.json')['trials']] == ['trial-01', 'trial-02', 'trial-03']
    with pytest.raises(ValueError, match='settings'):
        accuracy.run_session(SimpleNamespace(**dict(vars(args), demo=False)), survey)


def test_recording_monitor_logs_visual_disagreement_without_aborting():
    import time
    monitor = object.__new__(hover.CameraMonitor)
    monitor.queue = Queue()
    monitor.max_age_s = 1.
    monitor.demo = False
    monitor.in_flight = True
    monitor.last_warning = None
    monitor.position_history = hover.deque()
    events = []
    monitor.emit = events.append
    monitor.latest = dict(time=time.monotonic(), drone={}, drone_world_m=[.1, 0., .2])
    monitor.record_only = True
    monitor.check(np.array([0., 0., .2]))
    assert events[-1]['event'] == 'visual_tracking_error' and events[-1]['error_m'] == pytest.approx(.1)
    monitor.latest = dict(time=time.monotonic(), error='Drone not confidently visible')
    monitor.check(np.array([0., 0., .2]))
    assert events[-1]['event'] == 'vision_warning'
    monitor.record_only = False
    monitor.latest = dict(time=time.monotonic(), drone={}, drone_world_m=[.1, 0., .2])
    with pytest.raises(RuntimeError, match='5 cm'):
        monitor.check(np.array([0., 0., .2]))


def test_setup_camera_rms_limit_defaults_to_three_pixels():
    from flight.calibration import setup
    assert setup.CAMERA_RMS_LIMIT_PX == 3.
    assert cli.parser().parse_args(['setup']).max_camera_rms_px is None
    assert cli.main(['setup', '--advanced', '--max-camera-rms-px', '2']) == 1


def test_session_target_defaults_to_five_and_extends_without_devices(tmp_path, survey, view):
    files = calibrated_files(tmp_path, survey, view)
    out = tmp_path/'exp3'
    assert cli.main(['approach-accuracy', *files, '--demo', '--out', str(out)]) == 0
    assert artifacts.read(out/'session.json')['target_trials'] == 5
    session = artifacts.read(out/'session.json')
    artifacts.write(out/'session.json', dict(session, dry_run=False))
    with patch('flight.calibration.camera.open_camera', side_effect=AssertionError('camera opened')), \
            patch('flight.hardware.connect', side_effect=AssertionError('radio opened')):
        assert cli.main(['approach-accuracy', '--resume', str(out), '--trials', '20']) == 0
    assert artifacts.read(out/'session.json')['target_trials'] == 20
    assert cli.main(['approach-accuracy', '--resume', str(out), '--trials', '0']) == 1


def still_sample(offset=0.):
    angles = {str(k): (np.full((4, 2), .1)+offset).tolist() for k in (0, 1)}
    packets = [{k: (np.asarray(v)+(offset*i if offset else 0.)).tolist() for k, v in angles.items()} for i in range(50)]
    return dict(angles=angles, packets=packets, timestamp=0.)


def test_space_samples_are_lighthouse_only_and_reject_motion(tmp_path):
    from flight.calibration import setup
    raw = dict(version=1, base_stations=[0, 1], samples=[dict(role=r) for r in ('origin', 'x', 'xy')], reference_only=False)
    prompts = []
    results = iter([TimeoutError('no pairs'), still_sample(), still_sample(.001), still_sample(), still_sample()])

    def fake_collect(*_args, **_kwargs):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result
    with patch.object(setup.lighthouse, 'collect', fake_collect):
        setup.collect_space_samples(None, [0, 1], tmp_path, raw, lambda *a: prompts.append(a), space=2, checks=1)
    assert [s['role'] for s in raw['samples'][3:]] == ['xyz', 'xyz', 'verification']
    assert not any('camera' in s for s in raw['samples'][3:])
    # Timeout and motion each repeat the same prompt.
    assert prompts == [('Space sample', 1, 2), ('Space sample', 1, 2), ('Space sample', 2, 2),
                       ('Space sample', 2, 2), ('Check', 1, 1)]
    assert artifacts.read(tmp_path/'samples.json')['samples'][-1]['role'] == 'verification'
    assert setup.held_still(still_sample(.001))[0] is False


def test_motion_trigger_needs_a_move_then_a_still_hold():
    from flight.calibration import setup
    detector = setup.StillAfterMove()
    # Held still from the start: no sample, however long.
    assert not any(detector.update(t/20, 2.) for t in range(100))
    # Hand tremor below the move threshold does not count as a move either.
    assert not any(detector.update(5+t/20, 30.) for t in range(20))
    detector.update(6., 120.)                      # carried to a new spot
    assert not detector.update(6.1, 3.)
    assert not detector.update(7., 25.)             # wobble restarts the hold
    assert not detector.update(7.1, 3.)
    assert detector.update(8.6, 3.)                 # 1.5 s still after the move
    assert not detector.update(10., 3.)             # next sample needs a new move


def test_motion_trigger_records_after_logged_move_and_hold():
    from contextlib import contextmanager
    from queue import Empty
    from flight.calibration import setup
    clock = ManualClock(100.)
    closed = []

    class ScriptedGyro:
        """Nothing queued before the prompt; then one move followed by a still hold."""
        def __init__(self):
            self.feed = [120.]+[2.]*40
        def get_nowait(self):
            raise Empty
        def get(self, timeout):
            if not self.feed:
                raise Empty
            clock.sleep(.05)
            return clock.time(), {'gyro.x': self.feed.pop(0), 'gyro.y': 0., 'gyro.z': 0.}

    gyro = ScriptedGyro()

    @contextmanager
    def fake_samples(*_args, **_kwargs):
        try:
            yield gyro
        finally:
            closed.append(True)
    cf = SimpleNamespace(is_connected=lambda: True)
    with patch('flight.preflight.samples', fake_samples), patch('builtins.print'):
        wait = setup.motion_trigger(cf, LogConfig=object, clock=clock)
        wait('Space sample', 1, 8)
    assert 0 < len(gyro.feed) < 40   # returned after about 1.5 s of stillness, not at the end of the feed
    wait.stop()
    assert closed == [True]


def test_camera_alignment_uses_floor_references_with_space_samples(view):
    from flight.calibration import setup
    transform, intrinsics, _ = view
    raw = synthetic_lighthouse_samples()
    solution = lighthouse.solve(raw)
    assert not solution['reference_only'] and solution['fit_valid']
    inverse = transform.inverse()
    rv = Rotation.from_quat(wxyz_to_xyzw(inverse.quat)).as_rotvec()
    for record, solved in zip(raw['samples'][:3], solution['samples'][:3]):
        world = np.asarray(solved['position_m'])
        record['camera'] = dict(point_camera_m=inverse.apply(world).tolist(),
                                center_px=camera.project_points(world[None], rv, np.array(inverse.pos), intrinsics)[0].tolist())
    alignment = setup.align_camera(raw, solution, intrinsics)
    assert alignment['fit_valid'] and alignment['reprojection_rms_px'] < 1e-6
    np.testing.assert_allclose(Pose.from_dict(alignment['T_lighthouse_camera']).pos, transform.pos, atol=1e-6)


def test_saturated_and_missing_depth_are_invalid():
    raw = np.array([[0, 1530, 65535]], np.uint16)
    depth = camera.depth_metres(raw, .001)
    assert np.isnan(depth[0, 0]) and np.isnan(depth[0, 2])
    assert depth[0, 1] == pytest.approx(1.53)


def test_lighthouse_solver_gets_more_than_cflib_default_evaluations():
    import scipy.optimize
    from cflib.localization import lighthouse_geometry_solver as geometry_solver
    budgets, original = [], scipy.optimize.least_squares
    def recording(*args, **kwargs):
        budgets.append(kwargs['max_nfev'])
        return original(*args, **kwargs)
    with patch.object(scipy.optimize, 'least_squares', recording):
        assert lighthouse.solve(synthetic_lighthouse_samples())['converged']
    assert lighthouse.SOLVER_EVALUATIONS in budgets and 100 not in budgets  # 10 is a separate cflib step
    assert geometry_solver.SolverData().max_nr_iter == 100  # cflib default restored


def preflight_monitor(statuses):
    import time
    monitor = object.__new__(hover.CameraMonitor)
    queue = Queue()
    now = time.monotonic()
    for status in statuses:
        queue.put(dict(status, time=now))
    monitor.process = SimpleNamespace(start=lambda: None, join=lambda timeout: None, is_alive=lambda: False)
    monitor.stop = SimpleNamespace(set=lambda: None)
    monitor.queue = SimpleNamespace(get=queue.get, get_nowait=queue.get_nowait, close=lambda: None)
    monitor.max_age_s, monitor.settle_s = 30., .5
    monitor.demo = monitor.record_only = monitor.in_flight = False
    monitor.last_warning, monitor.position_history = None, hover.deque()
    monitor.emit = lambda _: None
    return monitor


def test_preflight_camera_check_tolerates_one_disturbed_frame():
    mismatch = dict(error='Camera/tag projection mismatch (2.02 px)')
    monitor = preflight_monitor([mismatch, dict(alignment_rms_px=.8)])
    assert monitor.__enter__() is monitor
    with pytest.raises(RuntimeError, match='persisted'):
        preflight_monitor([mismatch]).__enter__()


def test_frame_refit_follows_a_shifted_camera_with_one_tag_covered(survey, view):
    transform, intrinsics, _ = view
    # The cage (and camera) shifted 2 cm and rotated 0.5 degree after the session started.
    moved = Pose(tuple(np.add(transform.pos, [.02, -.01, 0.])), tuple(xyzw_to_wxyz(
        (Rotation.from_quat(wxyz_to_xyzw(transform.quat))*Rotation.from_euler('z', .5, degrees=True)).as_quat())))
    inverse = moved.inverse()
    rv = Rotation.from_quat(wxyz_to_xyzw(inverse.quat)).as_rotvec()
    found = {t['id']: camera.project_points(rig.corners(t), rv, np.array(inverse.pos), intrinsics)
             for t in survey['tags'] if t['id'] != 3}   # drone hovering over tag 3
    pose, info = hover.refit_frame(survey, found, intrinsics, transform, transform)
    np.testing.assert_allclose(pose.pos, moved.pos, atol=1e-6)
    assert info['camera_shift_m'] == pytest.approx(math.hypot(.02, .01), abs=1e-6)
    assert 3 not in info['tags'] and info['rms_px'] < 1e-4
    kept, info = hover.refit_frame(survey, {k: found[k] for k in (0, 1)}, intrinsics, transform, transform)
    assert kept == transform and 'kept previous camera pose' in info['warning']


def test_launch_offset_compares_lighthouse_and_camera_pad_positions():
    from flight.calibration import accuracy
    events = [telemetry_event(-2., [.012, -.004]), telemetry_event(-1., [.014, -.006]),
              camera_event(-1.5, [0., 0.]), camera_event(-.5, [0., 0.]),
              target_event(0., 'takeoff', [0, 0], dwell=2.), camera_event(3., [.5, .5])]
    result = accuracy.launch_offset(events, trial=4)
    np.testing.assert_allclose(result['offset_xy_m'], [.013, -.005], atol=1e-12)
    assert result['trial'] == 4 and result['camera_frames'] == 2


def test_preflight_reports_low_battery_before_waiting_for_lighthouse():
    from contextlib import contextmanager
    cf = SimpleNamespace(param=SimpleNamespace(toc=SimpleNamespace(get_element_by_complete_name=lambda _: object()),
                                               get_value=lambda _: '1'))
    scf = SimpleNamespace(cf=cf, is_params_updated=lambda: True)

    @contextmanager
    def no_log(*_args, **_kwargs):
        yield None
    with patch.object(preflight, 'set_parameter'), patch.object(preflight, 'samples', no_log), \
            patch.object(preflight, 'battery', return_value=3.72), \
            patch.object(preflight, 'wait_for_position', side_effect=AssertionError('waited for Lighthouse')):
        with pytest.raises(preflight.PreflightError, match='3.72 V .* swap the battery'):
            preflight.run_preflight(scf, dict(settings={}), None, ManualClock())


def test_station_consistency_flags_a_moved_station():
    raw = synthetic_lighthouse_samples()
    geometry = lighthouse.solve(raw)['geometry']
    for record in raw['samples']:
        result = lighthouse.station_consistency(record['angles'], geometry)
        assert result['consistent'] and result['worst_rms_mrad'] < .05
    resting = raw['samples'][0]['angles']
    np.testing.assert_allclose(lighthouse.station_consistency(resting, geometry)['position_m'], [0, 0, 0], atol=1e-4)
    moved = copy.deepcopy(geometry)
    moved['1']['rotation'] = (Rotation.from_euler('z', .5, degrees=True).as_matrix() @ np.asarray(moved['1']['rotation'])).tolist()
    result = lighthouse.station_consistency(resting, moved)
    assert not result['consistent'] and result['worst_rms_mrad'] > 2


def test_flight_stops_before_preflight_when_stations_moved(tmp_path, survey):
    raw = synthetic_lighthouse_samples()
    geometry = lighthouse.solve(raw)['geometry']
    moved = copy.deepcopy(geometry)
    moved['0']['origin'] = (np.asarray(moved['0']['origin'])+[.05, 0, 0]).tolist()
    events = []
    sample = dict(angles=raw['samples'][0]['angles'], packets=[], timestamp=0.)
    with patch.object(lighthouse, 'collect', return_value=sample):
        assert hover.check_stations(None, survey, geometry, [0, 0, 0], tmp_path, events.append)['consistent']
        with pytest.raises(RuntimeError, match='Re-run setup'):
            hover.check_stations(None, survey, moved, [0, 0, 0], tmp_path, events.append)
    assert [e['event'] for e in events] == ['station_check', 'station_check']


def test_resume_on_a_new_calibration_keeps_each_trials_own_survey(tmp_path, survey):
    import json
    from flight.calibration import accuracy
    root = synthetic_session(tmp_path/'exp3')
    session = artifacts.read(root/'session.json')
    session.update(rig_fingerprint=artifacts.fingerprint(survey), settings=dict(demo=True, controller='pid', dwell_s=5.),
                   target_trials=2)
    artifacts.write(root/'session.json', session)
    artifacts.write(root/'rig.json', survey)
    recalibrated = copy.deepcopy(survey)
    recalibrated['tags'][0]['pose']['pos'][0] += .004   # same tags, new Lighthouse solution
    # The new calibration surveys tag 0 2 cm along +X.
    new_survey = dict(version=1, kind='tag-survey', warnings=[], registration=None,
                      tags=[dict(id=0, centre_m=[.02, 0., -.06]), dict(id=1, centre_m=[1., 0., -.06])])

    def fake_run(args, rig, ask):
        output = artifacts.session(args.out, args.command, rig)
        (output/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in synthetic_trial_events()))
        artifacts.report(output, accepted=True, flight_complete=True)
        return output
    args = SimpleNamespace(command='approach-accuracy', trials=None, arm=True, check_only=False, camera_check_only=False,
                           resume=str(root), out=None, demo=True, controller=None, dwell_s=5., save_metrics=False,
                           setup='runs/setup-new')
    with patch.object(accuracy, 'capture_survey', return_value=new_survey), patch.object(hover, 'run', fake_run), \
            patch('builtins.print'):
        accuracy.run_session(args, recalibrated, ask=lambda _: '')
    session = artifacts.read(root/'session.json')
    assert [c['index'] for c in session['calibrations']] == [1, 2]
    assert session['calibrations'][1]['from_trial'] == 3 and session['trials'][-1]['calibration'] == 2
    assert (root/'survey-02.json').exists() and (root/'rig-02.json').exists()
    metrics = accuracy.write_metrics(root)
    by_trial = {(r['trial'], r['tag_id']): r for r in metrics['approaches']}
    # Same flight data, so tag 0's positioning error differs by exactly the 2 cm survey change.
    assert by_trial[(3, 0)]['positioning_error_xy_m'][0] == pytest.approx(by_trial[(1, 0)]['positioning_error_xy_m'][0]-.02)
    assert set(metrics['summary_by_calibration']) == {'1', '2'}
    other = copy.deepcopy(survey)
    other['tags'] = other['tags'][:5]
    with pytest.raises(ValueError, match='different tags'):
        accuracy.run_session(args, other, ask=lambda _: '')
