"""Test flight (src/testflight): calibration, detection, pose source and field plan on synthetic frames.

The synthetic camera renders the CAD footprints/heights of the A/B brick, the platform and the
fixture at 2 m with D455-like noise; no RealSense, no radio.
"""
import math

import numpy as np
import pytest

pytest.importorskip('cv2')

from flight.checks import check_plan
from flight.frames import Pose
from testflight import calib as Ca, fieldplan as FP, synth as Sy, vision as V
from testflight.demo import SynthWorld, run_demo
from testflight.model import load_model
from testflight.rig import Rig
from testflight.sense import CameraPoseSource

R, T = Sy.look_at([0.05, -0.15, 2.0], [0., 0.05, 0.])
M = load_model()


@pytest.fixture(scope='module')
def calibration():
    scene = Sy.lab_scene()
    return Ca.calibrate(Sy.SynthCamera(scene, R, T), scene.survey(), 'DICT_APRILTAG_25h9', frames=3)


@pytest.fixture(scope='module')
def detector(calibration):
    return V.Detector(M, Rig(overrides={'log_dir': None}), Ca.CameraPose(calibration))


def source(calibration, objects, seed=0, **over):
    rig = Rig(overrides={'log_dir': None, **over})
    return CameraPoseSource(rig, SynthWorld(objects, R, T, seed=seed), cam_pose=Ca.CameraPose(calibration), model=M)


def test_model_matches_ab_cad():
    assert M.brick_size == pytest.approx([0.0497, 0.025], abs=1e-4)
    assert M.brick_origin_height == pytest.approx(0.0115)
    assert M.platform_centre_x == pytest.approx(0.025)
    assert np.nanmin(M.brick_top(np.array([0.]), np.array([0.]))) > 0.02


def test_calibration_recovers_the_camera(calibration):
    c = calibration
    assert all(ok for ok, _ in c['checks'].values()), c['checks']
    assert np.linalg.norm(np.asarray(c['camera_position_m']) - [0.05, -0.15, 2.0]) < 0.005
    assert abs(c['depth_scale'] - 1) < 0.005


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_brick_in_fixture_and_platform(detector, seed):
    rng = np.random.default_rng(seed)
    b = (rng.uniform(-0.4, 0.4), rng.uniform(-0.6, -0.1), rng.uniform(-60, 60))
    p = (rng.uniform(-0.4, 0.4), rng.uniform(0.15, 0.6), rng.uniform(-60, 60))
    objs = [Sy.Placed('fixture', *b), Sy.Placed('brick', *b), Sy.Placed('platform', *p)]
    res = detector.detect(Sy.SynthCamera(Sy.lab_scene(objs), R, T, seed=seed).capture())
    assert not res.ambiguous, res.issues
    assert np.linalg.norm(res.brick.pos[:2] - b[:2]) < 0.003
    assert abs(res.brick.pos[2] - M.brick_origin_height) < 0.002
    assert abs(math.degrees(V.wrap(res.brick.yaw - math.radians(b[2])))) < 2
    assert np.linalg.norm(res.platform.centre - p[:2]) < 0.003
    assert abs(math.degrees(V.wrap(res.platform.yaw - math.radians(p[2])))) < 2
    # base0's frame origin sits 25 mm along -x of the platform's footprint centre
    origin = np.asarray(p[:2]) - 0.025*np.array([math.cos(math.radians(p[2])), math.sin(math.radians(p[2]))])
    assert np.linalg.norm(res.platform.pos[:2] - origin) < 0.003
    assert res.fixture is not None and np.linalg.norm(res.fixture.centre - b[:2]) < 0.004


def test_half_turn_follows_the_prior(detector):
    objs = [Sy.Placed('brick', 0.1, -0.3, 170.), Sy.Placed('platform', -0.2, 0.3, 175.)]
    cam = Sy.SynthCamera(Sy.lab_scene(objs), R, T)
    res = detector.detect(cam.capture())
    assert abs(math.degrees(V.wrap(res.brick.yaw)) - 170) > 90        # prior 0 -> reported as -10 deg
    rig = Rig(overrides={'log_dir': None, 'yaw_prior_deg': {'brick': 180., 'platform': 180., 'fixture': 180.}})
    res = V.Detector(M, rig, detector.cam).detect(cam.capture())
    assert abs(math.degrees(V.wrap(res.brick.yaw - math.radians(170)))) < 4     # a lone brick: no holder yaw
    assert abs(math.degrees(V.wrap(res.platform.yaw - math.radians(175)))) < 2


def test_brick_seated_on_platform(calibration):
    src = source(calibration, [Sy.Placed('platform', -0.1, 0.3, 20.)])
    first = src.poses()                                   # full view of the platform -> cached
    seat = first['base0'].compose(Pose((0.025, 0., 0.)))
    src.camera.objects.append(Sy.Placed('brick', seat.pos[0], seat.pos[1], math.degrees(seat.yaw)))
    src.camera._render()
    now = src.poses()
    err = seat.inverse().apply(now['b0_0'].pos)
    assert np.all(np.abs(err) < 0.003), err
    assert now['base0'] == first['base0']                 # partial view agrees -> the cached full view


def test_carried_brick_is_not_seen(calibration):
    src = source(calibration, [Sy.Placed('brick', 0.1, -0.3, 0., z_rel=0.09), Sy.Placed('platform', -0.2, 0.3, 0.)])
    assert 'b0_0' not in src.poses()


def test_two_bricks_are_ambiguous(calibration):
    src = source(calibration, [Sy.Placed('brick', 0.1, -0.3, 0.), Sy.Placed('brick', -0.2, -0.3, 0.),
                               Sy.Placed('platform', -0.2, 0.3, 0.)])
    with pytest.raises(V.SceneError):
        src.poses()


def test_brick_on_a_peak_is_flagged(calibration):
    # A brick resting 10 mm high (jammed on the sawtooth) is refused, not picked or counted as seated.
    src = source(calibration, [Sy.Placed('brick', 0.1, -0.3, 0., z_rel=0.0115 + 0.010)])
    with pytest.raises(V.SceneError, match='seated height'):
        src.poses()


@pytest.fixture(scope='module')
def compiled():
    return FP.compile_plan()


def test_field_plan(compiled, calibration):
    cam = Ca.CameraPose(calibration)
    sensed = {'b0_0': Pose((0.25, -0.35, 0.0115), (math.cos(0.1), 0, 0, math.sin(0.1))),
              'base0': Pose((-0.2, 0.4, 0.0115), (1, 0, 0, 0))}
    field = {'home': [-0.25, -0.6, 0.07], 'geofence': {'min': [-0.55, -0.85], 'max': [0.55, 0.85]}}
    plan = FP.adapt_plan(compiled, sensed, field, cam.floor_z, M.platform_origin_height, Rig()['names'])
    assert 'base0' not in plan['bases']                    # the mission must use the camera's platform
    assert plan['settings']['transit_z'] == pytest.approx(compiled['settings']['transit_z'] + 0.0115)
    assert plan['home']['pos'] == field['home']
    assert plan['bricks'][0]['pick']['slot_nominal'] == sensed['b0_0'].to_dict()
    check_plan(plan)                                       # retimed legs stay within speed limits
    assert 'sim' not in plan


def test_demo_mission_seats_the_sensed_brick_on_the_sensed_platform(compiled):
    s = run_demo(plan=compiled, log=lambda *a: None)
    assert s['mission_status'] == 'complete'
    assert s['bricks'][0]['outcome'] == 'seated'
    assert np.all(np.abs(s['final_brick_in_seat_frame_mm']) < 3)
    target = next(e for e in s['events'] if e['phase'] == 'PREPLACE_TARGET')['seat']['pos']
    assert np.linalg.norm(np.subtract(target[:2], (-0.20, 0.30))) < 0.003     # the platform's seat, as sensed


def test_demo_misplaced_brick_is_caught(compiled):
    s = run_demo(plan=compiled, place_error=(0.012, 0., 0.), log=lambda *a: None)
    places = [e for e in s['events'] if e['phase'] == 'VERIFY_PLACE']
    assert places[0]['status'] == 'recoverable'
    assert s['bricks'][0]['outcome'] != 'seated'


def test_lighthouse_yaml_solves():
    pytest.importorskip('yaml')
    from testflight import lighthouse as Lh
    r = Lh.solve()
    assert r['ok'] and len(r['base_stations']) == 2
    assert r['error_mm']['max'] < 10
