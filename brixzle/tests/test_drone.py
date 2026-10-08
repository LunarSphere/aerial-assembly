import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aerial_assembly.config import Physics
from brixzle import cad, drone as D, scene
from brixzle.cf import rot
from brixzle.drone_params import CF21B, DroneConfig, G
from brixzle.estimator import Lighthouse, LighthouseModel
from brixzle.params import V0
from brixzle.quat import wxyz_to_xyzw, xyzw_to_wxyz

QUIET = LighthouseModel(bias_sigma_mm=0, bias_gradient_mm_per_m=0, jitter_mm=0, vel_noise_mps=0,
                        att_noise_deg=0, latency_s=0)
DATA = Path(__file__).parent/'data'


def test_quaternion_conversion_round_trip_and_meaning():
    rng = np.random.default_rng(0)
    q = Rotation.random(20, random_state=1).as_quat()          # scipy xyzw
    assert np.allclose(wxyz_to_xyzw(xyzw_to_wxyz(q)), q)
    yaw90 = xyzw_to_wxyz(Rotation.from_euler('z', 90, degrees=True).as_quat())
    assert np.allclose(yaw90, [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])  # MuJoCo [w, x, y, z]
    v = rng.normal(size=3)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, yaw90)
    assert np.allclose(m.reshape(3, 3) @ v, Rotation.from_quat(wxyz_to_xyzw(yaw90)).apply(v))


def test_rotation_helpers_match_scipy():
    for r in Rotation.random(30, random_state=2):
        q = r.as_quat()
        assert np.allclose(rot.quat_to_mat(q), r.as_matrix())
        assert np.allclose(np.abs(rot.mat_to_quat(r.as_matrix()) @ q), 1.)
        assert np.allclose(rot.mat_to_euler(r.as_matrix()), r.as_euler('xyz'))
        assert np.allclose(rot.euler_to_mat(r.as_euler('xyz')), r.as_matrix())
        assert np.isclose(rot.quat_yaw(q), r.as_euler('xyz')[2])
        v = r.as_rotvec()
        assert np.allclose(rot.rotvec_to_mat(v), Rotation.from_rotvec(v).as_matrix())


def test_thrust_budget_numbers():
    assert CF21B.mass_g == pytest.approx(46.1)
    assert CF21B.t_w(30.) == pytest.approx(4*0.2942/(0.0761*G), rel=1e-6)
    assert CF21B.t_w(30.) > 1.5 > CF21B.t_w(30., 'crazyflow')
    assert CF21B.margins(30.)['binding'] == 'payload'
    assert DroneConfig(thrust_source='crazyflow').margins(10.)['binding'] == 'tw'
    assert CF21B.payload_at_tw(1.5, 'crazyflow') < 12.  # less than one V0 brick


def _hover(payload_g, seconds=6.):
    phys = Physics(timestep=0.001, friction=0.35, contact_timeconst=0.004, iterations=50)
    cfg = CF21B
    model, _, _ = scene.build_scene(V0, cad.build_brick(V0, solid=False), [(-20, -16)], 0, phys, visual=False,
                                    drone={'cfg': cfg, 'pos': (0, 0, .5)})
    if payload_g:
        model.body_mass[model.body('cf_deck').id] += payload_g*1e-3  # stand-in payload on the axis
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    total = model.body_subtreemass[model.body('cf').id]
    dr = D.Drone(model, data, cfg, Lighthouse(QUIET, np.random.default_rng(0)), ctrl_mass=D.firmware_mass(total))
    dr.place([0, 0, .5])
    dr.rotor_vel[:] = 20000.
    dr.cmd = D.full_state([0, 0, .5])
    D.step(model, data, dr, int(seconds/model.opt.timestep))
    return dr, total


@pytest.mark.parametrize('payload', [0., 17.6, 30.])
def test_hover_with_payload_thrust_matches_weight(payload):
    dr, total = _hover(payload)
    assert dr.thrusts.sum() == pytest.approx(total*G, rel=0.02)
    assert abs(dr.true_state()[0][2] - .5) < 0.01
    assert np.all(dr.thrusts < CF21B.thrust_per_motor_N['datasheet'])


def test_controller_matches_crazyflow_reference():
    """Port vs pure crazyflow on the same open-air waypoints (tests/data, made by scripts/crosscheck_crazyflow.py)."""
    spec = importlib.util.spec_from_file_location('cc', Path(__file__).parents[1]/'scripts/crosscheck_crazyflow.py')
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    out = cc.compare(DATA/'crazyflow_reference.npz')
    assert out['rms_m'] < 0.005 and out['max_m'] < 0.01


def test_lighthouse_bias_and_latency():
    m = LighthouseModel(bias_sigma_mm=10, bias_gradient_mm_per_m=0, jitter_mm=0, vel_noise_mps=0, att_noise_deg=0, registered=False,
                        latency_s=0.01)
    lh = Lighthouse(m, np.random.default_rng(3), dt=0.002)
    q = np.array([0, 0, 0, 1.])
    outs = [lh.observe([k*0.01, 0, 0], q, np.zeros(3), np.zeros(3))[0] for k in range(10)]
    assert np.allclose(outs[-1] - lh.b0, [0.04, 0, 0])            # 5 updates late
    assert np.allclose(LighthouseModel.from_dict({'registered': True}).registered, True)
    reg = Lighthouse(LighthouseModel(registered=True), np.random.default_rng(3))
    assert np.allclose(reg.b0, 0)
