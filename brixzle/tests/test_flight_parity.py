"""Sim vs real-hardware parity of the flight program. cflib is mocked; no radio is ever opened."""
import inspect
import json
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np
import pytest
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.commander import Commander
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.param import Param
from cflib.crazyflie.supervisor import Supervisor
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from aerial_assembly.config import Physics
from brixzle import cad, drone as D, flyassemble as FA, scene, simcf
from brixzle.drone_params import CF21B
from brixzle.estimator import Lighthouse, LighthouseModel
from brixzle.params import V0
from flight import fake, hardware

PLAN = Path(__file__).parents[2]/'flight/tests/data/plan_single.json'
QUIET = LighthouseModel(bias_sigma_mm=0, bias_gradient_mm_per_m=0, jitter_mm=0, vel_noise_mps=0, att_noise_deg=0,
                        latency_s=0)

PAIRS = [
    (HighLevelCommander, simcf.SimHighLevelCommander, ['takeoff', 'land', 'go_to', 'stop', 'define_trajectory',
                                                        'start_trajectory']),
    (Commander, simcf.SimCommander, ['send_position_setpoint', 'send_full_state_setpoint', 'send_notify_setpoint_stop',
                                     'send_stop_setpoint']),
    (Param, simcf.SimParam, ['set_value', 'get_value', 'add_update_callback', 'remove_update_callback']),
    (LogConfig, simcf.SimLogConfig, ['__init__', 'add_variable', 'start', 'stop', 'delete']),
    (SyncCrazyflie, simcf.SimSyncCrazyflie, ['__init__', 'open_link', 'close_link', 'wait_for_params',
                                             'is_params_updated']),
    (Supervisor, simcf.SimSupervisor, ['send_arming_request']),
]


@pytest.mark.parametrize('real, sim, methods', PAIRS, ids=[p[0].__name__ for p in PAIRS])
def test_sim_signatures_match_installed_cflib(real, sim, methods):
    for m in methods:
        assert inspect.signature(getattr(sim, m)) == inspect.signature(getattr(real, m)), f'{real.__name__}.{m}'


def test_sim_crazyflie_extends_cflib_constructor_only_keyword_only():
    real = list(inspect.signature(Crazyflie.__init__).parameters.values())
    sim = list(inspect.signature(simcf.SimCrazyflie.__init__).parameters.values())
    assert sim[:len(real)] == real
    assert all(p.kind is p.KEYWORD_ONLY for p in sim[len(real):])


def _hardware_calls(plan):
    """The real backend (``flight.hardware.fly``) with cflib's link and log classes mocked."""
    clock = fake.PublishingClock()
    cf = fake.RecordingCrazyflie(clock=clock, home=plan['home']['pos'])
    clock.cf = cf
    poses = fake.ScriptedPoseSource(plan)
    from flight import preflight
    run_preflight = preflight.run_preflight

    def preflight_then_reset(*args, **kwargs):
        report = run_preflight(*args, **kwargs)
        cf.rec.calls.clear()       # compare the mission only; the sim has no radio preflight
        return report
    with mock.patch.object(hardware, 'connect', return_value=fake.FakeSyncCrazyflie('radio://mock', cf=cf)), \
            mock.patch.object(hardware, 'RealClock', return_value=clock), \
            mock.patch('cflib.crazyflie.log.LogConfig', fake.FakeLogConfig), \
            mock.patch('flight.preflight.run_preflight', side_effect=preflight_then_reset):
        result = hardware.fly(plan, 'radio://mock', poses, arm=True, out=lambda *_: None, on_event=poses.on_event)
    return result, cf.rec.calls


def test_same_plan_gives_identical_cflib_calls_in_sim_and_on_mocked_hardware():
    plan = json.loads(PLAN.read_text())
    hw_result, hw_calls = _hardware_calls(plan)
    sim_rec = fake.Recorder()
    cfg = FA.config_from_dict(plan['sim']['config'])
    h = FA.Harness(V0, plan, cad.build_brick(V0), FA.Gr.build_gripper(FA.Gr.params_from_dict(plan['sim']['gripper']),
                                                                        cad.build_brick(V0), V0),
                   FA.S.CATALOG[plan['sim']['structure']](), cfg, seed=0)
    h.run(poses=fake.ScriptedPoseSource(plan), recorder=sim_rec)

    def norm(calls):   # log configs are different classes with the same name/period/variables
        return [c for c in calls if c[0] != 'log.add_config'] + [c for c in calls if c[0] == 'log.add_config']
    assert hw_result.status == 'complete'
    assert len(sim_rec.calls) > 50
    assert norm(sim_rec.calls) == norm(hw_calls)


def _drone_world():
    phys = Physics(timestep=0.001, friction=0.35, contact_timeconst=0.004, iterations=50)
    model, _, info = scene.build_scene(V0, cad.build_brick(V0, solid=False), [(-20, -16)], 0, phys, visual=False,
                                       drone={'cfg': CF21B, 'pos': (0, 0, .015)}, sleep=True)
    data = mujoco.MjData(model)
    dr = D.Drone(model, data, CF21B, Lighthouse(QUIET, np.random.default_rng(0)), ctrl_mass=D.firmware_mass(0.0461))
    dr.place([0, 0, .0153])
    world = simcf.SimWorld(model, data, info, dr, {})
    cf = simcf.SimCrazyflie(world=world)
    for name, value in (('stabilizer.estimator', 2), ('stabilizer.controller', 2), ('commander.enHighLevel', 1),
                        ('ctrlMel.mass', 0.0461)):
        cf.param.set_value(name, str(value))
    cf.supervisor.send_arming_request(True)
    return world, cf


def test_sim_go_to_reaches_target_at_duration():
    world, cf = _drone_world()
    cf.high_level_commander.takeoff(0.4, 2.0)
    world.advance(4.0)
    cf.high_level_commander.go_to(0.3, -0.2, 0.5, 0.5, 2.0)
    world.advance(2.0)
    pos, quat, _, _ = world.drone.true_state()
    assert np.linalg.norm(pos - [0.3, -0.2, 0.5]) < 0.02
    from brixzle.cf.rot import quat_yaw
    assert abs(quat_yaw(quat) - 0.5) < 0.05


def test_watchdog_levels_then_cuts_motors_when_setpoints_stop():
    world, cf = _drone_world()
    cf.high_level_commander.takeoff(0.5, 2.0)
    world.advance(3.0)
    for _ in range(20):
        cf.commander.send_position_setpoint(0.1, 0., 0.5, 0.)
        world.advance(0.05)
    world.advance(0.6)                                   # > 0.5 s without a setpoint
    assert world.drone.level and world.fw.motors_on
    z_level = world.drone.true_state()[0][2]
    assert z_level > 0.4                                 # level-out holds altitude
    world.advance(1.5)                                   # > 2 s total
    assert not world.fw.motors_on and world.fw.locked and world.drone.cmd is None
    world.advance(1.0)
    assert world.drone.true_state()[0][2] < z_level - 0.1   # falling with the motors off
    assert any('motors off' in m for _, m in world.fw.log)


def test_stop_cuts_motors_and_high_level_ignored_while_streaming():
    world, cf = _drone_world()
    cf.high_level_commander.takeoff(0.4, 2.0)
    world.advance(2.5)
    cf.commander.send_position_setpoint(0., 0., 0.4, 0.)
    cf.high_level_commander.go_to(1.0, 0., 0.4, 0., 1.0)   # low-level priority: ignored until notify
    for _ in range(20):
        cf.commander.send_position_setpoint(0., 0., 0.4, 0.)
        world.advance(0.05)
    assert abs(world.drone.true_state()[0][0]) < 0.03
    cf.commander.send_notify_setpoint_stop()
    cf.high_level_commander.go_to(0.2, 0., 0.4, 0., 1.0)
    world.advance(1.5)
    assert abs(world.drone.true_state()[0][0] - 0.2) < 0.03
    cf.high_level_commander.stop()
    world.advance(0.05)
    assert world.drone.cmd is None and not world.fw.motors_on
