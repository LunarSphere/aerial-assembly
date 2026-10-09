"""flight package tests: firmware polynomials, offline checks, dry run, Ctrl+C landing. No radio, no MuJoCo.

Run from the repo with the brixzle environment: ``uv run --project brixzle python -m unittest discover -s flight/tests``.
"""
import copy
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from flight import checks, cli, fake, hardware, poly7
from flight.frames import Pose
from flight.mission import Mission
from flight.sensing import JsonPoseSource, RealSensePoseSource

PLAN = json.loads((Path(__file__).parent/'data/plan_single.json').read_text())


class Poly7Tests(unittest.TestCase):
    def test_boundary_conditions_match_firmware_definition(self):
        c = poly7.poly7_nojerk(1.7, 0.2, 0.3, -0.1, 1.1, 0.0, 0.0)
        p = np.polynomial.polynomial
        for k, (start, end) in enumerate([(0.2, 1.1), (0.3, 0.0), (-0.1, 0.0)]):
            d = p.polyder(c, k) if k else c
            self.assertAlmostEqual(p.polyval(0, d), start)
            self.assertAlmostEqual(p.polyval(1.7, d), end)
        jerk = p.polyder(c, 3)
        self.assertAlmostEqual(p.polyval(0, jerk), 0)
        self.assertAlmostEqual(p.polyval(1.7, jerk), 0)

    def test_go_to_holds_goal_after_duration_and_takes_shortest_yaw(self):
        cur = poly7.TrajEval(np.zeros(3), np.zeros(3), np.zeros(3), 3.0, 0.)
        piece = poly7.plan_go_to_from(10., cur, [1, 2, 3], -3.0, 2.0)
        end = piece.eval(15.)
        np.testing.assert_allclose(end.pos, [1, 2, 3])
        np.testing.assert_allclose(end.vel, 0)
        self.assertAlmostEqual(end.yaw, 2*np.pi - 3.0)  # +0.28 rad, not -6 rad


class CheckTests(unittest.TestCase):
    def test_compiled_plan_passes(self):
        summary = checks.check_plan(PLAN)
        self.assertEqual(summary['bricks'], 1)
        self.assertLess(summary['peak_speed_mps'], PLAN['settings']['limits']['v_max'])

    def test_geofence_and_rate_violations_are_reported(self):
        plan = copy.deepcopy(PLAN)
        plan['settings']['geofence']['max'][2] = 0.05
        plan['settings']['setpoint_hz'] = 5
        with self.assertRaises(checks.PlanError) as ctx:
            checks.check_plan(plan)
        self.assertIn('outside geofence', str(ctx.exception))
        self.assertIn('setpoint_hz', str(ctx.exception))


class DryRunTests(unittest.TestCase):
    def test_default_is_dry_run_and_never_connects(self):
        with mock.patch.object(hardware, 'connect', side_effect=AssertionError('radio opened')), \
                mock.patch('sys.stdout', new_callable=io.StringIO) as out:
            code = cli.main([str(Path(__file__).parent/'data/plan_single.json'), '--uri', 'radio://0/80/2M/E7E7E7E7E7'])
        self.assertEqual(code, 0)
        self.assertIn('DRY RUN', out.getvalue())
        self.assertIn('"dry_run_status": "complete"', out.getvalue())

    def test_dry_run_mission_sequence(self):
        summary, calls = hardware.dry_run(PLAN, out=lambda *_: None)
        names = [c[0] for c in calls]
        self.assertEqual(names.count('supervisor.send_arming_request'), 2)   # arm, then disarm after landing
        self.assertLess(names.index('hl.takeoff'), names.index('cmd.send_full_state_setpoint'))
        self.assertEqual(names[-2:], ['hl.stop', 'supervisor.send_arming_request'])
        # Every low-level stream is closed before the high-level commander is used again.
        last = None
        for n in names:
            if n.startswith('hl.') and last == 'cmd.send_full_state_setpoint':
                self.fail('high-level command while low-level setpoints still have priority')
            if n.startswith(('hl.', 'cmd.')):
                last = n

    def test_cli_rejects_unsafe_combinations(self):
        with self.assertRaises(SystemExit):
            cli.parse_args(['p.json', '--sim', '--arm'])
        with self.assertRaises(SystemExit):
            cli.parse_args(['p.json', '--uri', 'radio://x', '--arm', '--check-only'])
        with self.assertRaises(SystemExit):
            cli.main([str(Path(__file__).parent/'data/plan_single.json'), '--uri', 'radio://x', '--arm'])


class InterruptTests(unittest.TestCase):
    def test_ctrl_c_lands_in_place_then_stops_and_disarms(self):
        clock = fake.PublishingClock()
        cf = fake.RecordingCrazyflie(clock=clock, home=PLAN['home']['pos'])
        clock.cf = cf
        poses = fake.ScriptedPoseSource(PLAN)

        def interrupt(e):
            poses.on_event(e)
            if e.get('phase') == 'TRANSIT':
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            Mission(fake.FakeSyncCrazyflie('x', cf=cf), PLAN, clock, poses, fake.FakeLogConfig,
                    on_event=interrupt).run()
        names = [c[0] for c in cf.rec.calls]
        self.assertEqual(names[-3:], ['hl.land', 'hl.stop', 'supervisor.send_arming_request'])
        self.assertEqual(cf.rec.calls[-1][1], False)
        land = [c for c in cf.rec.calls if c[0] == 'hl.land'][-1]
        self.assertAlmostEqual(land[1], round(PLAN['settings']['emergency_land_z'], 9))


    def test_interrupt_during_low_level_stream_hands_back_before_landing(self):
        clock = fake.PublishingClock()
        cf = fake.RecordingCrazyflie(clock=clock, home=PLAN['home']['pos'])
        clock.cf = cf
        poses = fake.ScriptedPoseSource(PLAN)

        def interrupt(e):
            poses.on_event(e)
            if e.get('phase') == 'RELEASE' and e.get('op') == 'line':
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            Mission(fake.FakeSyncCrazyflie('x', cf=cf), PLAN, clock, poses, fake.FakeLogConfig,
                    on_event=interrupt).run()
        names = [c[0] for c in cf.rec.calls]
        self.assertEqual(names[-4:-1], ['cmd.send_notify_setpoint_stop', 'hl.land', 'hl.stop'])
        self.assertIn('cmd.send_full_state_setpoint', names[-12:-4])   # it was streaming when interrupted


class SensingTests(unittest.TestCase):
    def test_json_poses_and_realsense_stub(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'poses.json'
            p.write_text(json.dumps({'b0_0': {'pos': [0.1, 0.2, 0.0], 'quat': [0, 0, 0, 1]}}))
            poses = JsonPoseSource(p).poses()
        self.assertAlmostEqual(abs(poses['b0_0'].yaw), np.pi)
        with self.assertRaises(FileNotFoundError):       # no calib/camera_pose.json next to this rig
            RealSensePoseSource(Path(__file__).parent/'data'/'no-rig.json')

    def test_pose_compose_inverse(self):
        a = Pose((0.1, -0.2, 0.3), (np.cos(0.4), 0, 0, np.sin(0.4)))
        b = Pose((0.02, 0.01, -0.03), (1, 0, 0, 0))
        r = a.compose(b).relative_to(a)
        np.testing.assert_allclose(r.pos, b.pos, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
