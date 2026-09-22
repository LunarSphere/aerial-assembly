import argparse
from contextlib import redirect_stderr
import io
from queue import Queue
import time
import unittest
from unittest.mock import MagicMock, call, patch

import hover


def fake_cf():
    cf = MagicMock()
    cf.is_connected.return_value = True
    cf.platform.get_protocol_version.return_value = 12
    cf.supervisor.is_armed = True
    cf.param.get_value.return_value = "1"
    return cf


def samples_for(positions, variance=0.0002):
    queue = Queue()
    for position in positions:
        data = dict(zip(hover.VARIANCE_NAMES, [variance] * 3))
        data.update(zip(hover.POSITION_NAMES, position))
        queue.put((time.monotonic(), data))
    return queue


class FlightTests(unittest.TestCase):
    @patch("hover.time.sleep")
    @patch("hover.connected_sleep")
    @patch("hover.stream_hover")
    def test_sequence(self, stream, wait, _sleep):
        cf = fake_cf()
        stream.side_effect = lambda *a, **kw: kw.get("on_height", lambda z: None)(a[3])
        hover.fly(cf, 1.2, 0.3, 5)
        self.assertEqual(stream.call_args_list[0].args, (cf, 3.0, 1.2, 1.5))
        self.assertEqual(stream.call_args_list[1:], [
            call(cf, 5, 1.5, 1.5),
            call(cf, 1.25, 1.5, 1.5, vx=0.2),
            call(cf, 0.5, 1.5, 1.5),
            call(cf, 3.0, 1.5, 1.2),
            call(cf, 0.5, 1.2, 1.2)])
        cf.commander.send_stop_setpoint.assert_called_once()
        self.assertEqual(cf.supervisor.send_arming_request.call_args_list,
                         [call(True), call(False)])
        wait.assert_called_once_with(cf, 1.0)
        cf.high_level_commander.takeoff.assert_not_called()

    @patch("hover.time.sleep")
    @patch("hover.connected_sleep")
    @patch("hover.stream_hover")
    def test_interrupt_lands_from_partial_height(self, stream, _wait, _sleep):
        cf = fake_cf()

        def interrupt(*args, **kwargs):
            if "on_height" in kwargs:
                kwargs["on_height"](-0.1)
                raise KeyboardInterrupt()

        stream.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt):
            hover.fly(cf, -0.2, 0.3, 5)
        self.assertEqual(stream.call_args_list[1], call(cf, 3.0, -0.1, -0.2))
        cf.commander.send_stop_setpoint.assert_called_once()
        self.assertEqual(cf.supervisor.send_arming_request.call_args, call(False))

    @patch("hover.time.sleep")
    @patch("hover.connected_sleep")
    @patch("hover.stream_hover")
    def test_send_and_landing_failures_cleanup(self, stream, _wait, _sleep):
        for effects in ([RuntimeError("send failed"), None, None],
                        [None, None, None, None, RuntimeError("landing failed")]):
            with self.subTest(effects=effects):
                cf = fake_cf()
                stream.reset_mock()
                stream.side_effect = effects
                with self.assertRaises(RuntimeError):
                    hover.fly(cf, 0, 0.3, 5)
                cf.commander.send_stop_setpoint.assert_called_once()
                self.assertEqual(cf.supervisor.send_arming_request.call_args, call(False))

    @patch("hover.time.sleep")
    @patch("hover.connected_sleep")
    @patch("hover.stream_hover")
    def test_refused_arming_does_not_take_off(self, stream, _wait, _sleep):
        cf = fake_cf()
        cf.supervisor.is_armed = False
        with self.assertRaisesRegex(RuntimeError, "arming"):
            hover.fly(cf, 0, 0.3, 5)
        stream.assert_not_called()
        self.assertEqual(cf.supervisor.send_arming_request.call_args, call(False))

    @patch("hover.connected_sleep")
    @patch("hover.stream_hover")
    def test_disconnect_does_not_claim_to_land(self, stream, _wait):
        cf = fake_cf()

        def lose_link(_cf, seconds, *args, **kwargs):
            if seconds == 5:
                cf.is_connected.return_value = False
                raise ConnectionError("lost")

        stream.side_effect = lose_link
        with redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(ConnectionError):
                hover.fly(cf, 0, 0.3, 5)
        self.assertIn("cannot command landing", output.getvalue())
        self.assertEqual(stream.call_count, 2)
        cf.commander.send_stop_setpoint.assert_not_called()

    @patch("hover.time.sleep")
    @patch("hover.time.monotonic", side_effect=[0, 0, 0.05, 0.1])
    def test_stream_forward_and_height_ramp(self, _clock, sleep):
        cf = fake_cf()
        hover.stream_hover(cf, 0.1, 0.2, 0.3, vx=0.2)
        self.assertEqual(cf.commander.send_hover_setpoint.call_args_list, [
            call(0.2, 0, 0, 0.2), call(0.2, 0, 0, 0.25), call(0, 0, 0, 0.3)])
        self.assertEqual(sleep.call_count, 2)


class PreflightTests(unittest.TestCase):
    def test_stable_position_is_measured(self):
        result = hover.wait_for_position(fake_cf(), samples_for([(2, -1, 0.4)] * 10))
        for actual, expected in zip(result, (2, -1, 0.4)):
            self.assertAlmostEqual(actual, expected)

    def test_no_telemetry_has_real_timeout(self):
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            hover.wait_for_position(fake_cf(), Queue(), timeout=0.01)
        self.assertLess(time.monotonic() - start, 1)

    def test_moving_position_does_not_pass(self):
        queue = samples_for([(i * 0.1, 0, 0) for i in range(10)])
        with self.assertRaises(TimeoutError):
            hover.wait_for_position(fake_cf(), queue, timeout=0.01)

    def test_unstable_variance_does_not_pass(self):
        queue = samples_for([(0, 0, 0)] * 10)
        timestamp, data = queue.get()
        data[hover.VARIANCE_NAMES[0]] = 1.0
        queue.put((timestamp, data))
        with self.assertRaises(TimeoutError):
            hover.wait_for_position(fake_cf(), queue, timeout=0.01)

    def test_nan_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid"):
            hover.wait_for_position(fake_cf(), samples_for([(0, 0, float("nan"))]))

    def test_old_samples_are_rejected(self):
        queue = Queue()
        for _ in range(10):
            queue.put((time.monotonic() - 10, {}))
        with self.assertRaises(TimeoutError):
            hover.wait_for_position(fake_cf(), queue, timeout=0.01)

    def test_log_error_is_reported(self):
        queue = Queue()
        queue.put((time.monotonic(), RuntimeError("missing log variable")))
        with self.assertRaisesRegex(RuntimeError, "missing log"):
            hover.wait_for_position(fake_cf(), queue)

    def test_parameter_acknowledgement_and_callback_cleanup(self):
        cf = fake_cf()

        def acknowledge(name, value):
            cb = cf.param.add_update_callback.call_args.kwargs["cb"]
            cb(name, value)

        cf.param.set_value.side_effect = acknowledge
        hover.set_parameter(cf, "stabilizer.estimator", 2)
        cf.param.remove_update_callback.assert_called_once_with(
            group="stabilizer", name="estimator",
            cb=cf.param.add_update_callback.call_args.kwargs["cb"])

    @patch("hover.position_samples")
    @patch("hover.connected_sleep")
    @patch("hover.set_parameter")
    def test_preflight_sets_controller_and_resets_before_logging(self, set_param, _sleep, logs):
        scf = MagicMock()
        scf.cf = fake_cf()
        scf.is_params_updated.return_value = True
        logs.return_value.__enter__.return_value = samples_for([(0, 0, 0)] * 10)
        hover.preflight(scf)
        self.assertEqual(set_param.call_args_list, [
            call(scf.cf, "stabilizer.estimator", 2),
            call(scf.cf, "stabilizer.controller", 1),
            call(scf.cf, "commander.enHighLevel", 0),
            call(scf.cf, "kalman.resetEstimation", 1),
            call(scf.cf, "kalman.resetEstimation", 0),
        ])
        scf.cf.supervisor.send_arming_request.assert_not_called()


class RunTests(unittest.TestCase):
    def test_defaults(self):
        args = hover.parse_args([])
        self.assertEqual((args.uri, args.height, args.duration, args.distance),
                         ("radio://0/80/2M", 0.3, 5.0, 0.25))

    @patch("hover.set_parameter")
    def test_missing_flow_deck_aborts(self, set_param):
        scf = MagicMock()
        scf.cf = fake_cf()
        scf.cf.param.get_value.return_value = "0"
        scf.is_params_updated.return_value = True
        with self.assertRaisesRegex(RuntimeError, "No Flow deck"):
            hover.preflight(scf)
        set_param.assert_not_called()
        scf.cf.supervisor.send_arming_request.assert_not_called()

    @patch("hover.cflib.crtp.init_drivers")
    @patch("hover.Crazyflie")
    @patch("hover.SyncCrazyflie")
    @patch("hover.preflight")
    @patch("hover.fly")
    def test_check_only_and_failed_preflight_never_fly(self, fly, preflight, sync, cf, _init):
        preflight.return_value = (0, 0, 0)
        args = hover.parse_args(["--uri", "radio://test", "--check-only"])
        hover.run(args)
        fly.assert_not_called()
        sync.return_value.__exit__.assert_called_once()
        args.check_only = False
        preflight.side_effect = TimeoutError("unstable")
        with self.assertRaises(TimeoutError):
            hover.run(args)
        fly.assert_not_called()
        cf.return_value.supervisor.send_arming_request.assert_not_called()

    def test_nonpositive_and_nonfinite_arguments_rejected(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                hover.positive_float(value)


if __name__ == "__main__":
    unittest.main()
