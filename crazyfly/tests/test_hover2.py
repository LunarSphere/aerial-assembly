import unittest
from unittest.mock import MagicMock, call, patch

import hover
import hover2


class ImmediateFlightTests(unittest.TestCase):
    @patch("hover.cflib.crtp.init_drivers")
    @patch("hover.Crazyflie")
    @patch("hover.SyncCrazyflie")
    @patch("hover.wait_for_parameters")
    @patch("hover.set_parameter")
    @patch("hover.connected_sleep")
    @patch("hover.position_samples")
    @patch("hover.wait_for_position")
    @patch("hover.fly")
    def test_no_position_wait_and_check_only(self, fly, wait, samples, sleep,
                                           set_param, params, sync, cf_type, init):
        cf = MagicMock()
        cf.param.get_value.return_value = "1"
        sync.return_value.__enter__.return_value.cf = cf
        args = hover.parse_args([])
        hover2.run(args)
        wait.assert_not_called()
        samples.assert_not_called()
        fly.assert_called_once_with(cf, 0.0, 0.3, 5.0, 0.25, 0.2)
        self.assertEqual(set_param.call_args_list[-3:], [
            call(cf, "kalman.initialZ", 0.0),
            call(cf, "kalman.resetEstimation", 1),
            call(cf, "kalman.resetEstimation", 0)])
        sync.return_value.__exit__.assert_called_once()

        fly.reset_mock()
        args.check_only = True
        hover2.run(args)
        fly.assert_not_called()

        args.check_only = False
        cf.param.get_value.return_value = "0"
        with self.assertRaisesRegex(RuntimeError, "No Flow deck"):
            hover2.run(args)
        fly.assert_not_called()


if __name__ == "__main__":
    unittest.main()
