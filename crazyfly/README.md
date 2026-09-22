# Flow deck flight

Default sequence: take off **30 cm**, hover **5 seconds**, fly **0.25 metres
forward** at 0.2 m/s, pause to brake, then land. Forward follows the drone's nose.
Takeoff and landing each take about three seconds. Uses the attached Flow deck;
no Lighthouse or external positioning setup is needed.

## Setup

Use a charged Crazyflie with a Flow deck and Crazyradio. The downward camera
needs a well-lit, textured floor and the range sensor needs a clear view of it.
Start stationary on a level floor with clearance ahead. Disconnect cfclient
before running. Default radio URI: `radio://0/80/2M`.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python hover.py --check-only
```

Check-only verifies Flow deck detection and estimator stability, without arming.
It does change controller/estimator settings and reset the estimator. A passing
check does not guarantee optical-flow quality throughout the flight.

To perform the flight:

```sh
.venv/bin/python hover.py
```

Optional settings (these are the defaults):

```sh
.venv/bin/python hover.py --uri radio://0/80/2M --height 0.30 --duration 5 --distance 0.25 --speed 0.2
```

Height and distance are in metres; speed is metres/second. Forward distance is
implemented as speed multiplied by time, so it is approximate and may drift.
The script streams body-frame hover commands at 20 Hz with zero yaw rate and
ramps height relative to the measured launch Z. It does not command a global
horizontal position or use Lighthouse geometry.

Preflight requires a detected Flow deck (`deck.bcFlow2` or `deck.bcFlow`), selects
Kalman/PID, disables the high-level commander, and resets the estimator. It
requires ten fresh samples with variance spread below 0.001 and position spread
below 2 cm per axis, with a 30-second timeout even if telemetry stops arriving.
Parameter acknowledgements have separate five-second timeouts. Settings are not
restored on exit.

Ctrl+C attempts to stop forward motion and descend from the last commanded
height, then sends stop/disarm. An exception during landing or another Ctrl+C
triggers stop/disarm, which can cut motors before touchdown. Radio loss prevents
scripted landing; onboard behavior depends on firmware. In-flight optical-flow
quality is not monitored. Keep the first flight supervised.

## Variant without the stability wait

```sh
.venv/bin/python hover2.py
```

`hover2.py` performs the same 30 cm takeoff, five-second hover, 0.25 m forward
flight, and landing, without subscribing to position telemetry or waiting for
estimator convergence. It initializes launch Z to zero; start on the floor.
Connection, parameter acknowledgements, Flow deck detection, estimator reset,
and the one-second arming wait still run. Position stability is not verified
before takeoff. It shares the flight and cleanup code in `hover.py`, so keep both
files together. All command-line options are the same; `--check-only` only checks
the deck and configuration in this variant.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests use mocked hardware; no automatic test arms the drone. Hardware validation
is still required: run check-only, then observe takeoff, hover, forward travel,
landing, and stopped motors.

Reference: [Bitcraze Flow deck motion example](https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/step-by-step/sbs_motion_commander.py)
and [Python API guide](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/user-guides/python_api/).
