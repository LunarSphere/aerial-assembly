# flight

One flight program for the Brixzle drone pick-and-place experiment (experiment 2). The mission
(`flight/mission.py`) talks only to the cflib 0.1.33 API, an injected clock and a camera pose
source, so the same `plan.json` flies in the MuJoCo simulator or on a real Crazyflie 2.1 Brushless.
This package does not import CadQuery or MuJoCo; `--sim` imports `brixzle` lazily.

```bash
# Compile a plan (brixzle environment)
cd brixzle && uv run brixzle fly-assemble tower --bricks 1 --out runs/plan-single

# Simulator
uv run fly runs/plan-single/plan.json --sim [--seed 0] [--out runs/fly-sim --video]

# Real hardware: dry run (default) -- offline checks + the cflib calls it would send, no radio
uv run fly runs/plan-single/plan.json --uri radio://0/80/2M/E7E7E7E7E7 [--show-calls]
# Connect and run the preflight only (never arms)
uv run fly runs/plan-single/plan.json --uri radio://0/80/2M/E7E7E7E7E7 --check-only
# Supervised flight only
uv run fly runs/plan-single/plan.json --uri radio://0/80/2M/E7E7E7E7E7 --arm --realsense src/testflight/rig.json
```

Real-hardware behaviour:

- **Preflight** (`flight/preflight.py`, after `crazyfly/hover.py`): acknowledged param writes,
  Lighthouse deck (`deck.bcLighthouse4`), Kalman reset and variance convergence, `pm.vbat` >= 3.8 V,
  drone resting on the launch stand at the plan's home pose, inside the geofence.
- **Controller**: `stabilizer.controller = 2` (Mellinger), unlike `hover.py` (PID), because the
  simulator ports crazyflow's Mellinger. The plan also sets `ctrlMel.ki_m_xy` / `i_range_m_xy` (roll/pitch
  attitude integral, 0 in the firmware default) and `ctrlMel.mass` after each pick and release.
  **Review these values before the first flight.**
- **Offline checks** (`flight/checks.py`): every waypoint inside the geofence, peak speed/acceleration of
  each firmware `go_to` piece and streamed segment, setpoint rate >= 10 Hz (watchdog: 0.5 s level, 2 s off).
- **Ctrl+C or any exception**: hand back to the high-level commander, land in place to
  `emergency_land_z`, `stop()`, disarm.
- **Camera**: `--realsense brixzle/src/testflight/rig.json` senses the brick and the platform with the overhead
  D455 before every pick and check (see `brixzle/src/testflight/README.md`; `--replay DIR...` replays captures).
  `--poses poses.json` remains as a manual fallback `{name: {"pos": [m], "quat": [w, x, y, z]}}`, re-read before
  every pick and check.

Later validation path (not built): CrazySim (https://github.com/gtfactslab/CrazySim) runs the real firmware
in the loop and accepts cflib over `udp://127.0.0.1:19850`; Ubuntu only.

Tests (cflib mocked, never a radio): `uv run --project brixzle python -m unittest discover -s flight/tests`
and `brixzle/tests/test_flight_parity.py`.
