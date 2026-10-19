# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

Two independent Python projects, each with its own environment. Run commands from the directory noted.

| Dir | Package / CLI | Toolchain |
|---|---|---|
| `brixzle/` | `brixzle` / `brixzle` | uv, Python 3.12 (CadQuery), MuJoCo drop trials and NSGA-III brick search; depends on `../flight` as an editable path dependency |
| `flight/` | `flight` / `fly`, `flight-cal` | uv, cflib 0.1.33; real Crazyflie or simulator |

Other dirs: `stl_exports/` (CAD meshes, incl. `2p5d_ab/` and the fork gripper). `runs/`, `assets/`, `.cache/`, model weights (`*.onnx`, `*.pt`), videos are git-ignored and local-only.

## Commands

**brixzle/**
```bash
uv sync --all-extras
uv run pytest
uv run brixzle export --params configs/v0.json --out runs/v0
uv run brixzle fly-assemble tower --bricks 1 --out runs/plan-single   # compiles plan.json for flight/
```

**flight/** (run from the repo root so `runs/` paths resolve)
```bash
uv run --project flight --extra test pytest flight/tests -q
uv run --project flight --extra test pytest flight/tests/test_calibration.py -k name
uv run --project flight --extra vision --extra realsense --extra detector flight-cal <setup|hover-check|tag-visit|camera|lighthouse|...> --out runs/<new-dir>
uv run --project brixzle fly runs/plan-single/plan.json --sim      # earlier plan.json mission; --sim needs brixzle's env
```
Tests mock cflib; they never open a radio.

## Architecture

- **Mission vs. hardware (`flight/flight/`)**: `mission.py` talks only to the cflib API, an injected clock, and a camera pose source, so one `plan.json` flies in MuJoCo (`--sim` lazily imports `brixzle`) or on a real Crazyflie. `fake.py` records cflib calls for dry runs; `checks.py` (offline geofence/speed/accel checks) and `preflight.py` (live deck/battery/Kalman checks) gate real flights. `poly7.py` mirrors firmware trajectory pieces. This is the *earlier* mission; documented at the end of `flight/README.md`.
- **Calibration pipeline (`flight/flight/calibration/`)**: Lighthouse geometry → overhead D455 camera registered to the Lighthouse frame via fixed AprilTags (`registration.py`, `camera.py`, `layout.py`) → flight routes: `hover.py` (hover-check), `visits.py` (tag-visit). `cli.py` is the `flight-cal` entry point; setup output (`runs/setup`) is reused by later commands. Docs: `flight/README.md`.
- **Safety model of `flight-cal` flights**: default is a dry-run route preview (`route.json`, `route-preview.png`); `--check-only` runs camera/radio preflight without arming; only `--arm` flies. `--demo` refreshes the camera pose from fixed tags and uses PID with relaxed tolerances. Every run needs a new `--out` directory.

## Conventions

- SI units; quaternions are `[w, x, y, z]` everywhere. Convert to scipy/crazyflow's scalar-last order only via `flight/flight/frames.py` (`wxyz_to_xyzw`).
- Four-space indent, `snake_case`, dataclasses for settings, explicit input validation, finite JSON. No formatter or linter configured.
