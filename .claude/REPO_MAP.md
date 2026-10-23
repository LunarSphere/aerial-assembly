# Repo map

Keep this file current: when you add, remove, rename, or change the role of a module, update it in the same change.

## Top level
- `brixzle/` brick design, MuJoCo trials, NSGA-III search, drone assembly sim. Depends on `flight` (editable).
- `flight/` cflib flight program, Experiment 2 calibration and Experiment 3 approach accuracy (`fly`, `flight-cal`).
- `figures/` figures with their data, generating scripts and notes: `exp3-approach-accuracy/` (Exp 3 raw logs, tables, figures), `lighthouse-rest-jitter.*` (Lighthouse jitter at rest vs calibration and battery).
- `stl_exports/` CAD meshes (`2p5d_ab/`, `2p5d_ring/`, fork gripper) and `export_2p5d.py`.
- `.github/` CI (`workflows/tests.yml`: `flight` and `brixzle` jobs) and Dependabot.
- Git-ignored, local only: `runs/`, `assets/`, `.cache/`, weights, videos.

## brixzle/src/brixzle (CLI: `brixzle`, entry `cli.py`)
Geometry and design
- `params.py` brick parameters, V0 guess, optimizer bounds
- `profile.py` 2D XZ brick/base profile (shapely); `convex.py` convex decomposition
- `cad.py` CadQuery solid, exports, mass, collision pieces; `baseplate.py` anchored base
- `gripper.py` passive grippers; `rules.py` analytic design rules
- `lattice.py` voxel lattice, placements, assembly order; `structures.py` trial structures; `ring.py` 3D wall/turn structures
Simulation
- `scene.py` MJCF scene; `trials.py` drop, sequential assembly, push trials; `fixture.py` pick fixture
- `drone.py`, `drone_params.py` simulated Crazyflie 2.1 Brushless and thrust budget
- `cf/` numpy port of crazyflow Mellinger controller and dynamics (`mellinger.py`, `dynamics.py`, `params.py`, `rot.py`, `meshes/`)
- `simcf.py` simulated cflib API; `estimator.py` Lighthouse state model
- `planner.py` compiles `plan.json` (`fly-assemble`); `flyassemble.py` closed-loop sim run of `flight.mission`
Search and output
- `optimize.py` NSGA-III; `render.py` PNG/MP4 from qpos frames; `quat.py` quaternion order helpers; `config.py`
- `configs/` JSON params (nsga-*, ab*, lighthouse-*, drone, gripper); `scripts/` crosscheck and benchmark spikes; `docs/exp2/` media
- `tests/` pytest (about 80 s)

## flight/flight (CLIs: `fly` -> `cli.py`, `flight-cal` -> `calibration/cli.py`)
Mission (earlier, plan.json)
- `mission.py` executes plan.json via cflib only; `hardware.py` real Crazyflie backend; `fake.py` recording stand-ins
- `checks.py` offline geofence/speed/accel; `preflight.py` live checks; `poly7.py` firmware trajectory pieces
- `clock.py` injectable time; `frames.py` poses and quaternion conversion; `sensing.py` overhead brick poses
Calibration (`calibration/`)
- `setup.py`, `discovery.py`, `lighthouse.py` Lighthouse geometry and radio discovery
- `camera.py` D455 adapter; `vision.py` AprilTag geometry; `layout.py` tag survey; `registration.py` camera-to-Lighthouse pose; `drone.py` detectors
- `hover.py` hover-check; `visits.py` tag-visit and over-centre routes; `rig.py`, `presets.py`, `artifacts.py` shared types, presets, JSON artifacts
- `accuracy.py` Exp 3 approach accuracy: tag survey, trial sessions (`approach-accuracy`), offline metrics and plot (`approach-metrics`)
- `configs/` intrinsics, tag layout, drone reference; `tests/` pytest (about 2 s, mocks cflib)
