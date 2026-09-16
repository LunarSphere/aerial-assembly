# Passive aerial assembly experiments

A small Python/MuJoCo workflow for one fixed block and one rigid falling block:
Onshape-to-robot or manual CAD import → geometry checks → deterministic drop → sweeps and random trials → candidate comparison.

**The demo geometry is a synthetic software test fixture, not your Onshape block.** It has stepped landings, a ramp, two blind tapered sockets, and two fixed pegs. It omits your guide lips and uses 30 mm legs so the demonstrated flush pose is possible. Real-block results require corrected exports and verified mating geometry; see the inspection of `export_example` below. No Onshape document was changed during development.

## Quick start

This workspace already has a working `.venv`. From `/home/kevius/aerial-assembly`:

```bash
source .venv/bin/activate
aerial demo --out assets/demo.json
aerial validate assets/demo.json --out runs/geometry-validation.json
aerial drop assets/demo.json --out runs/my-first-drop
aerial replay runs/my-first-drop --collisions
```

Run directories must be new: the CLI does not overwrite an existing experiment. Replay needs a graphical display; every scientific run is headless. Replay displays recorded states, so viewer interaction cannot change the scored experiment. The simulation runs for 3 seconds of simulated time, which can take appreciably longer than 3 wall-clock seconds.

Fresh installation (Python 3.11+; locked environment tested with Python 3.14.4 on Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
python -m pytest -q
```

If your platform lacks wheels for the locked versions, install `python -m pip install -e '.[test]'` and retain the resolved package versions with your results. Debian systems need the matching `python3-venv` package to create environments normally.

## Experiments

All lengths and velocities are SI; command-line release angles are degrees. See [examples/experiment.json](examples/experiment.json) for editable settings.

```bash
# Short diagnostic run; not the standard 3-second experiment.
aerial drop assets/demo.json --config examples/quick-check.json --out runs/quick

# Deterministic capture sweep. Output includes CSV, individual results, and release states.
aerial sweep assets/demo.json --axis x --values -0.005 -0.002 0 0.002 0.005 --out runs/x-sweep

# Seeded independent uniform uncertainty. Default: ±5 mm XY, ±5° roll/pitch,
# ±10° yaw, 10–50 mm additional clearance, zero initial velocity.
aerial batch assets/demo.json --count 100 --seed 42 --out runs/random-42

# Explicit Release objects support measured, correlated release distributions.
aerial batch assets/demo.json --releases measured-releases.json --out runs/measured

# Timestep halving keeps contact response fixed; aligned, miss, and offset trials.
aerial converge assets/demo.json --out runs/timestep-check

# Change facet count to test collision approximation independently.
aerial demo --facets 96 --out assets/demo-finer.json
aerial compare assets/demo.json assets/demo-finer.json --count 100 --seed 42 --out runs/facet-comparison

# Small exhaustive synthetic geometry search, not optimization of your CAD.
aerial demo-grid --ramp-angles 45 50 55 --lead-angles 25 30 35 --count 100 --seed 42 --out runs/demo-search

# Compare actual prepared CAD candidates with common release samples.
aerial compare assets/candidate-a.json assets/candidate-b.json --count 100 --seed 42 --out runs/cad-search
```

Use `--config examples/quick-check.json --count 3` for a fast pipeline check before a large batch. It uses a deliberately narrow release distribution and cannot characterize capture performance. For finalists use an independent seed and 1,000 trials, after calibrating physics and passing convergence tests.

The added release height is measured above **vertically separated world-axis bounding boxes**. This conservative convention prevents starting collisions even with tilt. Consequently it exceeds the gap between some corresponding peg tips and socket mouths. `initial_state.clearance_lift` and the exact initial pose are saved. Compare designs with the same release convention. To reproduce a measured initial pose exactly, use the `run_drop` Python interface with explicit `qpos` and `qvel`.

Perturbations rotate the upper block about its COM using extrinsic XYZ angles in world axes. Input velocities refer to the COM in world axes; the adapter converts to MuJoCo's free-joint origin velocity and local angular velocity. Quaternions in manifests and saved states are `[w,x,y,z]`.

## Results and numerical checks

Each experiment saves `scene.xml`, a self-contained hashed `geometry.json`, geometry validation, release samples, settings, summary JSON, CSV, and one result per trial. `drop` always records a compressed trajectory; use `batch --record` to retain batch trajectories. An interruption preserves already completed trials. Runs are not automatically resumed.

Success requires the target pose, both legs within the intended sockets at their target depths, near-zero seating gaps, supporting force, low speeds, and an uninterrupted final dwell. Default limits:

| Quantity | Limit |
|---|---|
| Translation error | 0.5 mm |
| Rotation error | 1° |
| Seating gap | 0.1 mm |
| Insertion-depth error | 0.5 mm |
| COM speed | 1 mm/s |
| Angular speed | 1°/s |
| Final dwell | 0.5 seconds |
| Receiver penetration | Bundle-specific; demo 25 µm |

Physical outcomes are `success`, `missed_receiver`, `stationary_misalignment`, `incomplete_insertion`, and `unsettled_timeout`. Intersecting starts, solver warnings, and excessive **receiver** penetration produce `invalid`. Floor contact is an irreversible miss; floor penetration is reported separately and does not use socket tolerances. Trials continue to the fixed horizon unless their state is numerically invalid.

CLI exit status 0 means the experiment completed with valid numerical trials, even if the block failed to assemble. Exit status 2 means invalid geometry/numerics or a configuration error. Results remain available for diagnosis. Optimization rankings exclude candidates with any invalid trials or a geometry rejection. Summaries report invalid frequency; seeded random runs also report a Wilson interval among valid trials. Deterministic runs and user-supplied sample lists do not claim a binomial confidence interval. Conditional success must not conceal excluded trials.

The original proposed `dt=0.25 ms`, `solref=(5 ms,1)` allowed approximately 2 mm of impact penetration on the synthetic fixture. Current defaults are **dt=12.5 µs, contact time constant=50 µs**, with implicitfast/Newton, elliptic sliding friction, native convex collision, and multiple contacts. The 25 µs offset case disagreed with refinement; the 12.5 µs and 6.25 µs offset cases both seated successfully. These are numerical debugging settings, not calibrated plastic properties. A coarse run passing by itself is insufficient: halving dt changes impact sampling. Retest every geometry and release envelope. See [validation evidence](VALIDATION.md).

The demo recesses socket-sector tops by 5 µm and supplies broad seating strips to reduce redundant support contacts at partition seams. This approximation is declared in its provenance. Socket faceting at 48 sectors is within the demo's 25 µm surface-error budget. Refine collision geometry separately from timestep. The convergence check allows final translational differences up to the functional clearance because there is a continuum of valid peg positions within a bore; misses compare outcome rather than exact floor-rest pose.

## Your CAD and Onshape

For the supplied GOAT export, the [download → drop → tweak workflow](docs/cad-iteration.md)
automates local rigid assembly, collision preparation and feature measurement:

```bash
aerial cad-drop export_example --out runs/goat-v1 --video
# After tweaking in Onshape and downloading a new revision:
aerial cad-drop downloads/goat-v2 --config examples/cad-experiment.json --count 100 --seed 42 --out runs/goat-v2-score
```

`cad-drop` records a physical drop even when the design cannot fully seat. The
supplied export's 38 mm pegs bottom out in its 34 mm sockets. Mass is explicitly
provisional unless you supply `--mass-grams`. See the guide for scoring and ranking.

The general strict export route is [onshape-to-robot → MuJoCo import](docs/onshape-to-robot.md).
It imports geometry, CAD mass properties and named reference sites into the existing
experiment workflow. Start with `aerial inspect-mjcf export_example/robot.xml`;
the supplied example needs rigid assembly mates, real mass properties, convex
collision decomposition and finite colors before preparation. The guide includes
an export configuration and seating metadata template.

For the alternative manual STL route, follow [the geometry handoff guide](docs/geometry.md). Importing a single visual STL as collision geometry is deliberately rejected if it is concave. Geometry preparation never silently fills a socket by taking the whole-block hull.

```bash
aerial prepare path/to/cad-manifest.json --out assets/goat.json
aerial validate assets/goat.json --out runs/goat-validation.json
aerial drop assets/goat.json --out runs/goat-aligned
```

The [optional Onshape adapter](docs/onshape.md) reads variables, exports named parts at a pinned microversion, caches exports, and can explicitly update independent variables in an experiment workspace. Its transport and preservation behavior have mocked tests; it has **not** been exercised against your account. It does not infer your target pose or mass properties from screenshots. CAD candidate generation remains staged so those checks cannot be bypassed by an optimizer.

## Assumptions and limits

The first implementation assumes rigid blocks, two circular blind sockets with circular tapered lead-ins, and explicit leg surface probes. General convex collision geometry is supported, but noncircular sockets need a corresponding feature evaluator. For each seating face provide at least three noncollinear point pairs; more points catch rocking better.

Printed material, infill, clearances, friction, and rebound must be measured. The demo's effective density (600 kg/m³) and friction (0.3) are placeholders. The importer requires explicit mass, COM and COM inertia; overlapping collision solids never determine them. Geometry probes check representative points, not a proof of complete CAD-to-collider equivalence. Visual section inspection is still required.

This tests pairwise local assembly only. There is no drone controller, aerodynamic/downwash model, structural analysis of arches, deformation, breakage, or claim that a simulated capture probability transfers to hardware.

## Documentation consulted

- [MuJoCo collision detection](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection)
- [MuJoCo contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
- [MuJoCo meshes](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh) and [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
- [Onshape exports](https://onshape-public.github.io/docs/api-adv/translation/), [configurations](https://onshape-public.github.io/docs/api-adv/configs/), [authentication](https://onshape-public.github.io/docs/auth/apikeys/), and [API limits](https://onshape-public.github.io/docs/auth/limits/)
