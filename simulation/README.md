# Local CAD drops and grid searches

Prepare a downloaded GOAT block, fix one copy in place, and drop a second rigid
copy onto it in MuJoCo. Everything runs locally: this package has no Onshape API
client or CAD editing commands.

Both the original three-part export and the single-piece `goat_mk2` profile are
supported. Single-piece geometry is partitioned locally with TetGen, preserving
the mesh surfaces and cavities; the original mesh supplies mass and visuals.

## 1. Install and prepare the files

Use Python 3.11+ from the `simulation/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,video]'
```

The video extra is optional; use `.[test]` for simulation and interactive replay.
For pinned dependencies, install `requirements.lock` (or
`requirements-cad.lock` including video), then `pip install -e . --no-deps`.
The lockfiles originated on Linux; wheel availability depends on the platform.

Run all commands below from `simulation/` with the virtual environment
activated. On a Windows PC, use Linux/WSL for the grid search: its current file
lock uses Unix-only `fcntl`, so native Windows Python cannot run `cad-grid`.

When moving to another computer, copy these alongside the repository:

- The complete `goat_mk2/` export, including `robot.xml`, `config.json`, and
  `assets/`. Keep the matching files together.
- `runs/goat-mk2-preview/` for the grid's accepted-reference scoring. It needs
  `summary.json`, `trial_00000/result.json`, and `trial_00000/trajectory.npz`.
- Optionally, `assets/cad-cache/` to reuse compatible prepared collision geometry.

`assets/` directories and `runs/` are Git-ignored: cloning the repository alone
does not transfer these files. Start a new experiment on the PC; resuming an
existing one requires matching its code, dependency versions, reference, and
configuration. The commands here use `goat_mk2`; the old `export_example` is not
available in this checkout.

Video rendering automatically selects CGL on **macOS**, EGL on Linux, and GLFW
on Windows. An explicit `MUJOCO_GL` environment variable overrides that default.
Linux needs a working OpenGL/EGL driver.
Interactive replay needs a graphical display. On macOS, replace `aerial replay`
with `mjpython -m aerial_assembly.cli replay` if the viewer requires that launcher.

## 2. Drop one block — 0.3-second simulation

This fixes the lower block and drops one aligned upper block from rest with
10 mm additional vertical clearance. It uses
[`examples/cad-experiment.json`](examples/cad-experiment.json), including your
**0.3-second duration** and **0.1-second final dwell**.

The single-drop example uses a **0.00005-second timestep** and a
**0.0001-second contact time constant**. This takes 6,000 steps per drop, four
times fewer than the previous fine settings. On the local MK2 aligned drop it
took about 123 seconds of simulation computation and kept peak penetration to
0.011 mm. Larger tested timesteps exceeded the 0.05 mm penetration limit;
see [validation evidence](VALIDATION.md#mk2-timestep-tuning).

The package defaults and grid examples still use a coarse 0.1-second timestep
and 0.2-second contact time constant; use the single-drop config above for the
tuned settings. Use a new output directory when changing physics settings.

### Run the drop and save a video

```bash
aerial cad-drop goat_mk2 --config examples/cad-experiment.json --out runs/drop --video
```

The MP4 is `runs/drop/drop.mp4`; `drop-start.png` and `drop-end.png` are saved
beside it. **Each simulation needs a new output directory.** For another run,
change `runs/drop` to a new name in the commands.

Omit `--video` to save only the simulation. Omitting `--config` uses the CLI's
**one-second default**, regardless of changes made to the example JSON files.

### Replay the saved drop

```bash
aerial replay runs/drop
aerial replay runs/drop --collisions
```

The second command displays the collision pieces instead of the visual mesh.

### Generate a video later

```bash
aerial render runs/drop --out runs/drop/replay.mp4
```

Replay and rendering use saved trajectories; they do not rerun the simulation.
Choose a new MP4 filename if the requested video already exists.

## 3. Run the grid-search experiment

The experiment config is
[`examples/mk2-grid-search.json`](examples/mk2-grid-search.json). It controls
X/Y offsets, vertical clearance, and three rotation angles. Each axis is
`[minimum, maximum, step]` with inclusive endpoints, using **millimeters and
degrees**. Every drop starts from rest. `theta_y_deg` tilts in the side profile;
the default grid pivot is the block's bounding-box center.

Both the main experiment and [`mk2-grid-pilot.json`](examples/mk2-grid-pilot.json)
now use the same small **27-state grid**, with **0.3 seconds per drop** and a
**0.1-second final dwell**:

| Axis | Values |
| --- | --- |
| X offset | −5, 0, +5 mm |
| Y offset | −5, 0, +5 mm |
| Z clearance | Fixed at 30 mm |
| Theta-Y (side-profile tilt) | −5, 0, +5 degrees |
| Theta-X / Theta-Z | Fixed at 0 degrees |

That is 3 × 3 × 1 × 1 × 3 × 1 = **27 drops**. This coarse grid is for initially
checking the simulation workflow; expand or refine its ranges after inspecting
the results. The single-drop example also remains at 0.3 seconds.
Simulated duration is not wall-clock runtime. The detailed collision model can
still take much longer, and grid states currently run sequentially.

Z is additional clearance above separated block bounding boxes and must be
nonnegative. Negative X/Y offsets and rotation angles are supported. A fixed
axis uses equal minimum/maximum values, such as `"dz_mm": [30, 30, 1]`.

### Check the configuration and count states

```bash
aerial cad-grid goat_mk2 --config examples/mk2-grid-search.json --out runs/mk2-grid --dry-run
```

This validates the ranges and prints the exact state count without running
physics or creating the output directory. It does not check the CAD export or
the reference recording. Correct any range error before proceeding.

### Run one grid state first and save its trajectory

```bash
aerial cad-grid goat_mk2 --config examples/mk2-grid-search.json --out runs/mk2-grid --max-states 1 --record-trial 0
```

This runs index 0: the first combination of range values, not necessarily the
aligned drop. `--max-states 1` limits this invocation to one state; it does not
reduce the experiment's total grid size. Use a new output directory for a new
experiment.

### Save a video of that grid state

```bash
aerial render runs/mk2-grid --trial 0 --out runs/mk2-grid/state-0.mp4
aerial replay runs/mk2-grid --trial 0
aerial replay runs/mk2-grid --trial 0 --collisions
```

Grid searches **do not generate videos automatically**. `--record-trial 0`
saves `trial_00000/trajectory.npz` for these commands. Repeat `--record-trial`
with other indices to retain selected trajectories, or use `--record-successes`
to retain successful ones. Recording must be requested before those states run;
it cannot be added retroactively to completed states.

### Continue the grid experiment

```bash
# Continue through every remaining state.
aerial cad-grid goat_mk2 --config examples/mk2-grid-search.json --out runs/mk2-grid --resume
```

Add `--max-states 10` to process at most ten additional states at a time.
Ctrl-C stops the run; completed states are checkpointed and `--resume` continues
from the next uncommitted state. Changing ranges, duration, scoring, physics,
geometry, code, or recorded dependency versions requires a new output directory.

To start the entire grid directly in a fresh directory, omit both `--resume`
and `--max-states`:

```bash
aerial cad-grid goat_mk2 --config examples/mk2-grid-search.json --out runs/mk2-grid-full --record-trial 0
```

### Find the results and understand success

| Output | Contents |
| --- | --- |
| `runs/drop/summary.json` | Single-drop outcome and final metrics |
| `runs/mk2-grid/grid_summary.json` | Successful, completed, total, and invalid state counts; success fractions; runtime estimate |
| `runs/mk2-grid/grid_results.jsonl` | One row per completed state, including its release pose, outcome, and final metrics |
| `runs/mk2-grid/grid_search.json` | Frozen grid, physics, scoring reference, versions, and code hashes |

Single-drop scoring checks flush seating. The MK2 pegs bottom out about 2.67 mm
before flush seating, so a settled drop can have status `stationary_misalignment`.
The grid instead uses the accepted MK2 preview's final pose as its explicit
reference. That recording must exist and match the geometry/mass. A grid success
must stay within the reference tolerances and satisfy stability/contact checks
for the final 0.1 seconds.

Shortening trials to 0.3 seconds can classify drops that are still settling as
unsuccessful. The original 0.3-second preview only demonstrated about 0.05 seconds
of acceptance at its recorded frames; accepting its final pose does not guarantee
that repeating it satisfies the 0.1-second dwell within 0.3 seconds. The reported
success fraction applies to the configured duration and grid. The total-grid
fraction is populated only when every state has completed.

See [grid-search details](docs/grid-search.md) for scoring thresholds, rotation
conventions, checkpoint recovery, and output interpretation.

## Units and validation limits

Lengths use meters, velocities use SI units, CLI configuration release angles use
degrees, and quaternions use `[w, x, y, z]`. The grid range fields explicitly use
millimeters/degrees; `trial` tolerance fields remain SI, including radians for
`angle_tolerance`. Mass defaults to provisional uniform density of 600 kg/m³;
supply `--mass-grams VALUE` for measured total mass. Reference-based grids need a
reference generated with the same mass.

Runs save shared scene/geometry and validation metadata. A single drop always
saves its trajectory; grids retain trajectories only when requested. A rendering
failure leaves the simulation recording intact.

Exit status 0 means a valid simulation, including a physical failure to assemble.
Status 2 signals configuration, collision-validation, numerical, or rendering
errors. An infeasible seating target still gets a diagnostic drop.

Physics is uncalibrated; a simulated grid success fraction is not a hardware success
rate. See [the local workflow](docs/cad-iteration.md) and
[validation evidence](VALIDATION.md).

## Development

Source is in `src/aerial_assembly/`; tests are in `tests/`, including synthetic
fixtures used only for regression checks.

```bash
python -m pytest -q
```

The complete suite requires both local CAD exports and fails explicitly if
they are missing. With only MK2 available, use
`python -m pytest -q -m 'not requires_export'`; this includes the `goat_mk2`
integration test. For synthetic tests only, use
`python -m pytest -q -m 'not requires_export and not requires_mk2'`.

The supported commands are `cad-drop`, `cad-grid`, `replay`, and `render`.
Single-drop configuration accepts `physics`, `trial`, and `release`; grid
configuration has its own schema. Older batch/ranking/API/import commands remain
removed; existing saved trajectories remain viewable.
