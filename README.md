# Local CAD drop

Prepare a downloaded GOAT block, fix one copy in place, and drop a second rigid
copy onto it in MuJoCo. Everything runs locally: this package has no Onshape API
client or CAD editing commands.

Both the original three-part export and the single-piece `goat_mk2` profile are
supported. Single-piece geometry is partitioned locally with TetGen, preserving
the mesh surfaces and cavities; the original mesh supplies mass and visuals.

## Install

Use Python 3.11+ from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,video]'
```

The video extra is optional; use `.[test]` for simulation and interactive replay.
For pinned dependencies, install `requirements.lock` (or
`requirements-cad.lock` including video), then `pip install -e . --no-deps`.
The lockfiles originated on Linux; wheel availability depends on the platform.

## Drop a local export

Keep `robot.xml` and its complete `assets/` directory together.

**This checkout is missing `export_example/assets/`. Restore the original
matching export assets before running this example.** A different revision's
meshes are not interchangeable.

```bash
aerial cad-drop export_example --out runs/drop --video
aerial replay runs/drop
aerial replay runs/drop --collisions
aerial render runs/drop --out runs/drop/replay.mp4
```

For the local single-piece export, use
`aerial cad-drop goat_mk2 --out runs/mk2 --video`. On macOS, prefix this with
`MUJOCO_GL=cgl` for video rendering. Keep STL simplification disabled.

Omit `--video` to record just the simulation. Each drop needs a new output
directory. Replay requires a graphical display; rendering uses the saved
trajectory and never reruns the physics. On macOS, interactive replay may require
the MuJoCo launcher: `mjpython -m aerial_assembly.cli replay runs/drop`.

The default is one aligned, one-second drop from rest with 10 mm additional
vertical clearance above separated block bounding boxes. Adjust physics, duration,
or release conditions using:

```bash
aerial cad-drop export_example --config examples/cad-experiment.json --out runs/custom
```

Lengths use meters, velocities use SI units, CLI configuration release angles use
degrees, and quaternions use `[w, x, y, z]`. Mass defaults to provisional uniform
density of 600 kg/m³; supply `--mass-grams VALUE` for measured total mass.

## Results and limits

Each run saves the scene, geometry, configuration, geometry checks, source/code
hashes, a concise `summary.json`, and `trial_00000/result.json` plus
`trajectory.npz`. Video also produces start/end PNGs. A rendering failure leaves
the simulation recording intact.

Exit status 0 means a valid simulation, including a physical failure to assemble.
Status 2 signals configuration, collision-validation, numerical, or rendering
errors. An infeasible seating target still gets a diagnostic drop.

The supplied baseline was previously measured with 38 mm pegs and 34 mm sockets:
it bottoms out roughly 4 mm above flush seating. This workflow preserves that
geometry. Physics is uncalibrated; one simulated drop is not a hardware success
rate. See [the local workflow](docs/cad-iteration.md) and
[validation evidence](VALIDATION.md).

## Development

Source is in `src/aerial_assembly/`; tests are in `tests/`, including synthetic
fixtures used only for regression checks.

```bash
python -m pytest -q
```

The complete suite requires both local CAD exports and fails explicitly if
they are missing. Without the original export, use
`python -m pytest -q -m 'not requires_export'`; this includes the `goat_mk2`
integration test. For synthetic tests only, use
`python -m pytest -q -m 'not requires_export and not requires_mk2'`.

The supported commands are `cad-drop`, `replay`, and `render`. API access,
batch experiments, ranking, sweeps, demo generation, and alternate import
commands have been removed. Configuration accepts only `physics`, `trial`,
and `release`. Old experiment APIs and summary statistics are not supported;
existing saved trajectories remain viewable.
