# Local export → drop → inspect

Run commands from the `simulation/` directory with `uv`. A local input
directory must contain `robot.xml` and every mesh it
references. The supplied two-peg model is in `two_peg_block/`.

## Run and view

```bash
uv run --all-extras aerial cad-drop two_peg_block --config experiment_configs/cad-experiment-fast.json --out runs/drop --video
uv run --all-extras aerial replay runs/drop
uv run --all-extras aerial replay runs/drop --collisions
uv run --all-extras aerial render runs/drop --out runs/drop/another-view.mp4
```

Each output directory must be new. The `uv sync --all-extras` setup includes
video support. Rendering depends on an available
graphics backend; interactive replay on macOS may require
`mjpython -m aerial_assembly.cli replay runs/drop`.

The single-drop config accepts `physics`, `trial`, and `release` sections. An
omitted trial duration/dwell defaults to 1 second/0.1 seconds. The default
release is aligned and at rest, with 10 mm of additional clearance.

## Preparation behavior

- Combines the local export's rigid parts in their assembly positions.
- Measures peg/socket dimensions, seating points, and target transform.
- Partitions collision solids while preserving the original visual surfaces.
- Caches compatible collision partitions and checks cached contents by hash.
- Computes mass, center of mass, and inertia from the original solids.

The importer supports a single watertight solid with two vertical peg/socket
axes and the legacy three-solid form with two pegs and one body. It is not a
general CAD importer. Default effective density is provisional at 600 kg/m³;
`--mass-grams VALUE` supplies a measured total mass while assuming a uniform
distribution.

## Results and limits

Read `summary.json` for the outcome, numerical validity, seating gap,
penetration, mass assumption, and geometry feasibility. Detailed diagnostics
are in `geometry_validation.json` and `trial_00000/result.json`. The source
mesh, scene, settings, and source/code hashes are recorded with the run.

The supplied two-peg block's pegs bottom out before the flush target. Physics
settings are numerical debugging values, not calibrated material properties.
The current 27-state grid and the simulation tests use timestep `0.002` s and
contact time constant `0.005` s. When changing these settings, rerun the grid
and inspect its penetration and insertion results.
