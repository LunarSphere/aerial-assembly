# Local export → drop → inspect

Run commands from the `simulation/` directory with its virtual environment
active. A local input directory must contain `robot.xml` and every mesh it
references. The supplied two-peg model is in `two_peg_block/`.

## Run and view

```bash
aerial cad-drop two_peg_block --config experiment_configs/cad-experiment-fast.json --out runs/drop --video
aerial replay runs/drop
aerial replay runs/drop --collisions
aerial render runs/drop --out runs/drop/another-view.mp4
```

Each output directory must be new. Video support is installed with
`python -m pip install -e '.[video]'`. Rendering depends on an available
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
settings are numerical debugging values, not calibrated material properties;
retain timestep refinement and collision checks when changing them.
