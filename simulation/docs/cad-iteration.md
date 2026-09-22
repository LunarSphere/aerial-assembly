# Local export → drop → inspect

Run the commands below from the repository's `simulation/` directory with its
virtual environment activated.

## Prepare the input

Use a local export directory containing `robot.xml` and every referenced asset.
The original GOAT profile supports three unmerged visual solids: two vertical
circular pegs and an extruded body with two blind tapered sockets. The
single-piece profile supports the two pointed vertical pegs and circular blind
sockets in `goat_mk2`, including its hollow pegs and thin supporting walls. Keep
`merge_stls: false` and mesh simplification disabled when exporting.
Unsupported topology fails explicitly. This is not a general CAD importer.

The supplied `export_example` currently lacks its `assets/` directory. Restore
the matching original files; do not mix exports from different revisions.
Preparation reads XML and meshes, never executes exporter pickle files, and
leaves the input export untouched.

## Run and view

```bash
aerial cad-drop export_example --out runs/drop --video
aerial replay runs/drop
aerial replay runs/drop --collisions
aerial render runs/drop --out runs/drop/another-view.mp4
```

Each run directory must be new. Video plays at half speed, holds the final frame,
and saves start/end PNGs. Install video support with
`python -m pip install -e '.[video]'`.

The existing renderer defaults to EGL. Select an available backend using
`MUJOCO_GL` where needed; graphical support depends on your platform. Interactive
replay on macOS uses `mjpython -m aerial_assembly.cli replay runs/drop`.
Rendering failure does not discard recorded results.

Use `--config examples/cad-experiment.json` to customize the single drop. Supported
sections are `physics`, `trial`, and `release`. An omitted trial duration/dwell
defaults to 1 second/0.1 seconds. The release defaults to aligned, from rest,
with 10 mm additional clearance above separated world-axis bounding boxes.
Offsets, angles, and initial velocities can be specified for that one release.
There is no random sampling or ranking.

## What preparation preserves

- Combines exported parts in their assembly positions into one rigid block.
- Remeasures peg/socket dimensions, seating points, and the target transform.
- Partitions collision solids while retaining socket openings, blind floors,
  ramps, and guide lips. Original visual surfaces remain unchanged.
- Reuses cached partitions only for matching geometry and preparation code;
  cached contents are checked by hash.
- Computes mass, COM, and inertia from original solids, ignoring placeholder
  exported inertias. Default density is provisional 600 kg/m³.
  `--mass-grams VALUE` supplies total mass but still assumes uniform distribution.

Single-piece exports use constrained tetrahedralization followed by convex
merging, with volume checks before and after merging. Collision vertices are
snapped to a 0.1 µm grid to regularize STL noise (at most 0.087 µm displacement);
visuals stay unchanged. The cavities are retained. Feature recognition follows inward-facing socket
surfaces so a separate hollow peg interior cannot be mistaken for socket depth.
Seating points come from matching planar surfaces. Piecewise socket radii are
used for insertion scoring and material probes. This conversion can produce
thousands of collision pieces and takes longer to simulate than the original
extruded-body profile; prepared pieces are cached between runs.

## Inspect the result

Read `summary.json` for the outcome, numerical validity, seating gap,
penetration, mass assumption, and geometry feasibility. Detailed diagnostics
are in `geometry_validation.json` and `trial_00000/result.json`.
The recorded trajectory, scene, geometry, configuration, and source/code hashes
remain alongside them.

The original baseline's 38 mm pegs bottom out in 34 mm sockets. Preparation
reports this and records a diagnostic drop without changing the target or
geometry. Valid physical failures return exit status 0; numerical errors and
unverified collision geometry return 2. A geometry-rejected run cannot report
successful seating.

The supplied `goat_mk2` has approximately 25.33 mm pegs and 22.67 mm deep receiving
sockets, giving 2.67 mm of nominal axial bottoming. The deeper hollow spaces in
the pegs are separate cavities. The simulator records this geometry as supplied.

Physics settings are numerical debugging defaults, not calibrated material
properties. Retain timestep-refinement and collision checks when changing the
model; do not interpret one drop as a capture probability.
