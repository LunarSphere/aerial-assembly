# Download, drop, tweak, repeat

The GOAT adapter prepares the downloaded three-part block locally. You do not
need to fix the assembly's exported free joints or rewrite seating metadata
between design iterations. Keep each download in its own directory with
`robot.xml` and its `assets/` folder intact.

## Watch a drop of your actual export

```bash
source .venv/bin/activate
aerial cad-drop export_example --out runs/goat-v1 --video
```

This prepares one rigid block, duplicates it into fixed/lower and falling/upper
bodies, records a one-second aligned drop and writes `runs/goat-v1/drop.mp4`.
The video plays at half speed and holds the last frame for inspection. It also
writes start/end PNGs. For an interactive viewer on a machine with a display:

```bash
aerial replay runs/goat-v1
aerial replay runs/goat-v1 --collisions
```

Video can be rendered later without repeating the physics:

```bash
aerial render runs/goat-v1 --out runs/goat-v1/replay.mp4
```

Install additional dependencies in a fresh environment with
`python -m pip install -e '.[cad,video]'`. Headless rendering defaults to EGL;
the tested versions are also available via `pip install -r requirements-cad.lock`.
set `MUJOCO_GL=osmesa` where appropriate if EGL is unavailable. A video-rendering
failure does not discard the recorded simulation.

## Compare Onshape revisions

1. Download/export the current CAD into a new directory.
2. Run an aligned `cad-drop` to check engagement and the video.
3. Evaluate randomized drops with the same configuration, seed and count:

```bash
aerial cad-drop downloads/goat-v1 --config examples/cad-experiment.json \
  --count 100 --seed 42 --out runs/goat-v1-score
```

4. Change parameters in Onshape, export into `downloads/goat-v2`, then run:

```bash
aerial cad-drop downloads/goat-v2 --config examples/cad-experiment.json \
  --count 100 --seed 42 --out runs/goat-v2-score
aerial cad-rank runs/goat-v1-score runs/goat-v2-score
```

The objective is successful-seating fraction, maximized; lower mean settling
time breaks ties. An aligned preview is not eligible for ranking. The ranking
command rejects comparisons with different release samples, physics, mass
policy or preparation code. Change only CAD design variables during a search.
After choosing finalists, use independent seeds, more trials and timestep
refinement. A finite search improves the measured objective; it cannot certify
a global maximum.

The Onshape tweaking/downloading step is manual in this workflow. No live CAD
variables are changed by `cad-drop`; it consumes each downloaded revision.

## What is automated

- Combines the three exported solids in their assembly positions into one rigid
  block, eliminating the unintended independently falling legs.
- Measures the two vertical pegs and each socket's floor, throat and mouth
  rings from the current export. The target transform, dimensions, clearance,
  seating points and surface probes are regenerated for every revision.
- Builds collision pieces from the body's extruded side profile and socket
  frusta. The guide lips, ramps, blind floors and circular facets are retained.
  The body volume is checked against the original mesh and socket probes verify
  empty space, wall material and floor material. Floating-point ring noise is
  regularized within 0.2 micrometers; collision vertices snap to 0.1 micrometers.
  Original visual surfaces stay unchanged. Cached partitions are hashed and
  reused only for the same geometry, socket features and preparation code.
- Replaces NaN export colors with the experiment's lower/upper colors.
- Computes mass, COM and inertia from the original watertight solids, never
  from overlapping or approximated collision pieces. Default effective density
  is **provisional 600 kg/m³**. Add `--mass-grams VALUE` to supply measured total
  mass; its spatial distribution is still assumed uniform. All results retain
  this mass assumption. The adapter deliberately does not trust the supplied
  export's `1e-9` placeholder inertias.
- Saves source-file hashes, available CAD microversions, inferred geometry,
  physics, releases, code hashes, collision checks, numerical validity, score,
  and the first trial's trajectory. Source downloads remain untouched.

This profile supports the current topology: three unmerged visual solids, two
vertical circular pegs, and an extruded body with two blind tapered sockets.
Changing ramp/lead-in angles, compatible lengths or clearances is supported by
remeasurement. Changing the topology or orientation fails explicitly rather
than silently reusing dimensions. Export with STL simplification disabled and
keep visual parts unmerged (`merge_stls: false`). The local profile prepares its
own colliders, so exporter-side convex decomposition is unnecessary.

## The supplied baseline

The measured peg length is 38 mm; the socket depth is 34 mm. At the intended
flush pose the tips penetrate the blind floors by about 4 mm. The workflow
reports this infeasible design and still records a physical drop. It does not
shorten the pegs, deepen the sockets, soften contacts to fake insertion, or move
the success target to the jammed pose.

An interfering target receives objective zero only when collision probes and
numerical checks pass. Collider/numerical failures have a null objective and
are excluded. For a geometrically rejected design, one diagnostic drop is
recorded and the remaining random trials are skipped; the zero comes from the
infeasible target, not a claimed 100-trial measurement. A failed aligned path
check without proven target interference has no score. Check
`geometry_validation.json` and `summary.json`. To remove
the baseline bottoming conflict, adjust peg length or socket depth in Onshape
while retaining adequate floor thickness, then download and repeat.

These outputs use a separate CAD-workflow bundle in each run. For the stricter
general importer, which requires explicit metadata and valid CAD inertias,
see [onshape-to-robot.md](onshape-to-robot.md).
