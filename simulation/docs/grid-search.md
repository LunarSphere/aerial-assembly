# Discrete CAD drop searches

Run the commands below from the repository's `simulation/` directory with its
virtual environment activated.

`cad-grid` enumerates the Cartesian product of six independent axes. Each
range is `[minimum, maximum, step]`, includes both endpoints, and requires a
positive step dividing the range exactly. A fixed axis uses identical endpoints.
Omitted translations/angles are zero except `dz_mm`, which defaults to 30 mm.
The order is X, Y, Z, theta-X, theta-Y, theta-Z, with theta-Z changing fastest.
Indices start at zero and remain stable across interrupted runs.

The grid uses millimeters and degrees, converting to SI at release. X/Y are
world-coordinate offsets from alignment. Z is **additional vertical clearance**
above separated bounding boxes after rotation, matching the single-drop height
convention. Thus Z=0 is the conservative hovering position; Z=30 adds 30 mm.
It is not an absolute world Z or clearance measured to the nearest cone.

Angles are right-handed, extrinsic world X/Y/Z Euler rotations, applied in that
order relative to the seated orientation. `theta_y_deg` tilts in the CAD's X/Z
side profile. `rotation_center: "bounds_center"` rotates about the center of the
local bounding box (side-profile center and mid-thickness); `"com"` uses the mass
center. Clearance is recomputed after rotation, so rotating can raise the pivot.
Every drop starts with zero linear and angular velocity.

## MK2 examples

```bash
# Count only: no CAD loading, output directory, or simulation.
aerial cad-grid goat_mk2 --config examples/mk2-grid-search.json --out runs/mk2-grid --dry-run

# Small 27-state pilot. Stop after three states to inspect timings/results.
aerial cad-grid goat_mk2 --config examples/mk2-grid-pilot.json --out runs/mk2-pilot --max-states 3 --record-trial 0

# Continue with at most three more states. Omit --max-states to finish the grid.
aerial cad-grid goat_mk2 --config examples/mk2-grid-pilot.json --out runs/mk2-pilot --resume --max-states 3

aerial replay runs/mk2-pilot --trial 0
aerial render runs/mk2-pilot --trial 0 --out runs/mk2-pilot/state-0.mp4
```

The main example and pilot now both use X/Y=-5..5 mm in 5 mm steps and
theta-Y=-5..5 degrees in 5-degree steps. Z is fixed at 30 mm; the other two
angles are zero. This is **27 states** (3 × 3 × 1 × 1 × 3 × 1), intended for
an initial simulation check. Run `--dry-run` after changing ranges to validate
and count them. All six axes, including a nonnegative Z range, are supported.
A coarse search or refinement of selected
regions does not establish the exact success fraction of the full fine grid.
Successful bounds are observed extrema, not a guarantee that everything inside
the resulting box succeeds.

Both examples use 0.3-second trials with a 0.1-second final dwell.
The CLI defaults to one second when `trial.duration` is omitted. Shorter trials
can turn late-settling cases into unsuccessful timeouts.

## Success definitions

The default `success: {"mode": "flush"}` uses the existing full-seating tests
and requires geometry validation to pass. MK2 cannot reach that flush target:
its pegs bottom out about 2.67 mm early. The user's accepted MK2 preview is
therefore configured explicitly as `success: {"mode": "reference",
"reference_run": "runs/goat-mk2-preview"}` in both examples. Reference paths
are relative to the working directory. This local recording must exist.

Reference scoring requires the same geometry and mass hash and a valid reference
recording with passing socket probes, no floor contact or solver warnings, and
acceptable penetration. Its final pose must satisfy the configured stability
and support thresholds. The final saved pose defines the target; the original
flush target and original classification remain in the metadata.

A new successful drop must remain within 0.5 mm COM displacement and 1 degree
orientation of that reference, with both insertion depths within 0.5 mm and
maximum seating gap no greater than the reference gap plus 0.1 mm. It must also
satisfy the existing speed, support, penetration, and floor-contact checks for
the final 0.1-second dwell. These thresholds are configured through the existing
`trial` fields (SI units, including radians for `angle_tolerance`). Reference
mode deliberately accepts the reference's incomplete flush insertion; it does
not require `legs_inside` from the unreachable original flush target.

The accepted reference final pose alone does not establish its historical dwell.
New drops are checked on every physics step for the full dwell at the end of the
configured duration. Early success does not end a trial. Numerical invalidity or
any floor contact prevents success. Timeouts remain unsuccessful under the chosen
duration and can be investigated with longer trials in a separate grid.

## Files, restart, and interpretation

Geometry preparation, model compilation, and validation are shared across states.
`grid_search.json` freezes all ranges, physics, settings, code hashes, dependency
versions, geometry hash, and the accepted reference. `grid_results.jsonl` stores
one durable row per completed state, including its actual initial pose, outcome,
final metrics, dwell, penetration, warnings, and runtime. `grid_summary.json`
reports successful/completed/total/invalid counts, outcome counts, observed success
bounds, and a runtime estimate. The estimate excludes preparation and can change
substantially when later states involve more contact.

`success_fraction_completed` counts successes over all completed states, including
invalid states in the denominator; invalids are also reported separately.
`success_fraction_total` is populated only after every requested state completes.
This is a fraction of a discrete simulated grid, not a hardware success probability.

New runs require a new directory. `--resume` validates the manifest and processes
only remaining states. `--max-states N` limits additional states per invocation.
Ctrl-C preserves committed states and refreshes the summary. After an abrupt
crash, an incomplete last JSONL line is backed up to a `.partial-*` file before
retrying that state. Corrupt complete rows are rejected. A lock prevents two
processes from writing the same run. Changes to geometry, scoring, physics, code,
or recorded versions require a new output directory.

Only compact results are retained by default. `--record-trial INDEX` (repeatable)
saves selected trajectories; `--record-successes` retains successful trajectories
from that invocation. Recording options may change on resume but cannot create
trajectories for already completed states. Saved recordings work with existing
`replay`/`render --trial INDEX` commands. Shared `scene.xml` and `geometry.json`
are stored once. A partial run is valid and resumable; exit 2 signals invalid
simulations or errors, and exit 130 indicates interruption.

The detailed MK2 collision geometry is expensive. Grid search does not coarsen
the collision geometry, increase the timestep, or infer outcomes for untested
states. Benchmark the pilot before committing to the full search. Physics remains
uncalibrated; compare timestep refinement and longer settling durations around
the measured success boundaries before making physical tolerance claims.
