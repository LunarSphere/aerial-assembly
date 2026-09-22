# Validation evidence

## Current verification

The single-drop example now explicitly uses the faster settings measured in
[MK2 timestep tuning](#mk2-timestep-tuning); the package defaults described next
are separate from that example.

The default timestep is now 0.1 seconds with a 0.2-second contact time constant
for faster, coarse previews. Historical seating and convergence evidence below
applies to the previous fine settings, which the accuracy tests now specify
explicitly (0.0000125-second timestep, 0.00005-second contact time constant).
Passing those tests does not validate collision accuracy at the new defaults.
The aligned synthetic 0.3-second drop at 0.1 seconds per step reported about
15.5 mm peak penetration; halving the step to 0.05 seconds (keeping the contact
time constant at 0.2 seconds) still reported 12 mm. Both runs were correctly
classified as `invalid` / `excessive_penetration`.

The retained automated checks cover gravity, deterministic trajectories,
release frame conversion, seating and final dwell, timestep refinement,
invalid starts, quaternion sign invariance, socket material checks, convex
partitioning, and local MJCF transforms/scaling. CLI tests cover single-trial
outputs, removed options, overwrite protection, diagnostic outcomes, and
preservation of recordings after rendering failure.

The cleanup was checked on macOS with Python 3.13.13 and MuJoCo 3.13.0:
the independent suite passed all 39 tests. Running the four export integration
tests separately produced four explicit setup errors for the missing assets.
Dependency compatibility and Python compilation checks passed. Video rendering
of a saved synthetic test trajectory produced an MP4 and start/end PNGs using
`MUJOCO_GL=cgl` outside the sandbox. Interactive replay was not launched.

Run `python -m pytest -q` for the full suite. Tests marked `requires_export`
require the original `export_example/assets/`, which is missing in this checkout.
They fail explicitly when those assets are absent. Use
`python -m pytest -q -m 'not requires_export'` to include the available `goat_mk2`
integration check while excluding the missing original export. Tests marked
`requires_mk2` require that local single-piece export; exclude both markers for
synthetic checks only.

After restoring the original assets, run the complete suite and a one-second
`aerial cad-drop export_example --out runs/baseline`. Compare the outcome,
seating gap, penetration, and trajectory against the original implementation
under the same dependency versions and settings. Check replay/video separately
on a supported graphics environment.

## MK2 timestep tuning

On 2026-09-18, five settings were measured against the same MK2 collision
geometry (`881dcc89f250` asset-hash prefix), with a 10 mm aligned release from
rest, 0.3-second duration, friction 0.3, damping ratio 1, and 100 solver
iterations. Trials are deterministic (no random sampling). The existing
`runs/drop3` was read for comparison and was not modified or rerun.

| Timestep (s) | Contact time constant (s) | Simulation wall time (s) | Peak penetration (mm) | Numerical result |
| --- | --- | --- | --- | --- |
| 0.001 | 0.002 | 6.6 | 0.938 | Invalid |
| 0.0002 | 0.0004 | 37.0 | 0.220 | Invalid |
| 0.0001 | 0.0002 | 65.7 | 0.527 | Invalid |
| 0.0001 | 0.0004 | 64.7 | 0.527 | Invalid |
| **0.00005** | **0.0001** | **122.8** | **0.0107** | **Valid** |

Timings measure `run_drop` on this Mac, excluding model compilation (about
2.1–2.3 seconds per candidate), geometry preparation/validation, and rendering.
They are individual measurements, not statistical speed benchmarks. All five
kept the upper block off the catch floor and produced no solver warnings, but
the four larger-step candidates exceeded the unchanged 0.05 mm penetration
limit. Trial configurations, trajectories, results, and timing summaries are
saved under `runs/timestep-tuning/` (Git-ignored).

The selected example setting uses 6,000 steps instead of `drop3`'s 24,000.
At 0.3 seconds, the block had a 2.687 mm seating gap, 0.0000073 m/s linear
speed, and 0.1266 N upward support against its 0.1274 N weight. Its
`stationary_misalignment` classification is expected for this bottoming CAD;
the geometry warning remains. No scoring thresholds or collision geometry
were relaxed.

A second run through the normal `cad-drop goat_mk2 --config
examples/cad-experiment.json` command reproduced the complete result JSON and
recorded trajectory exactly, returned exit code 0 and `valid: true`, and is
saved as `runs/timestep-tuning/selected`. This checks repeatability and the
normal CAD preparation/validation path as well as the benchmark path.

The saved finer `drop3` used a 0.0000125-second step and 0.00005-second contact
time constant. It had 0.0136 mm peak penetration and a 2.998 mm final gap.
Both support the block, but their resting poses differ: these measurements
establish usable aligned-drop behavior, not timestep convergence or validated
performance for offset/tilted releases. In particular, the non-monotonic
penetration results rule out assuming every intermediate timestep is safe.

## Single-piece GOAT Mk2

The local single-solid adapter measures approximately 25.33 mm peg length and
22.67 mm receiving socket depth. It follows the socket surface separately from
the deeper hollow peg interior. The nominal flush target therefore has 2.67 mm
of axial bottoming; the loader does not modify the part to make it seat.

The final constrained mesh partition has 2,770 convex pieces. Collision vertices
use a 0.1 µm grid; original visuals and mass properties remain unchanged. Volume
checks and all 132 socket-material probes pass. Static geometry checks use
collision queries without solving forces at deeply interpenetrating poses.
Large collision models receive a larger MuJoCo contact-memory arena.

Regression checks cover translation/scaling of measured features, rejection of
unsupported axes, preservation of hollow spaces and material, profile-based
bottoming detection, and the actual Mk2 export. The original timestep, friction,
and contact-response defaults are retained. This detailed collision partition
is slow during contact; short previews use an explicit shorter trial duration.

With the final adapter, `python -m pytest -q -m 'not requires_export'` passed
all 44 tests, including the Mk2 integration check. Four tests for the missing
original export were deselected.

The recorded 0.3-second preview is in `runs/goat-mk2-preview/`, using
`runs/goat-mk2-preview-config.json`. It completed with valid numerics and outcome
`stationary_misalignment`: maximum final seating gap 2.998 mm, peak penetration
13.61 µm against the 50 µm budget, and no solver warnings. Both peg tips reached
approximately 22.667 mm insertion. Final COM speed was 0.0057 mm/s. The mass
remains provisional at 12.982 g with uniform effective density 600 kg/m³.
This is an initial-settling preview, not a full one-second or convergence study.

## Grid-search verification

All 54 available tests passed across focused invocations, including the MK2
integration check; four original-export tests were excluded because their assets
are absent. The two reference tests also passed after the final scoring update.

The grid extension is covered by 10 tests for decimal inclusive ranges, six-axis
indexing, profile-center rotation and clearance, SI conversion, interruption and
resume, partial checkpoint recovery, output protection, outcome counts, geometry
gates, reference loading, reference dwell, and invalid-result precedence. Synthetic
grid tests exercise the real simulation and confirm one success and one miss in
a two-state grid. The existing CLI, physics (including deterministic trajectories
and timestep refinement), geometry, and MK2 integration checks are retained.

The saved MK2 preview's final pose passes the reference scorer's pose, stability,
support, and penetration thresholds. Replaying its 31 saved states through
MuJoCo diagnostics produces six consecutive accepted final frames, spanning
approximately 50 ms. This is a recorded-frame check, not a new simulation or proof
of the 100 ms every-timestep dwell. The current example grids use 0.3-second
trials and retain a 100 ms final dwell; they may time out before a drop settles.
No full MK2 grid or new MK2 tolerance envelope has
been measured as part of this implementation.

The original full example's dry run counted 301,401 states. The main example
and pilot have since been reduced to 27 states each (three X offsets, three Y
offsets, three theta-Y angles, and one Z height); both pass configuration-only
dry runs. See [grid-search conventions](docs/grid-search.md) for the
accepted-reference scoring definition and the current small grid.

## Historical numerical findings

These observations predate the simplification and are not fresh verification
of the incomplete export in this checkout.

Synthetic fixture:
- The original 250 µs timestep and 5 ms contact time constant allowed roughly
  2 mm of receiver penetration.
- The previous defaults were a 12.5 µs timestep and 50 µs contact time constant.
  A three-second aligned run seated with about 20.28 µm peak penetration.
- A 1 mm offset disagreed at 25 versus 12.5 µs; 12.5 versus 6.25 µs both seated.
  A coarse run alone is insufficient evidence of numerical convergence.

Original GOAT export:
- A one-second aligned drop was numerically valid with a final seating gap of
  approximately 4.007 mm and outcome `stationary_misalignment`.
- Peak penetration was approximately 21.74 µm within the roughly 25 µm budget.
- All 132 socket probes passed. The 38 mm pegs exceeded socket depth by 4 mm.
- Provisional mass was 113.894 g at uniform effective density 600 kg/m³.
- Collision preparation produced 693 convex pieces while preserving open sockets.

These are software and geometry checks, not evidence of calibrated physical
assembly performance. Printed dimensions, mass distribution, friction, and
rebound need measurement. Synthetic test geometry is not the actual CAD design.
