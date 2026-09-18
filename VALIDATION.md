# Validation evidence

## Current verification

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
of the 100 ms every-timestep dwell. New example grids use one-second trials and
retain a 100 ms final dwell. No full MK2 grid or new MK2 tolerance envelope has
been measured as part of this implementation.

The full example's dry run counts 301,401 states (81 X offsets, 61 Y offsets,
61 theta-Y angles, one Z height). See [grid-search conventions](docs/grid-search.md)
for the accepted-reference scoring definition and the 27-state pilot.

## Historical numerical findings

These observations predate the simplification and are not fresh verification
of the incomplete export in this checkout.

Synthetic fixture:
- The original 250 µs timestep and 5 ms contact time constant allowed roughly
  2 mm of receiver penetration.
- The retained defaults are a 12.5 µs timestep and 50 µs contact time constant.
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
