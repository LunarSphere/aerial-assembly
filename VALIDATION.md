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
