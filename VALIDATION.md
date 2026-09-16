# Validation evidence

All geometry used for this evidence is synthetic; none of these results evaluates the actual Onshape design or a physical printed block.

## Numerical findings

| Experiment | Outcome |
|---|---|
| Original proposed dt=250 µs, contact time constant=5 ms | About 2 mm receiver penetration; rejected |
| Aligned, 3 seconds, dt=25 µs, contact time constant=50 µs | Success; peak penetration 2.92 µm |
| Aligned, 3 seconds, final default dt=12.5 µs, contact time constant=50 µs | Success; peak penetration 20.28 µm; settles at approximately 0.154 s |
| Aligned, dt=12.5 µs versus 6.25 µs | Both succeed; peak penetration approximately 20.3 and 22.4 µm |
| 1 mm X offset, 0.6 seconds, dt=25 µs versus 12.5 µs | Timeout versus success: coarse convergence check failed |
| 1 mm X offset, 0.6 seconds, dt=12.5 µs versus 6.25 µs | Both succeed; final position error approximately 0.199 and 0.079 mm |
| Deliberate miss, dt=25 µs versus 12.5 µs | Both classified as missed receiver |
| Three narrow-envelope random trials, dt=25 µs | Three valid successes; pipeline smoke test only |
| Two synthetic lead-angle candidates, one shared narrow-envelope sample | Both valid; ranking pipeline completed, no statistical design conclusion |

The supported conclusion is that the baseline pipeline works and finer resolution was necessary for the tested offset. This is not a full capture-region or mesh-convergence validation. Collision-resolution sweeps and actual-material calibration remain gates before interpreting geometry optimization.

The local `runs/` directory contains raw JSON, trajectories where requested, and the intentionally preserved unsuccessful coarser convergence report. **`runs/default-final` is the three-second run with final default settings.** `runs/aligned-validated` is the earlier three-second 25 µs result. `runs/convergence-validated` is a **failed** 25→12.5 µs diagnostic; its filename does not imply a pass. `runs/refinement-offset-600ms.json` records the successful 6.25 µs offset test. Subsequent runs record source-file hashes as well as package versions.

## Automated checks

`python -m pytest -q`: **23 passed** with the final default timestep. `python -m pip check`: no broken requirements.

Tests cover free fall, deterministic replay, initial-state velocity/frame conversion, stable seating, aligned timestep refinement, deliberate misses, intersecting starts, final-dwell enforcement, quaternion sign invariance, closed and missing socket material, concave-mesh rejection, STL unit conversion, geometry hash verification, seeded uncertainty, confidence intervals, candidate rejection, preservation of derived Onshape expressions, redirect signing, credential-host isolation, API quotas, and rate-limit retry.

Headless simulation and CLI commands have been exercised. The graphical replay viewer is provided but was not launched here. Onshape behavior is tested with mocked responses, not against a live account.

## Onshape-to-robot importer

The existing 23 tests and seven importer tests pass. The full suite passed with
the initial five importer tests (28 total); the final expanded importer suite
passed separately (seven tests).

Importer checks cover a rotated/translated export passing socket and insertion
validation, rotated rigid-child inertia aggregation, scaled external STL assets,
site references, export diagnostics, incorrect bounds, and overwrite protection.

`aerial inspect-mjcf export_example/robot.xml --out runs/export-inspection.json`
exits 2 as expected: three floating roots, placeholder mass properties,
nonfinite colors, and a concave main-body collider. This is inspection of the
actual supplied export, not validation of its assembly performance. The local
drop of the prepared model is documented below; no live CAD export was performed.

## Automated GOAT download workflow

The local `cad-drop` adapter now prepares and simulates the actual supplied
export. Evidence is in `runs/goat-export-first/`, with the fully framed video
at `drop-framed.mp4`. The one-second aligned drop is numerically valid:

- Final seating gap: approximately 4.007 mm; outcome `stationary_misalignment`.
- Maximum contact penetration: approximately 21.74 micrometers, within the
  approximately 25 micrometer budget.
- All 132 socket probes pass. At the intended target, the 38 mm pegs intersect
  the floors of the 34 mm sockets by approximately 4 mm.
- The three rigidly combined source solids have provisional mass 113.894 g
  under a uniform effective density of 600 kg/m³. This is not a measured mass.
- The original visual meshes are retained. Collision geometry has 693 convex
  pieces; it is derived from the side profile and socket rings, checked for
  volume preservation, and does not fill the sockets.

This single aligned run is a preview, not a sampled success-rate estimate.
The preparation and simulation run locally; no Onshape document was modified.
The full 35-test suite passed after adding the initial five CAD-workflow tests.
