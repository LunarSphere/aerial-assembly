# Brixzle

A research pipeline for repeatable PLA bricks: CadQuery geometry, voxel connectivity and insertion planning, MuJoCo contact trials, and NSGA-III optimization. One candidate brick geometry is reused throughout each structure. Designs have top/bottom tapered peg sockets, side retaining rails, or both; the body contains pickup bearings and optional I-beam-like ribs.

**Floating placement is the default.** The unmodeled transport step initializes a free brick above its connector with the sampled pose error. A compliant virtual carrier lowers it toward the seat, then all carrier forces are removed. Gravity and physical contact determine final seating and stability. No force or weld holds a released brick, and previously placed bricks are never reset. Pickup performance and transport time are excluded in this mode. The passive two-tine fork remains available with `--mode fork`.

## Setup

Python 3.12 is recommended. Core dependencies are pinned in `pyproject.toml`; the complete resolved environment is in `requirements.lock`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

The lock targets Python 3.12 on Linux and includes pytest. For other supported Python/platform combinations, install `pip install -e '.[dev]'` to resolve the pinned core dependencies anew. Everything runs on CPUs without a display or GPU.

## First experiments

```bash
brixzle defaults --out runs/defaults
brixzle generate --out runs/brick
brixzle plan --out runs/plans
brixzle trial --nominal --benchmark tower --out runs/tower
brixzle trial --benchmark cantilever --seed 1234 --out runs/cantilever
brixzle evaluate --trials 8 --out runs/baseline
brixzle report runs/baseline
```

Generated designs include STEP, STL, mass/center-of-mass/inertia, connector frames, pickup geometry, and convex collision pieces. The CAD and collision representations share a polyhedral CSG recipe; their volumes must agree. The simulator uses explicit CAD inertia rather than accumulating inferred hull masses. Units are millimeters in CAD and metadata, meters/kilograms/seconds in MuJoCo.

Trial folders contain their configuration, disturbance samples, failure reason, intermediate placement errors, an MJCF scene, and the final physical state. Floating scenes exclude future staging placeholders from contact until introduction; final state files also record the collision masks and gravity compensation needed for an exact state restore. Every introduced brick has gravity compensation disabled. `--nominal` fixes friction at the profile midpoint and disables pose, dimensional, tracking, and mass perturbations. It retains all contact dynamics and stability checks.

Use `--parameters runs/defaults/parameters.json` and `--config runs/defaults/config.json` to customize experiments. A configuration JSON may contain just the fields being overridden; unknown fields are rejected. `--target` accepts voxel target JSON as exported by `defaults`. Voxels have unique IDs, integer `[x,y,z]` cells, quarter-turn yaw rotations, and optional anchored fixtures at `z=0`.

## Optimization

Run a short end-to-end check before spending time on a search:

```bash
brixzle optimize --config examples/smoke.json --families combined --reinforcements ibeam --out runs/my-smoke
brixzle optimize --config examples/smoke.json --families combined --reinforcements ibeam --out runs/my-smoke --resume
brixzle optimize --config examples/pilot.json --out runs/pilot
```

The full defaults use population 32, 20 generations, and eight seeded trials per mandatory benchmark, **for each geometry/reinforcement campaign**. Shortlist ten candidates, reevaluate with 50 new trials, and validate three finalists with 200 held-out trials per benchmark. Run time depends on CAD validity, contact count, and cores; run the pilot first. `--search-only` defers candidate validation; resume without this flag to perform it later. `--workers` parallelizes candidate evaluations in separate CPU processes. Successful and failed evaluations are cached. Resume requires the same configuration, source code, and dependency versions.

NSGA-III minimizes five separate objectives: worst benchmark failure probability, worst benchmark peak-force p95, mean capped trial duration, brick mass, and mean unintended collision events. Failed trials receive the full benchmark timeout in the duration objective. Expected guide and connector contacts are excluded from collision events. Force is the sum of contact-force magnitudes per body pair, and includes controller reaction force in fork mode. It is a simulation load metric, not a certified material stress or drone thrust measurement.

Geometry/printing violations, unreachable mandatory assembly graphs, and numerical warnings are constraints. Assembly planning uses deterministic backtracking with a 20,000-state budget; exhausting the budget is reported separately from incompatible connectivity. Continuous genes cannot add connector types: peg-only bricks cannot form these overhangs, and rail-only bricks cannot form the tower. Their baseline failures are saved and those campaigns are skipped. Combined candidates proceed to optimization.

Pareto ranking precedes optional normalized weighted preferences. Search, reevaluation, validation, and diagnostic seeds are separate. Candidates use common disturbance seeds for comparable experiments. Each completed generation stores population results and an atomic checkpoint, including RNG states. Only load checkpoints created locally by this pipeline; Python pickle can execute code.

The report contains an offline HTML table and mass/success chart, sliders for objective weights, per-design failure details, JSON, Markdown, and standalone SVG/PNG plots. The result JSON never claims the target was met unless a held-out finalist passes every mandatory benchmark's confidence bound and the numerical sensitivity check.

## Benchmarks and acceptance

Floating mode evaluates an eight-brick tower, three unsupported sequential extensions from one anchored brick, and a bridge across a three-voxel gap between two anchored fixtures. Fork mode additionally evaluates passive pickup, transport, and release onto a table. `--diagnostics` adds stepped voxel arch and dome trials; `--benchmark arch` selects a single diagnostic trial.

Overhang fixtures stand 40 mm above the floor. There is no invisible floor beneath the extensions. Connectivity is a planning hypothesis: all movable bricks remain free bodies, with no artificial connector welds. Every placed brick and all prior placements must stay within 0.5 mm and 1° of their lattice poses throughout five seconds after a one-second settling interval.

The reliability target is a one-sided 95% exact binomial lower bound of at least 95% **for each complete mandatory benchmark**, not per placement. For example, 59 successes in 59 trials meet that bound; a smoke run cannot establish it. Finalists undergo paired trials at half the timestep and twice the solver iterations. Success counts must agree; p95 contact forces must agree within 25% or 0.05 N, whichever is larger. Report numerical failures instead of treating unstable contacts as success.

For floating trials with side rails, the controlled approach begins above the complete rail opening. The virtual carrier uses configurable force/torque limits and compliant position/orientation gains. It aims at the perturbed pose, so it does not correct the sampled estimation error. Its forces are removed at the configured 0.5 mm nominal release gap; gravity and contact then provide the final alignment. Time includes this approach, release, seating, and stability observation; it excludes drone flight, camera processing, and transport. Floating and fork results therefore have different time scopes, recorded in every evaluation.

## Fabrication and physical calibration

Provisional defaults: PLA density 1.24 g/cm³, 0.4 mm nozzle, 0.2 mm layers, 1.2 mm minimum walls/ribs, and solid walls with explicit cavities. Bricks must weigh 1–25 g; a 10 g provisional fork plus mass uncertainty must remain below a 40 g payload. Exported geometry is intended for upright printing; wall, bridge-span, ramp, and retaining-feature checks are preliminary filters, not a slicer certificate.

The fork has fixed upturned tips to retain the payload without a grasp actuator. Set `fork.lip_height_mm` to zero for straight tines. Channel height includes lip withdrawal clearance. Fork dimensions, mass, controller gains, and force/torque limits are configurable; the defaults do not represent measured drone hardware.

Print coupons for clearance and surface friction, then measure final brick mass, fork capture range, seating range, and overhang deflection. Update the JSON profiles and mark them calibrated only after measurement. The material beam/shear screen is an approximate rejection filter; MuJoCo trials treat bricks as rigid and cannot establish PLA fracture, creep, buckling, or layer adhesion. Validate finalist bending and connector strength on the bench.

Full drone dynamics, global camera perception/occlusion, basestation latency, rotor wash, and flight-control integration are subsequent work. Geometry metadata and seeded pose-error inputs provide their integration points. A voxel representation does not guarantee every bridge, arch, or dome can be assembled; incompatible connectivity, access, and intermediate stability are explicit failures.

## Development checks

```bash
pytest -q
```

Checks cover CAD/contact agreement, preserved openings, connected exports, mass limits, lattice compatibility, blocked fork access, exact reliability bounds, common/held-out seeds, passive pickup without attachments, matching connector clearance, floating gravity placement, and optimization/checkpoint integration.

Implementation references: [CadQuery exports](https://cadquery.readthedocs.io/en/latest/importexport.html), [MuJoCo contact and collision modeling](https://mujoco.readthedocs.io/en/stable/computation/index.html), and [pymoo NSGA-III](https://pymoo.org/algorithms/moo/nsga3.html).
