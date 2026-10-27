# Brixzle: drop-assembled bricks

A parametric brick that a drone can drop *coarsely* into place and that
gravity slides home. It is built with CadQuery, tested in MuJoCo assembly
trials, and tuned with NSGA-III. Drone dynamics are not modelled: a brick is a
free body released at a sampled pose error above its lattice seat.

This is a separate uv project pinned to Python 3.12 because CadQuery has no
3.14 wheels.

```bash
cd brixzle
uv sync --all-extras
uv run pytest
```

## The V0 brick: the A/B pair (`configs/ab.json`)

`uv run brixzle export --params configs/ab.json --part A --out runs/v0-A` (and `--part B`) writes the
STL, STEP, a profile plot, and the rule margins.

The brick is a 2.5D "chevron": an XZ profile spanning two voxels
(2U × U × ~U, with U = 25 mm), swept through the depth. It comes as two parts
that alternate by course: **A** (even courses, 14.3 g of PLA at 100% infill)
and **B** (odd courses, 13.5 g). The A/B pair is the only design; the earlier
`alternate` (B = A rotated) and `uniform` lean modes were removed in #22 and
live on in git history (`5c15572`).

| Feature | Why |
|---|---|
| Top is a continuous asymmetric sawtooth: a 50° long ramp and a 52° steep face, peaks at voxel boundaries | The **hopper**. Every upward face slopes into a valley at a steeper angle than friction requires (atan 0.4 = 22° + 8° margin), so there are no flat landing spots. The catchment is the whole voxel. |
| Bottom keel matches the top of the course below | Courses nest as V in V, which gives precision from geometry rather than from the drone. |
| Tapered prismatic tooth at each keel point, leaning α = 38° and entering a slot cut along the steep face | **Fine seat plus hook.** The entry gap is 2.6 mm (taper + clearance) and the seated gap 0.6 mm. A slanted tooth cannot be lifted straight out, which resists tipping. |
| A has reversed teeth under normal slots, B normal teeth under reversed slots | Every tooth tip trails toward −x, so **every joint hooks against a +x overhang**. −x builds use the other part rotated 180°. |
| Y gutter: 30° sides with a ±3 mm flat centre band | Y capture of about ±9 mm, while straight Y tine bores still fit. |
| Two through-Y tine bores (2 mm rods), 24.75 mm apart, about 7.6 mm above the COM and straddling it, **the same in A and B** | **Passive fork.** One non-actuated fork carries either part level. |
| Lightening holes leave a 1.2 mm skin-and-rib frame (I-beam-like) | Reduces mass. |

### Placement-error model (coarse drone)

The default ("legacy guess", hand-set) aims at a point offset by a 3 mm bias toward the side the
tooth stroke comes from. Errors are x/y ~ N(0, 4 mm) clipped at ±12 mm, yaw σ 5°, roll/pitch σ 3°,
release 5–30 mm above the highest-reach height, and residual velocity σ 0.03 m/s.

**Measured X-Y error (Exp 3, #25).** `--approach-error configs/error-exp3-r4.json --error-stat mean|max|empirical`
replaces the X-Y part with the positioning error measured in `runs/exp3-r4` (2026-10-27, drone
marker, 24 approaches over 5 trials; `runs/` is git-ignored, so the values are copied into the
config by `approach_error.load`): mean 14.8 mm, RMSE 16.8 mm, p95 27.8 mm, max 38.5 mm. It is used
zero-mean: `mean` and `max` put every placement at that fixed radius in a random direction,
`empirical` resamples the 24 radii. The measured mean vector (−8.5, +3.9) mm is recorded but not
applied, since a sticker offset from the deck centre would show up as exactly such a bias. Yaw,
tilt, release height and velocity were not measured and keep the defaults above (labelled
`unmeasured` in every output's `config.error.source`). The optimizer takes
`eval.approach_error = {"config": ..., "statistic": ...}`; the required tolerance then sets the
`capture_x/y` rule targets and adds a simulated capture constraint (`capture_tolerance`), and
`fixture.fixture_rules(..., tolerance=)` adds the pickup funnel rule.

A/B V0 under each model (simulated, uncalibrated contact; `configs/ab.json`, drops n = 96 seed 0,
structures seeds 1–4; run directories `runs/ab-legacy-guess-*`, `runs/ab-exp3r4-mean-*`,
`runs/ab-exp3r4-max-*`):

| Trial | Legacy guess | Exp 3 mean (14.8 mm) | Exp 3 max (38.5 mm) |
|---|---|---|---|
| Drop A onto base, μ 0.2 / 0.35 / 0.5 | 92 / 79 / 82% | 19 / 15 / 11% | 0 / 0 / 0% |
| Drop B onto A, μ 0.2 / 0.35 / 0.5 | 91 / 82 / 77% | 20 / 17 / 12% | 0 / 0 / 0% |
| Wall 3×4, stable bricks per seed | 0, 4, 0, 3 of 10 | 0, 0, 1, 1 | 0 |
| Bridge, stable per seed | 0, 3, 0, 2 of 7 | 0, 0, 1, 1 | 0 |
| Tower 6, stable per seed | 0, 3, 0, 1 of 6 | 0, 0, 1, 1 | 0 |
| Ring 6×6×4, seated per seed (`--keep-going`) | 10, 9, 6, 13 of 32 | 1, 0, 1, 1 | 0 |

With the measured mean error, A/B misses its ±10 mm capture on most drops (jams, or falls off the
gutter in Y); with the max, nearly every drop falls. A/B fails the measured `capture_x/y` rules
(`capture_x` = 10 − 14.8 mm). That motivates the distributed-funnel brick (#42).

### V0 results (simulated, uncalibrated contact; not hardware rates)

Measured on this branch with `configs/ab.json`, seed 0 (run directories in `runs/`, local only):

| Trial | Result |
|---|---|
| X capture sweep (`drop`), A onto the base / B onto a seated A | Seats from −7.5 to +16.5 mm / −8.5 to +15.5 mm |
| Drop Monte Carlo, n = 96, μ = 0.2 / 0.35 / 0.5: A onto the base (`ab-only-drop-A-*`) | 92 / 79 / 82% seated; all failures jam |
| Same, B onto a seated A (`ab-only-drop-B-*`) | 92 / 84 / 71% seated (re-run 2026-10-27 on master and this branch, `runs/ab-legacy-guess-drop-B-*`: 91 / 82 / 77%) |
| Ideal assembly: running-bond wall 3×4, corbelled bridge with key (`ab-only-wall-ideal`, `ab-only-bridge-ideal`) | 10/10 and 7/7 stable. Max sag 0.48 mm |
| Single-brick staircase overhang, ideal aim, 12 requested (`ab-only-overhang±-ideal`) | 5 stable in both +x and −x; the 6th jams on a supporter that has sagged 5 mm |
| Ideal 6×6 ring, 4 courses (`ring-ab-ideal`) | 32/32 seated |

### What the sims taught (encoded as rules)

- **Clearance dominates jamming.** Jams come from two-point wedging of the
  tooth against the slot mouth while the keel rides the steep face. A tapered
  tooth gives a large entry gap and small seated play.
- **Seated play limits overhang**, but tight seats cost capture. That is a
  real trade-off, so it is left to the optimizer.
- **Teeth must lean against the build direction at every joint.** With
  B = A rotated (the removed `alternate` mode), every other joint's extraction
  direction lined up with the overhang rotation; the A/B pair hooks at every
  joint without the drop and wall losses of an all-same-lean brick (see the A/B
  section).
- Gutter angle, tine-bore placement, and centre-band width are coupled. A full
  V gutter leaves no room for straight tines above the COM.

## Commands

```bash
uv run brixzle rules  --params configs/ab.json --part B   # rule margins (the design's justification)
uv run brixzle export --params configs/ab.json --out runs/v0
uv run brixzle drop   --params configs/ab.json --samples 96 --out runs/v0-drop --video 2   # --part B: B onto a seated A
uv run brixzle assemble bridge --params configs/ab.json --ideal --video --out runs/v0-bridge
uv run brixzle assemble wall   --params configs/ab.json --seed 3 --out runs/v0-wall-noisy
uv run brixzle push --params configs/ab.json              # lateral push on an ideal tower
uv run brixzle optimize --config configs/nsga-ab.json --out runs/nsga
```

Structures: `tower`, `overhang+`, `overhang-`, `wall`, `bridge`. Every output
directory must be new.

## Pipeline

- `params.py`: `BrickParams`, `V0` (the A/B pair), NSGA bounds.
- `profile.py`: shapely XZ profile of parts A and B (sawtooth, slots, teeth, slanted features, shared
  tine bores, lightening, base).
- `cad.py`: CadQuery solid (Y-band sweep), STL/STEP, mass properties. Collision pieces are convex
  **by construction** (ear clipping + Hertel–Mehlhorn, then an affine sweep per Y band). There is no
  tetgen; the collision volume equals the outer profile volume exactly.
- `rules.py`: analytic rules used as NSGA constraints and as a pre-simulation filter.
- `lattice.py`: voxel placements, part (A/B) and orientation, `networkx` support graph, and the assembly-order
  planner. Supporters go first, and the approach path is swept against placed bricks, so courses fill
  against the tooth stroke and the key brick goes last. It also has a running-bond voxel tiler.
- `structures.py`: trial structures on anchored bases.
- `scene.py`: MJCF built once, with parked gravity-compensated bodies activated one at a time.
- `trials.py`: drop MC, capture sweeps, sequential assembly (seating judged relative to the actual
  supporter, collapse = motion during a stage, floor contact, or sag > H0/2), push test.
- `optimize.py`: pymoo NSGA-III. Objectives are −P(success), F_max, T, mass, and collisions; rules are
  constraints. The population is seeded around the base design (or named parents), and the final pick uses the
  weighted score w·(P, F, T, M, C) on normalised objectives. Each generation is checkpointed, with
  per-worker evaluation logs.

## Simulated drone pick-and-place (`fly-assemble`, Experiment 2)

A simulated Crazyflie with the passive fork picks bricks from a fixture, flies them and seats them
using the same `flight.mission` code as the real drone (see `flight/README.md`). Because A and B share
their tine bores, one fork carries both parts. The fixture holds each part in its own cradle:
slot kinds `A` and `B`, plus `Ar` / `Br` for cradles turned 180° for −x builds. The plan records each
brick's part, slot kind and its own loaded mass for the controller.

Results with the default registered, zero-bias Lighthouse model (simulated, uncalibrated contact and
aerodynamics; not hardware rates):

| Run | Result |
|---|---|
| `fly-assemble tower --bricks 1 --seeds 40` (`runs/ab-only-fly-single`) | 40/40 seated |
| `fly-assemble tower --bricks 2 --seeds 40`, an A then a B (`runs/ab-only-fly-two`) | 80/80 bricks seated, 40/40 towers |

The earlier single-brick figure (97.5%, Experiment 2 Checkpoint 2) was measured with the removed
`alternate` brick.

## Limits

- Friction, contact softness, and the error model are uncalibrated guesses.
- Structure trials stop at the first failed placement; real drones could retry.
- 2.5D only. The lattice and graph are 3D-ready, but crossing courses and domes are not implemented.
- Fork release (tine withdrawal) is checked geometrically only, not simulated.

## Why the A/B pair: pegs leaning against the build direction

Idea: keep the alternating sawtooth body, but use a distinct **type-B brick**
so every course's teeth lean opposite to the build direction. Every joint then
hooks against the overhang rotation. In the removed `alternate` mode B was just
A rotated, so only every other joint hooked.

- **A** (even courses): bottom T′ with *reversed* teeth, top T with normal slots.
- **B** (odd courses): bottom T with normal teeth, top T′ with *reversed* slots.
- A reversed interface mirrors the slot axis about vertical. It stays inside
  the nesting cone while α ≤ 90 − θ (the existing `nest_cone` rule).
- **−x builds use the same two parts rotated 180°.** `Placement(direction=-1)`
  handles this, and the base's slots follow the build direction.

Configs: `configs/ab.json` (V0) and `configs/ab-tight.json`
(clearance 0.05). Commands: `brixzle export --part B`, `brixzle drop --part B`
(drops B onto a seated A), and `brixzle assemble ... --track` (aim at the
supporter's measured pose, as a global camera would allow).

Single-brick staircase, 20 requested (simulated, uncalibrated contact; measured before #22, which
removed `alternate`/`uniform` and moved the tine bores so A and B share them):

| Mode | Fit (clearance) | Ideal aim | Camera-tracked aim |
|---|---|---|---|
| alternate | 0.3 | 4 | 4 (falls) |
| alternate | 0.05 | 4 | 6 |
| uniform | 0.05 | 8 | 10 |
| **ab** | 0.3 | 5 (5 also for −x) | 6 |
| **ab** | 0.05 | 8 (9 with stiffer contact) | **10–11**, still standing |

Other `ab` results from the same runs:

- **Drops:** A onto the base seated 92/75/69%, and B onto A 92/73/71%, at
  μ = 0.2/0.35/0.5, the same as `alternate`, so the reversed lean costs
  nothing. The tight fit drops to 52–56% at μ = 0.35. Current V0 numbers are
  in the V0 results table above.

What limits it:

1. **The hooks hold.** Seated bricks stay within 0.2° of their supporter.
   The stack instead accumulates sag (joint play plus contact compliance
   under an n² moment). In tracked runs the stop is the H0/2 sag criterion,
   not a fall.
2. **Placing a cantilevered brick is the real bottleneck.** With ideal aim,
   brick 9 lands level on a supporter that has sagged 5 mm and pitched 3°,
   so it tips before hooking. Tracking the measured pose fixes this.
3. **Coarse placement onto a cantilever still fails.** Seated: 1–2 bricks
   with the full error model (`--track --exact-courses 1`). Tilt (3°) and
   drop height are tolerated, but more than about 1 mm of XY error, or
   residual velocity, tips the one-tooth brick before it catches. Aim bias
   in either direction doesn't help.

Next: a capture feature for the hanging voxel (for example a downward guide
lip that engages the supporter's front ramp before the COM leaves support),
or a place-then-push strategy for cantilever bricks.

## 3D from the 2.5D system: walls along X and Y with turn pieces (`ring.py`)

The A/B brick goes 3D without changing its profile. Every brick is built from
**cells**: one voxel of the A or B profile swept through the depth with the
gutter. A cell can sit at any quarter-turn yaw. Courses alternate parts (A even,
B odd) as in the 2.5D lattice, so a cell nests on the one below when it has the
**same yaw** and the other part. A wall along Y is just the same brick rotated 90°.

| Part | Cells | Role | Mass A / B |
|---|---|---|---|
| S2 | 2 straight | the existing brick | 14.3 / 13.5 g |
| **L3 turn** | corner + 1 along x + 1 along y | the turn, even courses (A) | 21.3 g |
| I3 | 3 straight along +x, owns the corner | corner on odd courses (B) | 21.1 g |
| K2 closer | 2 straight, upstream (+x) end cut back | last brick of every course | 14.2 / 13.4 g |

Design rules that came out of building it:

- **A rigid turn seats along one stroke.** Only the L3's corner and x-arm
  cells carry teeth. Its y-arm cell (yaw 270, profile along its own wall)
  nests by keel and gutter. Teeth along both x and y would fight each other.
- **Corner bond.** Corners alternate L3 / I3 per course, like masonry. Every
  wall keeps one yaw, so the corner cell always runs along the wall that
  starts there: the even-course L3 turns into the previous wall through its
  toothless arm, and the odd-course I3 carries the corner along its own wall.
  The straights step one cell per course, and every cell column keeps its yaw
  while the part alternates. `ring.check_columns` verifies this.
- **Every course needs a closer.** A/B teeth trail −x, so every seating stroke
  runs the same way round the loop on every course and points back into its
  neighbour; no insertion order exists without a closer (the depth-first
  planner proves this, `tests/test_ring.py`). K2's upstream end is cut back by
  the stroke's sideways travel ((tooth_L + slot_extra)·cos φ + 0.5 mm), so it
  can go in last.
- Base cells are oriented to match whatever sits on them in course 0.

Results (simulated, uncalibrated contact):

- **Perfect placement:** a 6×6 ring 4 courses tall assembles **32/32
  seated** (`runs/ring-ab-ideal`: `assembly.mp4`, `final.png`, part STL/STEP
  in `part_*`).
- **Coarse placement (σ 4 mm, tilt 3°, μ 0.35, seeds 1–4,
  `runs/ring-ab-noisy-s*`):** 8, 5, 8 and 13 of 32 seated before a collapse.
  Every first failure is a jam with the brick 2–10 mm off across the wall's
  depth and 6–22° of tilt: the brick's limited Y capture (±3 mm flat band,
  30° sides), as with the earlier `alternate` ring. The ring geometry itself
  isn't the problem.

Commands:

```bash
uv run brixzle ring --ideal --export --video --out runs/ring
uv run brixzle ring --n 8 --courses 6 --seed 2 --keep-going --out runs/ring-noisy
```

Next steps:

- Give straights and turns a deeper Y gutter, or use the 3D project's square
  funnels on the cells: they captured about 95% of coarse drops.
- Add T and X junction pieces built from the same cells for interior walls (rooms).

## Span-weighted NSGA-III (history)

`configs/nsga-span-chosen.json` is the A/B pick of a span-weighted run (θ = 41.5°, α = 39.5°, 13.2 g)
that searched the lean mode as a gene. Its ideal-aim validation gave 5 / 5 stable bricks on ±x
overhangs, the bridge 7/7 and the wall 10/10; with the full coarse error model every design still
failed within the first 0 to 3 bricks of a bridge or overhang, so the limit is placement, not
geometry. The run's config and the non-A/B picks were removed in #22 (see git history, `5c15572`).
`eval.span_weight` and `run.patience` / `run.min_improve` remain in `optimize.py`.

## Baseplate

```bash
uv run brixzle baseplate --params configs/nsga-span-chosen.json --n 4 --out runs/plate4 [--direction -1] [--step]
```

`baseplate.py` sweeps `profile.base_outer` through the brick's Y gutter for each of the n rows, fuses
them, and cuts the bottom flat. The STL spans n·U × n·D and is watertight (`tests/test_baseplate.py`).

## Concept genes and multi-parent search (`configs/nsga-concepts.json`)

Three new brick concepts, each a gene where 0 gives the original brick (`params.CONCEPT_BOUNDS`):

- **Capture lip** (`lip_L`, `lip_w`): a centre tooth under the brick's middle. Every brick centre sits
  over a joint (overhang step, running bond) or a centre (stack bond), so each top carries a centre
  slot and two end half-slots, and the base gets a slot at every voxel boundary.
- **Snap barb** (`barb_h`, `barb_len`): a catch on each tooth's near flank with a pocket in the slot
  wall. A rigid barb cannot pass the slot mouth, so it is simulated as soft contact (`barb_timeconst`,
  uncalibrated) and kept out of the rigid outer profile.
- **Scarfed / stepped ends** (`end_scarf`, `end_step`): the course-neighbour joint leans toward the
  insertion axis, with a ledge one neighbour rests on. The planner check `course_insertable` rejects
  shapes that block the vertical approach; scarfs much past 0.3 usually do.

The run seeds 10 distinct parents (ab-tight, two lip variants, barb, two scarf-ledge variants, and
thick-stiff, shallow-long, compact and loose-capture bodies), so crossover mixes concepts rather than
near-clones. It scores tracked overhangs in both directions, an ideal 4-course bridge and a wall
(span weight 2), and drops of A and B.

Result (12 generations, early stop; measured before #22's shared tine bores): no lip or barb design reached the top 30. The winner descends
from the thick-stiff parent (`configs/ab-thick.json`: H0 19, tooth_L 8, 15.2 g). Validation
(96 drops per case; overhang with ideal + tracked aim):

| Design | Overhang +x / −x (20) | Bridge 3 | Wall 3×4 | Drops A, μ 0.2/0.35/0.5 | Drops B |
|---|---|---|---|---|---|
| ab-tight | 10 / 10 | 7/7 | 5/10 | 85 / 53 / 40% | 89 / 66 / 52% |
| **ab-thick** | **11 / 11** | 7/7 | **10/10** | 80 / 54 / 54% | 82 / 62 / 49% |
| nsga-concepts-chosen (lightest balanced) | 10 / 10 | 7/7 | 10/10 | 89 / 57 / 50% | 84 / 59 / 41% |

A deeper body and longer teeth buy one more overhang brick and a completed wall at +1.2 g. The
overhang still ends by sag, not by joints letting go.
