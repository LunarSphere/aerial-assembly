# Brixzle: drop-assembled bricks

A parametric brick that a drone can drop *coarsely* into place and that
gravity slides home. It is built with CadQuery, tested in MuJoCo assembly
trials, and tuned with NSGA-III. Drone dynamics are not modelled: a brick is a
free body released at a sampled pose error above its lattice seat.

This is a separate uv project pinned to Python 3.12 because CadQuery has no
3.14 wheels. It reuses `aerial_assembly` from `../simulation` through an
editable path dependency.

```bash
cd brixzle
uv sync --all-extras
uv run pytest
```

## The V0 brick (initial educated guess)

`uv run brixzle export --params configs/v0.json --out runs/v0` writes the
STL, STEP, a profile plot, and the rule margins.

The brick is a 2.5D "chevron": an XZ profile spanning two voxels
(2U × U × ~U, with U = 25 mm), swept through the depth. It weighs 13.7 g of
PLA at 100% infill.

| Feature | Why |
|---|---|
| Top is a continuous asymmetric sawtooth: a 50° long ramp and a 52° steep face, peaks at voxel boundaries | The **hopper**. Every upward face slopes into a valley at a steeper angle than friction requires (atan 0.4 = 22° + 8° margin), so there are no flat landing spots. The catchment is the whole voxel. |
| Bottom keel matches the top of the course below | Courses nest as V in V, which gives precision from geometry rather than from the drone. |
| Tapered prismatic tooth at each keel point, leaning α = 38° and entering a slot cut along the steep face | **Fine seat plus hook.** The entry gap is 2.6 mm (taper + clearance) and the seated gap 0.6 mm. A slanted tooth cannot be lifted straight out, which resists tipping. |
| In `alternate` mode the teeth lean opposite to the top slots | This forces **ABA** courses (B = A rotated 180° about Z), so one brick type covers both orientations, and joints are loaded from both sides. |
| Y gutter: 30° sides with a ±3 mm flat centre band | Y capture of about ±9 mm, while straight Y tine bores still fit. |
| Two through-Y tine bores (2 mm rods), 25 mm apart, about 9.5 mm above the COM and straddling it | **Passive fork.** The brick hangs level on non-actuated tines. |
| Lightening holes leave a 1.2 mm skin-and-rib frame (I-beam-like) | Reduces mass. |

### Placement-error model (coarse drone)

The drone aims at a point offset by a 3 mm bias toward the side the tooth
stroke comes from. Errors are x/y ~ N(0, 4 mm) clipped at ±12 mm, yaw σ 5°,
roll/pitch σ 3°, release 5–30 mm above the highest-reach height, and residual
velocity σ 0.03 m/s.

### V0 results (simulated, uncalibrated contact; not hardware rates)

| Trial | Result |
|---|---|
| Single-brick X capture sweep onto the anchored base | Seats from −8.5 to +15.5 mm (about ±11 mm around the aim point) |
| Drop Monte Carlo onto the base, μ = 0.35, n = 96 | 81% seated; 95th-percentile peak force 27 N; settles in 0.19 s |
| Drop MC at μ = 0.2 / 0.35 / 0.5 (n = 96, other seed) | 88% / 78% / 69% |
| Ideal assembly: tower 6, running-bond wall 3×4, corbelled bridge with key | All complete. Max sag 0.5 mm |
| Single-brick staircase overhang (ideal) | 4 stable (+x), 3 stable (−x). Baseline two-peg block: 4 |
| Error-sampled assembly, per-brick success | Tower 0.68, wall 0.47, bridge 0.29. Light receiving bricks are knocked by impacts, and errors compound |

### What the sims taught (encoded as rules)

- **Clearance dominates jamming.** Jams come from two-point wedging of the
  tooth against the slot mouth while the keel rides the steep face. A tapered
  tooth gives a large entry gap and small seated play.
- **Seated play limits overhang**, but tight seats cost capture. That is a
  real trade-off, so it is left to the optimizer.
- **Alternating lean resists loads from both sides but not a one-sided
  cantilever.** In an ABA staircase every other joint's extraction direction
  lines up with the overhang rotation. `uniform` mode (every course A) hooks at
  every joint and reaches about 7 overhang bricks, but it hurts drops and walls.
  `lean_mode` is a design variable.
- Gutter angle, tine-bore placement, and centre-band width are coupled. A full
  V gutter leaves no room for straight tines above the COM.

## Commands

```bash
uv run brixzle rules  --params configs/v0.json            # rule margins (the design's justification)
uv run brixzle export --params configs/v0.json --out runs/v0
uv run brixzle drop   --params configs/v0.json --samples 96 --out runs/v0-drop --video 2
uv run brixzle assemble bridge --params configs/v0.json --ideal --video --out runs/v0-bridge
uv run brixzle assemble wall   --params configs/v0.json --seed 3 --out runs/v0-wall-noisy
uv run brixzle push --params configs/v0.json              # lateral push on an ideal tower
uv run brixzle optimize --config configs/nsga.json --out runs/nsga
```

Structures: `tower`, `overhang+`, `overhang-`, `wall`, `bridge`. Every output
directory must be new.

## Pipeline

- `params.py`: `BrickParams`, `V0`, NSGA bounds.
- `profile.py`: shapely XZ profile (sawtooth, slots, teeth, slanted features, tine bores, lightening, base).
- `cad.py`: CadQuery solid (Y-band sweep), STL/STEP, mass properties. Collision pieces are convex
  **by construction** (ear clipping + Hertel–Mehlhorn, then an affine sweep per Y band). There is no
  tetgen; the collision volume equals the outer profile volume exactly.
- `rules.py`: analytic rules used as NSGA constraints and as a pre-simulation filter.
- `lattice.py`: voxel placements, A/B orientation, `networkx` support graph, and the assembly-order
  planner. Supporters go first, and the approach path is swept against placed bricks, so courses fill
  against the tooth stroke and the key brick goes last. It also has a running-bond voxel tiler.
- `structures.py`: trial structures on anchored bases.
- `scene.py`: MJCF built once, with parked gravity-compensated bodies activated one at a time (the
  `aerial_assembly.chain` pattern).
- `trials.py`: drop MC, capture sweeps, sequential assembly (seating judged relative to the actual
  supporter, collapse = motion during a stage, floor contact, or sag > H0/2), push test.
- `optimize.py`: pymoo NSGA-III. Objectives are −P(success), F_max, T, mass, and collisions; rules are
  constraints. The population is seeded around V0 in both lean modes, and the final pick uses the
  weighted score w·(P, F, T, M, C) on normalised objectives. Each generation is checkpointed, with
  per-worker evaluation logs.

## Limits

- Friction, contact softness, and the error model are uncalibrated guesses.
- Structure trials stop at the first failed placement; real drones could retry.
- 2.5D only. The lattice and graph are 3D-ready, but crossing courses and domes are not implemented.
- Fork release (tine withdrawal) is checked geometrically only, not simulated.

## First NSGA-III run (35 × 20, `runs/nsga-full`, chosen design in `configs/nsga-chosen.json`)

The chosen design is `uniform` lean, 12.0 g, U = 24.6, H0 = 13.1, θ = 46°,
and shorter teeth (5.3 mm). Re-validated with the current code against V0:

| Trial | V0 | NSGA pick |
|---|---|---|
| Drop seated, μ = 0.2 / 0.35 / 0.5 (n = 96) | 0.88 / 0.73 / 0.68 | 0.90 / 0.68 / 0.60 |
| Ideal overhang +x / −x | 4 / 3 | 6 / 1 |
| Ideal wall 3×4, bridge | 10/10, 7/7 | 5/10, 3/7 |
| Error-sampled per-brick: tower / wall / bridge | 0.65 / 0.41 / 0.28 | 0.56 / 0.67 / 0.54 |

Neither design dominates. This run used a small noisy fitness (24 drops per
friction, one structure seed) and the earlier narrow bases. Before trusting a
pick, rerun it with more samples and seeds.

## A/B brick pair: pegs leaning against the build direction (`lean_mode: "ab"`)

Idea: keep the alternating sawtooth body, but use a distinct **type-B brick**
so every course's teeth lean opposite to the build direction. Every joint then
hooks against the overhang rotation. In `alternate` mode B is just A rotated,
so only every other joint hooks.

- **A** (even courses): bottom T′ with *reversed* teeth, top T with normal slots.
- **B** (odd courses): bottom T with normal teeth, top T′ with *reversed* slots.
- A reversed interface mirrors the slot axis about vertical. It stays inside
  the nesting cone while α ≤ 90 − θ (the existing `nest_cone` rule).
- **−x builds use the same two parts rotated 180°.** `Placement(direction=-1)`
  handles this, and the base's slots follow the build direction.

Configs: `configs/ab.json` (V0 fit) and `configs/ab-tight.json`
(clearance 0.05). Commands: `brixzle export --part B`, `brixzle drop --part B`
(drops B onto a seated A), and `brixzle assemble ... --track` (aim at the
supporter's measured pose, as a global camera would allow).

Single-brick staircase, 20 requested (simulated, uncalibrated contact):

| Mode | Fit (clearance) | Ideal aim | Camera-tracked aim |
|---|---|---|---|
| alternate | 0.3 | 4 | 4 (falls) |
| alternate | 0.05 | 4 | 6 |
| uniform | 0.05 | 8 | 10 |
| **ab** | 0.3 | 5 (5 also for −x) | 6 |
| **ab** | 0.05 | 8 (9 with stiffer contact) | **10–11**, still standing |

Other `ab` results:

- **Drops:** A onto the base seats 92/75/69%, and B onto A 92/73/71%, at
  μ = 0.2/0.35/0.5. That is the same as `alternate`, so the reversed lean
  costs nothing. The tight fit drops to 52–56% at μ = 0.35.

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
