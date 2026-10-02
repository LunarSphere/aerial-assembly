# Brixzle-3D: drop-assembled polycube bricks

A 3D follow-up to `../brixzle` (2.5D), which it does not import or modify.
The brick is a polycube, an L tromino by default. Every cell carries the same
4-fold-symmetric interface, so a brick in any of its four rotations nests on
any occupied cell below. Structures are voxel sets.

Uses the repo-root venv (Python 3.14):

```bash
source ../.venv/bin/activate
pip install -e '.[test,video]'   # already done; also needs manifold3d
python -m pytest
```

## Design

| Feature | Why |
|---|---|
| Each cell's top is an inverted square-pyramid **funnel** (35° faces), and its bottom is a matching pyramid **keel** | Coarse placement: a brick landing anywhere within ±U/2 in X and Y slides home. Yaw is corrected too, and there is no flat spot to perch on. |
| Tapered square peg under each keel tip; tapered socket at each funnel bottom | Big entry gap (taper + clearance), small seated play. Wedging was the jam mode in 2.5D. |
| **v1 hook:** pegs lean 15° toward body −x, and each socket is a **cross of four slanted lobes** | A leaning peg can only leave along its own axis, so joints carry some tension. Four lobes accept the lean in any lattice rotation. Orient a brick so its extraction direction points up and toward the overhang. |
| Constant-thickness cells (H = 13 mm, U = 24 mm) | Courses stack at a fixed pitch and every cell is a voxel. |
| Two straight X tine bores, one per occupied row, about 5 mm above the COM and straddling it | Passive fork; the brick hangs level. |
| FDM mass model: 0.8 mm shell + 20% infill | The L weighs 11.7 g printed (≈26 g if solid). The 25 g cap and 40 g payload are met. |

Collision geometry is **convex by construction**. Each cell splits into four
diagonal quadrants, and in each quadrant every face is one plane, so the
pieces are half-space intersections (A–F in `geometry.py`). MuJoCo gets them
exactly. The print mesh is their manifold union minus the bores, and the STEP
is fused through CadQuery.

## Results (simulated; contact parameters uncalibrated, not hardware rates)

Error model: X/Y σ 4 mm (clipped at ±12), yaw σ 5°, tilt σ 3°, release
5–30 mm above the highest-reach height, residual velocity σ 0.03 m/s.

| Trial | 2.5D V0 (`../brixzle`) | 3D v0 (vertical pegs) | **3D v1 (15° hook)** |
|---|---|---|---|
| Drop seated, μ 0.2 / 0.35 / 0.5 | 0.88 / 0.78 / 0.69 | 0.98 / 0.98 / 0.98 | 0.95 / 0.95 / 0.95 |
| Capture (aligned sweep) | ≈ ±11 mm in X only | ±8 mm square in X and Y | ±8 mm square in X and Y |
| Peak net impact force (5 ms average) | – | ≈ 1.8 N | ≈ 1.8 N |
| Ideal 8-course spiral pillar (1 L per course) | – | creeps and fails at 7 | **8/8** |
| Ideal 2×3 column, 3×4 crossed-seam wall | – | complete | complete |
| Ideal single-L cantilever staircase | 4 | 2 | **4** |
| Noisy per-brick seated: column / wall / pillar | – | 0.94 / 0.86 / 0.47 | 0.91 / 0.76 / 0.33 |

## What iterating taught

1. **Funnels beat sawtooth.** Square-pyramid funnels capture in X and Y and
   correct yaw. Drop success is about 98% and barely depends on friction.
2. **No tension means no cantilever.** With vertical pegs the pillar creeps
   until it fails, and a stepped L corbel always leaves a front L with one
   third of its cells supported. Hooks are needed for overhangs.
3. **Hook direction matters.** Only a lean whose extraction points up and
   toward the overhang helps; the opposite sign behaves like vertical pegs.
   Seating cost grows with lean: 15° keeps 95%, 25° drops to 80–90%.
4. **L-tromino tiling facts.** A 3×3 square can't be tiled with Ls, so 2×3
   is the basic block. A balanced L (COM over its seated cells) can advance a
   corbel front by at most half a column per course. The L's COM sits in its
   corner cell, so an L balances on that one cell.
5. **Measure net force, not summed normals.** Summed normals report wedge
   squeeze and solver spikes (100+ N); net force gives physical numbers.

## Commands

```bash
brixzle3d rules
brixzle3d export --out runs/v1                        # STL, STEP, mass model, rules
brixzle3d drop --samples 64 --out runs/drop --video 2  # MC + XY capture grid
brixzle3d assemble pillar --ideal --video --out runs/pillar
brixzle3d assemble wall --seed 3 --keep-going --out runs/wall-noisy
brixzle3d push column
brixzle3d optimize --config configs/nsga.json --out runs/nsga
```

Structures: `pillar`, `column`, `overhang`, `wall`. Configs: `v0-vertical.json`,
`v1-hooked.json` (the default). `params.SHAPES` also has I2, I3, O, T, S and L4.
In drop trials they match the L's success rate but weigh 25–30% more.

## Next steps

- A corbel/bridge planner that picks rotations so every overhanging brick
  hooks the right way. `lattice.tile` already enforces balance, but not hook
  orientation.
- Noisy multi-brick success (0.76–0.91 per brick) is the main gap: light
  receiving bricks get knocked by impacts.
- Fork release and dome trials are not simulated yet.

## First NSGA-III run (21 × 8, `runs/nsga`, pick in `configs/nsga-chosen.json`)

Fitness: drops at μ 0.2/0.5, ideal pillar and overhang, noisy column. The pick
is a smaller and lighter L: U = 20.3, H = 11.5, θ = 43°, lean 17°, infill
10%, **7.5 g**. Re-validated against v1 with the full trials:

| Trial | v1 (11.7 g) | NSGA pick (7.5 g) |
|---|---|---|
| Drop seated, μ 0.2 / 0.35 / 0.5 | 0.95 / 0.95 / 0.95 | 0.84 / 0.84 / 0.86 (6/64 land in the wrong cell) |
| Ideal pillar, column, wall | all complete | all complete |
| Ideal overhang | 4 | 4 |
| Noisy per-brick: column / wall / pillar | 0.91 / 0.76 / 0.33 | 0.91 / 0.71 / 0.38 |

It trades about 10 points of drop capture for 36% less mass. A smaller cell
means a smaller funnel, so the 4 mm placement error more often reaches the
neighbouring cell. Keep v1 unless mass matters most; the weights in
`configs/nsga.json` set this trade.
