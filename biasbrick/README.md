# biasbrick: half-brixzle sockets on a rod, aimed off the seat

A brick family (issue #45) designed against the drone's **measured** approach error from Experiment 3
(14.8 mm mean, 27.8 mm p95, 38.5 mm max, zero-mean; `../brixzle/configs/error-exp3-r4.json`). It keeps
brixzle's principles:

- the drone drops coarsely and geometry does the precision;
- every guiding face is steeper than friction;
- a rigid part seats along one stroke, and its leaning tooth hooks;
- one passive fork carries it;
- it prints in one piece and stays inside the payload budget.

What is new is the **aim bias**: the drone aims off the seat, up a one-sided slope, so the error's ring
lands where everything slides one way into the seat. It reuses brixzle's profile, scene, settle loop,
error model and fork, and dfbrick's hull-and-fuse CAD (`../brixzle` and `../dfbrick` are editable
dependencies). An earlier 3D version of the same idea lives on as [`../biasbrick3d/`](../biasbrick3d/README.md).

```bash
cd biasbrick
uv sync --all-extras
uv run pytest          # about 20 s
```

## The design

A brick is brixzle's A/B brick cut in half: two **sockets**, each one voxel of the A (or B) profile, joined
by a **rod**. There is no voxel grid; sockets sit on sites 121 mm apart.

| Feature | Why |
|---|---|
| Socket: one brixzle voxel (sawtooth top, 38° leaning tooth on the keel, slot under the valley), U = 41 mm wide | brixzle's X catchment and hook. Courses alternate A and B, as in brixzle |
| Y section: an asymmetric sawtooth, a 42° main slope rising toward +y and a 12 mm, 60° **nest face** toward -y; the keel copies it | Landings slide down the slope and nest V-in-V, as in X. The slope must beat friction **inside the X sawtooth's V**, which multiplies it by about 1/cos 52°: 33° stalled at μ 0.5 |
| 43.5 mm corrugated **flange** past the +y face, slot channels left open | Catches the far side of the error without a deep socket; a tooth that lands on it slides down its own channel |
| Teeth only where the Y slope runs straight | The slot channel bends at the Y valley; a rigid tooth cannot slide through the bend |
| **Rod**: an 80 mm, 11 × 3 mm web | Longer rods tilt less when one socket catches first. Under the mean error, B onto A seats 57/62/61% at 40 mm and 78/77/84% at 80 mm (before the later fixes) |
| Two shelled square **sleeves** with a 45° V roof crossing the rod, 24 mm apart, 2.8 mm play | brixzle's fork, but a coarse tine still enters and the V roof centres the hanging brick. They fit one course (13.8 of 14 mm) |
| **Aim** (2, 20) mm from the seat | X: a little toward the side the tooth stroke comes from (+x). Y: up the slope, so the error's ring lands on the slope or the nest face |

Final design: `configs/bb-mean.json`. Part A is 23.5 g, B 23.0 g. With brixzle's 5.27 g fork (assumed, see
the limits) that is inside the 30 g payload. The brick is 162 × 64 × 92 mm. `configs/bb-p95.json`, sized for the
p95 error, is 40.7 g: the drone cannot carry it.

### How it got here

Each step was a simulated finding; the run directories are listed in `runs/biasbrick/INDEX.md` (local).

1. **v0/v1, 3D corner hoppers** (now `biasbrick3d/`): 46 g and 38.7 g, too heavy.
2. **v2, the rod brick with a 4 mm lip** at -y: shallow slopes stalled at μ 0.5, and steep ones threw bricks
   over the lip. The best lip variant seated 75/86/48%.
3. **v3, the Y nest**: the slide ends V-in-V. 2D capture maps set the aim.
4. **Jam fix**: near-side landings parked on the ridge of a 5 mm nest face; 12 mm took B onto A from 78/77/84 to 97/94/90%.
5. **Socket width**: far -x landings jammed at the 36.5 mm socket's edge; sizing for the aim's offset gives 41 mm.

### Rules (`biasbrick rules`, margin ≥ 0 means OK)

- **Sliding:** `long_ramp_slides`, `steep_face_slides`, `y_slope_slides` (with the V-groove factor), `nest_face_slides`.
- **Catchment** (a ring of the error plus 5° yaw around the aim): `capture_x`, `y_on_slopes`, `capture_y` (the far landing's COM over the flange).
- **Joint** (brixzle's): `nest_cone`, `clearance_absorbs_tilt`, `seated_play`, `hook_depth`, `tooth_printable`.
- **One course:** `rod_in_end_faces`, `sleeve_fits_course`, `sleeves_between_sockets`, `sleeves_on_rod`.
- **Fork:** `cavity_capture`, `roof_centres`, `hangs_stable`, `tines_straddle_com`, `com_in_sleeve_span`.
- **Build:** `wall_printable`, `single_piece`, `mass_max`.

The final design passes all of them. `rules.size_for_tolerance` gives U and the flange for a measured error.

## Commands

```bash
uv run biasbrick size   --approach-error ../brixzle/configs/error-exp3-r4.json --error-stat mean --out ../runs/bb-size.json
uv run biasbrick export --params configs/bb-mean.json --out ../runs/bb-export                 # STL, STEP, views, sections, rules
uv run biasbrick rules  --params configs/bb-mean.json --approach-error ../brixzle/configs/error-exp3-r4.json
uv run biasbrick animate --params configs/bb-mean.json --out ../runs/bb-anim                  # geometry only, not physics
uv run biasbrick drop   --params configs/bb-mean.json --on-brick --approach-error ../brixzle/configs/error-exp3-r4.json --out ../runs/bb-drop
uv run biasbrick drop   --params configs/bb-mean.json --samples 0 --map --ideal --on-brick --map-radius 14.82 --out ../runs/bb-map
uv run biasbrick assemble wall --seed 1 --approach-error ../brixzle/configs/error-exp3-r4.json --out ../runs/bb-wall   # --place: carried
uv run biasbrick push   --out ../runs/bb-push
```

`--set key=value` overrides a parameter. Every output directory must be new.

## Results

All results are simulated, with uncalibrated contact; they are not hardware rates. They were produced on
branch `45-biasbrick` and are local in `runs/biasbrick/current/` (git-ignored). The final design's runs are
`bb8-final-*`. Drops are n = 96, seed 0. Unmeasured terms keep brixzle's defaults: yaw σ 5°, tilt σ 3°,
release 5–30 mm and velocity σ 0.03 m/s.

### Drops (% seated, μ 0.2 / 0.35 / 0.5)

| Error | Onto the base | B onto a seated A | DF-mean, B onto a brick (dfbrick README) |
|---|---|---|---|
| legacy guess | 99 / 96 / 96 | 100 / 99 / 96 | 100 / 100 / 100 |
| **Exp 3 mean** | **99 / 97 / 96** | **99 / 94 / 94** | 96 / 95 / 90 |
| Exp 3 empirical | 85 / 82 / 80 | 89 / 80 / 83 | 77 / 78 / 72 |
| Exp 3 p95 | 43 / 44 / 41 | 40 / 42 / 36 | |
| Exp 3 max | 1 / 1 / 1 | 2 / 2 / 1 | 0 / 0 / 0 |

### Structures (μ 0.35; stable bricks per seed; stop at the first failed placement)

| Structure | Ideal aim | Legacy guess, seeds 1–4 | Exp 3 mean, seeds 1–8 | DF-mean, Exp 3 mean, seeds 1–4 |
|---|---|---|---|---|
| Tower (6) | 6 | 6, 6, 6, 6 | 6, 6, 1, 2, 6, 6, 6, 6 (6 of 8 complete) | 4, 4, 1, 6 |
| Wall, 4-3-2-1 (10) | 10 | 10, 10, 10, 10 | 10, 10, 1, 10, 10, 10, 10, 10 (7 of 8) | 4, 10, 1, 10 |
| Bridge, key last (5) | 5 | 5, 5, 5, 4 | 5, 5, 1, 5, 5, 5, 5, 5 (7 of 8) | 4, 4, 1, 5 |
| Overhang ± (12) | 1 | 1 | 1 | 1 |

- **Seed 3** draws a 16.6° yaw (3.3σ of the unmeasured model) on its second placement and fails every
  structure, as it does for DF.
- **Push** (ideal 6-tower, ramped force on the top brick until it moves 2 mm): 1.26 N in +x, 1.37 N in -x,
  1.12 N in -y and 0.58 N in +y (up the slope). DF-mean took 0.79 N and A/B V0 0.40 N.
- **Overhangs.** A dropped overhang brick falls before its tooth engages: its hanging socket drops first and
  twists it. Set at its seat at rest instead (`--place`, an ideal carried placement), one overhang step holds in
  both directions. At the third brick the +x stair moves 2.5 mm, past the 2 mm collapse threshold, and the -x
  stair falls. That one step already reaches 121 mm, further than brixzle's 5-brick stair (4 steps of 25 mm,
  per its README), but a second step puts about 4 times the first step's moment on the lower joint.

## Limits and open problems

- Contact, friction and the error model are uncalibrated. Yaw, tilt and release height are not measured,
  and the structures are sensitive to large yaw.
- **Fork:** the sleeves take brixzle's fork geometry, but no fork was built or weighed for them. The payload
  check assumes brixzle's 5.27 g fork.
- **Size:** 162 × 64 × 92 mm. Whether it clears the prop guards on the simulated drone is untested (`fly-assemble`
  is not ported yet).
- **No corners or ring:** a brick along Y would turn its sawtooth and nest by 90°, and no socket receives a
  crossing brick.
- **Overhangs** need carried placement and a way to pin the tail (or a finer step); see above.
- A tooth lengthened to 8 mm jammed every drop; the cause was not found.
- The capture maps behind the aim ignore yaw; the aim was then checked by drops.
