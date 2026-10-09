# Test flight: pink A/B brick → pink platform, sensed by the overhead RealSense

The Crazyflie 2.1 Brushless (MSH2-30 fork) picks the pink brick from its blue cradle and seats it
on the pink platform. **Nothing is hard-coded.** Before every pick and every check, the overhead
D455 measures the brick and the platform in the Lighthouse frame.

**Status (2026-10-09).** Everything below runs end to end, but only on *synthetic* camera
frames: calibration, sensing, the field plan, and the real `flight` mission against a fake
Crazyflie. Nothing has seen a real D455 frame yet; the HSV ranges come from phone photos.
Stage 0 (§5) is the first real test.

```bash
cd brixzle && uv sync --all-extras
uv run testflight demo --out runs/tf-demo        # synthetic: calibrate, sense, plan, fly the mission (fake CF)
uv run pytest -q tests/test_testflight.py        # 14 tests, ~1 min
```

## Run order (lab checklist)

Run from `brixzle/`. Set `URI` to your drone's radio address. Details for each step are in the
sections below.

```bash
URI=radio://0/80/2M/E7E7E7E7E7
uv run testflight lighthouse                                  # check the Lighthouse geometry file
# 1. Tags: gripper off; drone centred on each tag, nose towards the tag's right edge, Enter.
#    --size = black-square edge (m); --drone-height = deck sensors above the floor (m). Measure both.
uv run testflight survey tags --uri $URI --size 0.080 --drone-height 0.030
# 2. Gripper on. Home: drone on the stand. Fence: drone in two opposite inside corners of the cage.
uv run testflight survey home  --uri $URI
uv run testflight survey fence --uri $URI
# 3. Camera: all 6 tags visible. Every check must print "ok".
uv run testflight calibrate
# 4. Stage 0 (no flight): check the overlays in runs/s0; move the objects and repeat about 10 times.
uv run testflight sense --repeat 10 --debug runs/s0
uv run testflight survey point --uri $URI                     # drone placed on an object: the reference position
# 5. Plan for the current layout (re-run with a new --out whenever the objects move).
uv run testflight plan --out runs/tf-plan-1
# 6. Preflight over the radio (never arms), then fly.
uv run fly runs/tf-plan-1/plan.json --uri $URI --check-only --realsense src/testflight/rig.json
uv run fly runs/tf-plan-1/plan.json --uri $URI --arm --realsense src/testflight/rig.json
```

Before the first `--arm`, do stages 1–2 (§Stages): the IR projector check and a hover with no payload.

## Parts and layout

| Thing | What it is |
|---|---|
| Brick | A/B pair, part **A** (`configs/ab.json`, V0 size: 49.7 × 25 mm, `stl_exports/2p5d_ab/brick_A.stl`), pink |
| Platform | `base_plate_build_plus_x_4vox.stl` (100 × 25 mm, pink). This is the plan's `base0`; the brick seats 25 mm along its +x from the frame origin. |
| Cradle | The blue pick fixture. The brick's origin is 11.5 mm above the floor in it, the same as on the platform. |
| Camera | D455, top of the cage, looking down, all 6 tags in view |
| Tags | **AprilTag 25h9, ids 0–5** (detected in your photos; not 36h11) |
| Lighthouse | 2 base stations; geometry in `testflight_lighthouse.yaml` (see below) |
| Launch stand | About 54 mm boxes. With the fork on, its lowest point is about 31 mm above the floor. |

**Placement rule (yaw).** The camera sees each footprint as a rectangle, so yaw is known only up to a
half-turn. Mark the **+x end** of the cradle and of the platform once (for example with tape), and
place both with +x within ±60° of Lighthouse +x (`yaw_prior_deg` in `rig.json`). Standing on the
cradle's tine-guide side (−y), +x is to your right. The sensor refuses yaws near ±90° from the prior.
Randomize positions and yaw within that band between trials.

## Lighthouse (`testflight_lighthouse.yaml`)

`uv run testflight lighthouse` solves it with cflib's estimator:
- **Base stations:** BS0 is at (0.06, −1.23, 1.43) m and BS1 at (0.12, 2.89, 1.71) m. They face each
  other and are 100° apart as seen from the origin, which is good geometry.
- **Error:** the solver's error is 4–5 mm.

Before relying on it:
- **Scale.** cfclient forces the x-axis sample to exactly **1.000 m**, so the whole frame's scale
  depends on that one placement. Tape-check it: 1 cm off means 5 mm of error at 0.5 m.
- **Samples.** It has **no xyz-space or verification samples**. Add 10–20 xyz-space samples around the
  pick/place area at 0–30 cm, plus a few verification samples, then re-solve.
- **Origin height.** z = 0 is the **deck's sensor plane at the origin sample**, not the floor. The tag
  survey measures where the floor is.
- **After any change.** Re-running the geometry estimation changes the frame. Redo the tag survey and
  the camera calibration afterwards.

## Calibration (in order; files land in `calib/`)

1. `testflight survey tags --uri radio://... --size <m> --drone-height <m>`.
   - Take the gripper off and put the drone on each tag in turn: centred, nose towards the tag's
     right edge.
   - `--size` is the black-square edge, measured with calipers.
   - `--drone-height` is the height of the Lighthouse deck's sensors above the floor while the drone
     sits on a tag.
   - It writes `tag_survey.json` and warns if the tag heights disagree by more than 1 cm.
2. `testflight survey home` with the gripper on and the drone on the launch stand. Then
   `testflight survey fence`, with the drone in two opposite inside corners of the cage, inset 15 cm.
   Both write `field.json`.
3. `testflight calibrate`. This runs PnP on all 24 tag corners, then refines it.
   - Pass criteria: reprojection under 1 px, each left-out tag predicted within 3 mm, depth scale
     within 1 ± 0.03, and tag depth within 4 mm of the floor.
   - It writes `camera_pose.json` and prints PASS/FAIL per check.
4. Optional: save a burst for offline work with `testflight capture --out runs/cap1`. Every command
   takes `--capture` to replay one.

The survey only reads the state estimate. It never arms and never sends setpoints.

## Sensing algorithm (`vision.py`)

1. **Depth to points.** Take the median depth over a burst of 8 frames, aligned to colour. Each pixel
   becomes a Lighthouse point, and h is its height above the surveyed floor plane.
2. **Masks.** HSV pink and blue masks are grown by 1 px; the height window then decides the edge pixels.
   - **Brick:** pink pixels 30–55 mm above the floor. That keeps 83% of the brick top and leaves the
     platform's sawtooth peaks (≤ 27 mm) about 2σ below the window.
   - **Low parts:** pink or blue pixels 3–32 mm above the floor, minus the brick's footprint.
   - **Dropped:** a carried brick (higher up), the floor, the tags and clutter.
3. **Fitting.** Points are clustered on a 5 mm floor grid. Each cluster gets the object's CAD
   rectangle, placed to cover the most area over (yaw, x, y). This works with holes and with partial
   views, such as the platform's two ends when the brick sits on it.
   - For the brick, the centroid and principal axis of the points inside its rectangle refine the fit.
   - A brick seated in a cradle or on the platform takes its holder's yaw, which is measured over
     100 mm instead of 50 mm.
4. **Brick height.** The brick's z is the median of (measured height − CAD top surface at that point).
   A brick more than 8 mm above seated is refused.
5. **Safety rules.** Two consecutive bursts must agree within 2.5 mm and 2°. The platform pose is
   cached from its last full view. Two brick-like objects, a partial brick, or an unclear half-turn
   raise `SceneError`, and the mission then lands in place.

Accuracy on synthetic 2 m frames, 25 random layouts × 2:

| Case | Worst error |
|---|---|
| Brick in the cradle | 1.5 mm in x/y, 2.0 mm in z, 1.3° |
| Brick on the platform | 1.6 mm, 0.8 mm, 1.3° |
| Platform | 1.7 mm, 1.3° |

These numbers come with optimistic noise (1.5 mm after the burst median). Stage 0 measures the real
ones. At 2 m one pixel is about 3 mm, so the brick spans about 16 × 8 px. If the real error exceeds
3 mm, lower the camera.

## Flying it

```bash
uv run testflight plan --out runs/tf-plan-1          # sense now, compile the A/B plan for this layout, check, dry run
uv run fly runs/tf-plan-1/plan.json --uri radio://0/80/2M/E7E7E7E7E7 --check-only --realsense src/testflight/rig.json
uv run fly runs/tf-plan-1/plan.json --uri radio://0/80/2M/E7E7E7E7E7 --arm --realsense src/testflight/rig.json
```

The field plan (`fieldplan.py`) changes the compiled plan in these ways:
- shifts every absolute height into the Lighthouse frame (seat level = floor + 11.5 mm);
- takes the surveyed home and geofence;
- re-times the transit legs and the flight home for this layout's distances;
- removes `base0` from `bases`, so the mission seats on the camera's platform;
- sets the nominal frames to the sensed poses, so the offline checks and the dry run cover this layout.

The mission re-senses before every pick, check and place. Re-plan whenever you move the objects.
The camera tolerance is 1.5 mm and 1° (`rig.json → field`), which widens the seat check to 5.5 mm.

## Stages (do not skip)

| Stage | Pass criterion |
|---|---|
| 0. Static sensing | Run `testflight sense --repeat 10 --debug runs/s0`, moving the cradle and platform between runs (10 spots). Compare with `testflight survey point` with the drone placed on the object. The criterion is ≤ 3 mm and 2°, with no wrong detections; look at the overlays. Re-tune `colors` in `rig.json` here if needed. |
| 1. Lighthouse with the IR projector | With the drone on the stand, compare the Kalman noise with `camera.emitter` true and false. If the projector hurts tracking, set it to false (the floor texture is enough for stereo). |
| 2. Hover, no payload | Check the `ctrlMel` gains first, then hover over each sensed object at transit height. Hover error should be under 5 mm. |
| 3. Dry run | `testflight plan` prints "dry run complete". |
| 4. Pick only | 10 pick + test-lift attempts. |
| 5. Pick and place | 10+ trials, randomizing positions and yaw. Log `runs/testflight-log/` (overlays and JSON per camera call). |

## Files

| File | What it does |
|---|---|
| `rig.json` / `rig.py` | Config. Any key of `DEFAULTS` can be overridden. |
| `model.json` / `model.py` | Footprints, seat heights and the brick's top surface, from the brixzle CAD (`testflight model`). |
| `calib.py` | Tag detection, PnP, leave-one-out check, depth scale, `CameraPose` |
| `camera.py` | D455 burst capture; save and replay |
| `vision.py` | The detector and the debug overlay |
| `sense.py` | `CameraPoseSource`: what `fly --realsense` uses |
| `fieldplan.py` | The field plan |
| `survey.py` | Lighthouse readings with cflib (read-only) |
| `lighthouse.py` | Solves and checks `testflight_lighthouse.yaml` |
| `synth.py` / `demo.py` | Synthetic camera and world; the end-to-end demo |

## Open items

- **Not validated on real data.** HSV, depth noise and the D455's colour/depth alignment at object
  edges are all untested. Stage 0 decides.
- **Cradle height.** The brick's seated height (11.5 mm) assumes the cradle's floor matches the CAD
  base. If `sense` reports `z_rel_mm` consistently off, measure it and set `brick.origin_height` in
  `model.json`.
- **Roll and pitch** are not estimated by default (`estimate_tilt`). Depth noise gives about 2° of
  noise on a 25 mm-wide brick.
- **The MuJoCo sim (`fly --sim`) can't fly A/B bricks yet**, because `fixture.build_fixture` raises for
  `lean_mode: ab`. The plan compiler now handles A/B, and the plan was dry-run and flown against the
  fake Crazyflie.
- **The brick's ~6 mm slide along the tines** (from the sim) is still open. VERIFY_PLACE catches a
  misplaced brick (tested at 12 mm) and tries one re-grasp from where the camera sees it.
