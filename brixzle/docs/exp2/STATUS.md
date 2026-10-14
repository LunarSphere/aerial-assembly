# Experiment 2 status: drone pick-and-place with a passive gripper

Branch `brixzle-exp2-passive-gripper` (from `brixzle-claude-exp2`). Last update 2026-10-08.
All results are simulated with uncalibrated contact and aerodynamics; they are not hardware rates.

## Decisions so far
| Topic | Decision |
|---|---|
| Thrust model | Datasheet: 30 gf (0.294 N) per motor. crazyflow's 0.2 N is kept as a named alternative. |
| Drone mass | 43.4 g (crazyflow) + 2.7 g Lighthouse deck = 46.1 g |
| Simulation architecture | Hybrid: the drone flies inside the brixzle MuJoCo scene, with full tine contact in transit |
| Lighthouse | Build area registered (surveyed in the Lighthouse frame), **no bias** until measured in real flights |
| Gripper | Fork, printed in one piece of PLA |
| Controller | Firmware Mellinger (`stabilizer.controller = 2`), matching crazyflow; `hover.py` uses PID |
| Fixture | Unhooked cradle slots with open-top tine guides |
| Hardware | Only you fly it. I never arm the drone or open a radio link. |

## Checkpoint 1: setup, thrust budget, dependencies, architecture (`26982bf`)
- **Thrust budget.** On datasheet thrust, T/W is 2.60 with no payload and 1.58 with 30 g, so your 30 g limit binds first. On crazyflow's thrust it is 1.07 with 30 g, and T/W ≥ 1.5 allows only about 8 g of payload, which is less than one brick.
- **Dependencies.** A uv `override-dependencies` pin to mujoco 3.13 works: crazyflow's suite gives 713 passed and 46 skipped, plus 28 of its example scripts passed and 4 skipped. brixzle doesn't import crazyflow at all. Its Mellinger controller and rotor model are ported to numpy (MIT licence, commit d28ec70). The port matches pure crazyflow to 0.8 mm RMS on a waypoint flight.
- **Speed.** Resting bricks were the cost: 7–8 s of compute per simulated second. MuJoCo island sleeping cuts that to 0.14 s, so a placement takes about 3 s of wall time.
- **Hover offset.** crazyflow's Mellinger hovers about 8 cm high (linear PWM map). The sim compensates through the firmware's `ctrlMel.mass` parameter.

## Checkpoint 2: first full pick, transit and place (`2a4a910`)
- **What works.** The simulated drone picks a V0 brick from the fixture, flies it, seats it on the tower base and withdraws, using cflib-style commands only.
- **Single-brick success** (seated per ground truth):

  | Lighthouse model | Success |
  |---|---|
  | Unregistered (bias σ 6.3 mm) | 28% |
  | Registered, with bias gradient | 80% |
  | Registered, no bias (current default) | 97.5% |

- **Failure modes the simulation exposed, and the fixes:**
  - Tines couldn't hit the 2.6 mm bores → tapered guides on the fixture, plus aiming 1.5 mm low.
  - A guide closed on all sides trapped hooked bricks → unhooked cradles; the brick lifts straight up.
  - Prop guards clipped the brick → brick top now 8 mm below the guards.
  - The default Mellinger has no roll/pitch integral, so a 1 mm centre-of-mass offset gave about 3 cm of hover error → the mission sets `ctrlMel.ki_m_xy = 1e5` and `i_range_m_xy = 0.1`.
  - Collision artefacts (tines snagging) → capsule tines and square bore holes for collision.
- **Measured errors** (registered, no bias): the Lighthouse error is under 1 mm. The brick sits about +5.8 mm off in Y because it slides along the tines after the pick. This is still open.
- **Videos:** `docs/exp2/cp2-single-brick-close.mp4` and `docs/exp2/cp2-single-brick-wide.mp4`.

## Checkpoint 3 (before long optimisation runs): not reached yet

## Checkpoint 4: one flight program for sim and hardware (`efc164d`)
- **One `plan.json`, three ways to run it:**
  - `fly plan.json --sim`: flies it in MuJoCo.
  - `fly plan.json --uri radio://...`: dry run by default, with no radio. It checks the plan offline (geofence, speed and acceleration limits, setpoint rate) and lists all 177 cflib calls it would send.
  - `--check-only` connects and runs the preflight without arming. `--arm` flies and requires `--poses poses.json` (the RealSense code is a stub).
- **Preflight**, in the style of `hover.py`:
  - acknowledged parameter writes
  - Lighthouse deck present, Kalman estimator converged
  - battery at least 3.8 V
  - drone on the launch stand, inside the geofence
- **Interrupts.** Ctrl+C or any error hands back to the high-level commander, lands in place, stops the motors and disarms.
- **Parity tests (cflib mocked):**
  - The simulator's classes match cflib 0.1.33 signatures exactly.
  - The same plan gives an identical cflib call log in the sim and on the mocked hardware.
  - A simulated `go_to` ends within 2 cm.
  - The watchdog levels the drone at 0.5 s without setpoints and cuts the motors at 2 s.
- **Tests:** brixzle 85 passed, flight 11 passed, simulation 55 passed (the tests that need local CAD exports are excluded).
- **Artifacts:** `runs/exp2-cp4/` holds `plan.json`, `dry_run.txt` and the sim video.

## Printable fork (`fccfdfd`)
- **Files:** `docs/exp2/gripper-fork-v0/` holds `gripper.stl` (one watertight solid, 34 × 39 × 36 mm, about 5.1 g PLA), the STEP file, `gripper.json` and `preview.png`.
- **Simulation:** the payload is 19.0 g (T/W 1.86 on datasheet thrust), and the sim still seats 20 of 20 bricks.
- **Before printing:**
  - The pin-hole positions (2 rows × 10, 2.0 mm pitch, y = ±11 mm) come from CAD. Check them on the real drone.
  - The bracket probably needs long-pin headers, glue or tape to stay on.
  - Print on its side with supports, so the tines lie along the layers.

## Fork with pressed steel pins (2026-10-09)
- **Pins:** two MISUMI MSH2-30 (2 mm h7, 30 mm, 440C), pressed into 1.9 mm through holes. Each pin ends flush with the back of a 5 mm boss behind the plate, so 6 mm of pin is held and 24 mm is exposed.
- **Shorter tines:** 30 mm can't reach through the 25 mm brick and still leave a press depth. The tips now stop 6 mm short of the far face, which leaves 19 mm in the bore, 6.5 mm past the brick's centre of mass.
- **Fixture change:** the plate sits 5 mm from the brick (gap 5), which needs 3 mm tine guides on the fixture instead of 6 mm (`configs/fly-msh2-30.json`). With the 6 mm guides and gap 8, the tines reached only 16 mm into the bore. The brick hung at 3.3° and one pick in 40 lost the brick with a 154 N contact spike.
- **Simulation:** 40/40 seated (`runs/exp2-msh2-30-single-final`). The brick hangs at 0.46°. Payload is 20.0 g (gripper 6.3 g with pins), T/W 1.81 on datasheet thrust. The ~5.7 mm slide along the tines is unchanged.
- **Files:** `docs/exp2/gripper-fork-msh2-30/` holds `gripper.stl` (4.7 g PLA), `press_coupon.stl` (1.8–2.0 mm holes, notch at 1.8, 6 mm deep), `gripper_rods.step` (pin positions) and `preview.png`.
- **Not modelled:** the real pin end has only a ~0.27 mm lead-in (15° chamfer or R2 dome), while the sim's capsule tines have a 1 mm hemispherical one.

## Real test flight with the overhead RealSense (2026-10-09)
- **New package:** `src/testflight/`, with an operator guide in its README. It covers AprilTag
  (25h9, ids 0–5) calibration into the Lighthouse frame, colour + depth sensing of the pink A/B brick
  and the pink platform, and a field plan in which `base0` comes from the camera.
- **New flags:** `fly --realsense rig.json`, plus `testflight survey|calibrate|sense|plan|demo|lighthouse`.
- **Bricks:** the A/B pair (`configs/ab.json`). `compile_plan` now handles `lean_mode: ab`; the MuJoCo
  sim still doesn't.
- **Validation is synthetic only:** brick within 1.6 mm and 1.3°, platform within 1.7 mm, and the full
  mission seats the brick against a fake Crazyflie. Real-camera stage 0 is next.
- **Lighthouse geometry** (`testflight_lighthouse.yaml`): there are no xyz-space or verification
  samples yet, and the scale rests on the 1 m x-axis placement.

## Before the first real flight
- Review the `ctrlMel` gains and the payload-mass switching. Try the gains first in a hover with no payload.
- Lay out the base at the origin, the fixture at y = −0.30 m, and the launch stand at (−0.25, −0.30) m. Transit height is 0.15 m.
- Measure the Lighthouse bias in that layout; the sim currently assumes none.

## Still to do (spec phases)
- **Phase 4.** The other four gripper families (U-cradle, hook and eye, magnet, snap). The full planner: 3D swept clearance, downwash cost, beam search, battery swaps. Fitting the measured error model and rerunning the old drop Monte Carlo. Ideal vs simulated drone on tower, wall, bridge and overhang±.
- **Phase 5.** NSGA over gripper (then gripper and brick together). Stop at checkpoint 3 with smoke-run timing first.
- **Phase 6.** The brick scale study, U from 18 to 45 mm.
- **Phase 7.** The README "Experiment 2" section.
- **Open issue.** The brick slides about +6 mm along the tines; a detent or a different engagement may fix it.

## Key commands
```bash
cd brixzle
uv run brixzle fly-assemble tower --bricks 1 --seeds 40 --workers 10 --video --out runs/<new>
uv run brixzle gripper export --out docs/exp2/<new>
uv run fly <plan.json> --sim                 # or: --uri radio://0/80/2M/E7E7E7E7E7 [--show-calls]
uv run pytest -q                             # brixzle
uv run --project brixzle python -m unittest discover -s ../flight/tests
```
