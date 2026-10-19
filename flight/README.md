# Flight: calibration, hover and AprilTag checks

Hardware preset: Crazyflie 2.1 Brushless, Lighthouse deck, two Lighthouse v2 base stations,
Crazyradio, overhead RealSense D455 and the passive steel-pin fork. This package calibrates
the Lighthouse and camera coordinate systems from four drone placements, then flies
supervised hover and tag-visit checks. The earlier plan.json mission is [at the end](#earlier-planjson-mission-fly).

Run every command from the **repository root**, so `runs/` is shared. The commands below use
this prefix, shortened to `flight-cal`:

```sh
uv run --project flight --extra vision --extra realsense --extra detector flight-cal ...
```

Model weights and run archives are git-ignored and stay local.

## Quick start

```sh
flight-cal setup                                   # four placements; writes runs/setup
flight-cal hover-check --setup runs/setup --demo           # dry run: route only, no devices opened
flight-cal hover-check --setup runs/setup --demo --check-only   # camera + radio preflight, never arms
flight-cal hover-check --setup runs/setup --demo --arm     # supervised 10 s hover
flight-cal tag-visit --setup runs/setup --demo --arm       # supervised visits beside every tag
```

Setup is not reusable across sessions in practice: recalibrate before a test run, or whenever
the preflight reports no stable Lighthouse position. Each command writes to `--out`
(default `runs/<command>`, `runs/setup` for setup) and **replaces the previous run there**. Only
a directory that already holds a session report is replaced.

## 1. Setup

Prerequisites: [prepare the detector](#drone-detector) once, power the drone and both stations,
connect the Crazyradio and the D455, and close cfclient so it releases the radio. Keep the
drone disarmed. Both stations and the camera must see the placements; keep hands clear of the
top deck. Tags 0, 1, 2 and at least two other AprilTag **25h9** tags must be visible, and the
tags must stay fixed relative to the base stations.

```sh
flight-cal setup
```

The command asks **four questions**, each answered with Enter:

1. Put the drone level on its support at the origin, nose along the intended +X.
2. Move the same support and drone exactly **1 metre along +X**, at the same height and
   heading. Measure this distance; the camera does not define the metre.
3. Place it elsewhere on the same plane, away from the X-axis, keeping height and heading.
   This defines the XY plane, as in cfclient.
4. Return it to the original launch support and heading.

There are no extra geometry or verification samples. Camera registration is a rigid fit of the
three depth-derived reference points to their Lighthouse positions, so setup requires the D455
in RGB-D mode. Reprojection RMS must be at most 1 px; set `--max-camera-rms-px 2.0` (or higher)
to accept a looser fit, which is recorded in the reports. The return to launch must be within
2 cm of the origin observation.

Setup discovers the radio URI (several radios need `--uri`), exactly two live Lighthouse
stations, the D455 and the drone. Station discovery uses fresh `bsReceive`/`bsCalCon`
bitmasks: each station must appear in at least three samples in a rolling 0.75 s window, with
the two-station set stable for one second (history in `station-discovery.json`). The D455
profile follows the USB link (640×480 at 15 fps on USB 2; prefer a USB 3 port and cable).

At each placement, setup waits up to ten seconds for three consecutive drone detections within
3 px at confidence ≥ 0.40. Failed captures save `drone-*-last.png` and
`drone-*-visibility.json`. Keep all placements fully inside the camera view: detections that
touch the image edge are rejected. Reference placements need not be centred over the tags.

The D455 infrared projector is switched off while the camera is open (it prevented Lighthouse
collection; depth uses passive stereo and needs scene texture) and restored on close.

Output (`runs/setup`): tag layout (`layout.json`, `floor.png`), intrinsics, paired
camera/radio samples under `lighthouse/`, the resolved `rig.json`, and the
Lighthouse and camera reports. Setup never arms or flies and does **not** upload geometry; the
first live flight command does.

**Frozen camera images.** If all reference images are byte-identical, the fit is invalid. Restart
the camera connection (USB 3 if possible) and run setup again; raising the RMS limit cannot
repair it. The camera adapter rejects repeated frame numbers and content, checks color/depth
timing, and records frame numbers, timestamps and image hashes per observation.

## 2. Flights

Both flight commands take `--setup DIR`, which fills in the calibration and camera settings.

| Mode | Command flags | Effect |
|---|---|---|
| Route preview (default) | none | Writes `route.json` and `route-preview.png`; opens no devices |
| Camera check | `--camera-check-only` | Camera registration only; drone may be off; no radio |
| Preflight | `--check-only` | Camera, radio, geometry upload/readback, estimator, battery, launch checks; never arms |
| Flight | `--arm` | Preflight, then supervised flight |

`hover-check` takes off vertically, holds over launch for ten seconds (`--hover-seconds`,
greater than 0 and at most 60) and lands. `tag-visit` holds five seconds beside every tag, then
returns and lands. Both keep at least 20 cm between the gripper and the tag plane.

```sh
flight-cal hover-check --setup runs/setup --demo --camera-check-only
flight-cal tag-visit --setup runs/setup --demo           # inspect runs/tag-visit/route-preview.png
flight-cal tag-visit --setup runs/setup --demo --check-only
flight-cal tag-visit --setup runs/setup --demo --arm
```

Before every flight, return the drone to its original launch position and heading and clear the
area. Ctrl+C or any exception hands back to the high-level commander and lands in place.

**Camera pose refresh (`--demo`).** Before connecting the radio, `--demo` captures ten stable
views of at least three spread-out tags and refits the camera pose to their saved Lighthouse
coordinates, writing `camera-registration.json`. Overall RMS must be at most 2 px and each
held-out tag at most 3 px. It cannot detect the whole tag map moving relative to the stations.
`--demo` also selects the firmware PID controller, 10 cm tolerances, peak speed 0.2 m/s and
acceleration 0.3 m/s², and Lighthouse sweep-angle estimation (`lighthouse.method=1`, which works
with one visible station). Without `--demo`, flights use the saved camera pose (which must still match the
view) and Mellinger (peak 0.1 m/s, 0.2 m/s²); the commands in this guide use `--demo`. `--controller pid|mellinger` overrides the controller either way.

**Tag visits.** Each visit starts with a 15 cm horizontal offset toward the geofence centre
(`--tag-offset-m` changes this minimum). The planner projects the drone/gripper envelope,
enlarged by 2 cm, through the camera model and rejects any overlap with a tag plus a 4 px
border, increasing the offset in 1 cm steps up to 40 cm. Routes without a clear waypoint are
rejected. `route.json` records the actual offsets; `route-preview.png` shows predicted footprints.
This predicts visibility at nominal dwell positions only; `tag-visibility.json` records actual
detections during completed dwells.

**Safety checks.** Lighthouse positions control the flight; the camera and YOLO observations
are diagnostics (in `--demo` they are advisory; otherwise visual disagreement above 5 cm stops
the flight). Fresh radio telemetry, estimator convergence, geometry readback, workspace limits,
attitude/heading limits (20° in demo) and explicit arming always apply. Losing all stations
for more than 0.5 s ends the flight. Demo flights land after voltage stays below 3.35 V for one
second or immediately below 3.1 V; preflight requires 3.8 V at rest (thresholds are recorded in
`flight_battery_policy`). A lost radio link prevents host commands, so keep a person ready to
intervene.

**Geometry upload.** The first live command uploads the setup's geometry and reads it back.
The uploader detects the supported station slots (four on this drone), preserves factory
calibration and verifies persistence; the result is in `lighthouse_upload`.

## Drone detector

The prepared YOLOE-L model (`runs/models/drone-yoloe-l.onnx`) is preferred automatically;
nano (`runs/models/drone-nano.onnx`) is the fallback, or pass `--detector-model FILE` to setup.
Visual prompts show this drone at the origin and +X positions (listed in
`flight/configs/drone-reference.json`). Runtime inference uses ONNX Runtime on CPU, without
PyTorch or a GPU.

Prepare it once on a fresh checkout, from the repository root with Python 3.11 or newer
(model preparation installs PyTorch and needs network access, disk space and time):

```sh
uv run --project flight --python 3.12 --extra vision --extra detector --extra yolo \
  flight-cal prepare-detector --size l --image-size 640
```

This uses the pinned Ultralytics package to download the official `yoloe-26l-seg.pt`
checkpoint when absent, computes visual prompts from the three checked-in examples
(`flight/configs/drone-reference.json`) and exports the drone-specific detector. No radio or
camera is opened. The outputs, all git-ignored, are:

- `runs/models/yoloe-26l-seg.pt`: downloaded base weights, about 76 MiB.
- `runs/models/drone-yoloe-l.onnx`: prompted detector, about 107 MiB.
- `runs/models/drone-yoloe-l.json`: reference metadata and export SHA-256.

To download the base weights explicitly (the official
[Ultralytics release asset](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt),
SHA-256 `a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9` on the original
workstation), then run the same command:

```sh
mkdir -p runs/models
curl --fail --location --retry 3 \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt \
  --output runs/models/yoloe-26l-seg.pt
```

The exported ONNX fingerprint can depend on export dependencies; it is recorded in the JSON
metadata. For another drone or a substantially different view, update the reference images and
boxes before rebuilding. A new checkout also needs its own physical `flight-cal setup`; public
weights cannot recreate calibration.

A detection box estimates the drone body's
centre, not the Lighthouse sensors; confidence does not establish millimetre accuracy, and
occlusion, low confidence or several plausible drones stop the workflow.

## Acceptance and assumptions

- Lighthouse estimation must converge with valid reference placements. Station geometry is
  scaled by the measured 1 m X placement.
- Camera registration must meet the RMS limit above. Tags are reconstructed from the camera
  floor plane (depth) and transformed to the Lighthouse frame; they are estimates, not
  independently surveyed ground truth.
- Mass and envelope come from the CF21B/fork CAD preset, including the steel pins, with the
  sensor centre assumed 20 mm above the body frame. These are recorded assumptions; a modified
  build needs an updated preset (`flight/calibration/presets.py`). Fork exports are in
  `brixzle/docs/exp2/gripper-fork-msh2-30/`.
- The workspace is the tags and launch position plus an 18 cm XY margin. It assumes a clear
  area and does not detect obstacles or prove the landing surface is suitable. Review
  `rig.json` and `floor.png` before flying.

## Other commands

```sh
flight-cal discover                  # read-only: Crazyradio scan and Lighthouse stations
flight-cal devices                   # connected RealSense cameras
flight-cal estimate-tags --out DIR   # capture an initial floor-tag layout
flight-cal replay-lighthouse --samples samples.json --out DIR   # offline re-solve; no radio
```

The earlier manual workflow remains: `setup --advanced` (measurement questionnaire with
`--layout` and `--measure-check-tags`), and the separate `lighthouse` and `camera` commands.
Normal setup does not use them. The initial D455 tag estimates are in
`flight/configs/tag-layout-overhead.json`.

## Tests

```sh
uv run --project flight --extra test pytest flight/tests -q
uv run --project brixzle --extra test pytest brixzle/tests/test_flight_parity.py -q
```

cflib is mocked; no test opens a radio.

## Earlier plan.json mission (`fly`)

One flight program for the Brixzle pick-and-place experiment. The mission (`flight/mission.py`)
talks only to the cflib 0.1.33 API, an injected clock and a camera pose source, so the same
`plan.json` flies in the MuJoCo simulator or on a real Crazyflie 2.1 Brushless. This package does
not import CadQuery or MuJoCo; `--sim` imports `brixzle` lazily. It is the passive-gripper
mission inherited from the parent branch and is separate from the calibration flights above.

```sh
# Compile a plan (brixzle environment)
(cd brixzle && uv run brixzle fly-assemble tower --bricks 1 --out runs/plan-single)

# Simulator (needs the brixzle environment: MuJoCo and CadQuery)
uv run --project brixzle fly runs/plan-single/plan.json --sim [--seed 0] [--out runs/fly-sim --video]

# Real hardware: dry run (default), offline checks and the cflib calls it would send; no radio
uv run --project flight fly runs/plan-single/plan.json --uri radio://0/80/2M/E7E7E7E7E7 [--show-calls]
# Connect and run the preflight only (never arms)
uv run --project flight fly runs/plan-single/plan.json --uri radio://0/80/2M/E7E7E7E7E7 --check-only
# Supervised flight only
uv run --project flight fly runs/plan-single/plan.json --uri radio://0/80/2M/E7E7E7E7E7 --arm --poses poses.json
```

- **Preflight** (`flight/preflight.py`): acknowledged param writes, Lighthouse deck
  (`deck.bcLighthouse4`), Kalman reset and variance convergence, `pm.vbat` ≥ 3.8 V, drone resting at
  the plan's home pose, inside the geofence.
- **Controller**: `stabilizer.controller = 2` (Mellinger), matching the simulator's crazyflow
  port. The plan also sets `ctrlMel.ki_m_xy` / `i_range_m_xy` and `ctrlMel.mass` after each pick
  and release. **Review these values before the first flight.**
- **Offline checks** (`flight/checks.py`): every waypoint inside the geofence, peak speed and
  acceleration of each firmware `go_to` piece and streamed segment, setpoint rate ≥ 10 Hz
  (watchdog: 0.5 s level, 2 s off).
- **Camera**: the RealSense pose source is not implemented (`RealSensePoseSource` raises);
  `--poses poses.json` is a manual fallback `{name: {"pos": [m], "quat": [w, x, y, z]}}`, re-read
  before every pick and check.

Not built: CrazySim (https://github.com/gtfactslab/CrazySim) runs the real firmware in the loop
and accepts cflib over `udp://127.0.0.1:19850` (Ubuntu only).

References: Bitcraze [high-level commander](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/high_level_commander/),
[Lighthouse positioning methods](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/lighthouse/positioning_methods/),
[controllers](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/controllers/),
[cfclient Lighthouse tab](https://github.com/bitcraze/crazyflie-clients-python/blob/master/src/cfclient/ui/tabs/lighthouse_tab.py),
[YOLOE visual prompting](https://docs.ultralytics.com/models/yoloe/).
