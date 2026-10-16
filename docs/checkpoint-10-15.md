# Checkpoint — 2026-10-15

## Working state

Experiment 2 now completes a supervised flight beside all five AprilTags, keeping
the tags visible from the overhead camera. Calibration, stationary hovering, and
tag visits are separate CLI operations. The current commands do not perform
gripping, transport, or assembly.

- Branch: `brixzle-exp2-flight-calibration`, based on the passive-gripper branch.
- This document describes the current working tree, including uncommitted code.
- Working setup: [`runs/exp2-setup37`](../runs/exp2-setup37/).
- Successful flight: [`runs/exp2-tag-visit42`](../runs/exp2-tag-visit42/).
- The operator confirmed that the flight and landing looked good.
- Setup37 deliberately skipped extra geometry and verification samples. Its
  calibration remains **unverified**, even though the demonstration succeeded.

Run-directory links in this document refer to local evidence on the original
workstation. Those archives and the video are not included in the GitHub branch.

The flight ran on October 15 in America/New_York. JSON timestamps use UTC and
therefore show October 16 for this evening session.

## RealSense flight replay

**[Open the RealSense snapshot replay](../runs/exp2-tag-visit42/realsense-snapshot-replay.mp4)**

This is an approximately 81-second replay assembled from **69 saved D455 color
images**, including the preflight view, flight, and landing. The flight itself
lasted approximately 68 seconds. Existing green detector boxes are preserved.

The flight code did **not** record continuous video or a RealSense `.bag` file.
It saved a PNG after more than one second had elapsed since the previous saved
image; the median interval in this run was 1.159 seconds. The MP4 repeats these
stills in a 10 fps container, with a short extra hold on the last frame. It does
not contain additional captured motion, audio, or a depth-video stream.

Timing uses the logged camera capture timestamps for 62 images. Seven images
were saved without their status reaching the main process; their times are
estimated from neighbouring snapshots. These are predominantly preflight images
and the final image. Per-image provenance is in
[`realsense-snapshot-replay.json`](../runs/exp2-tag-visit42/realsense-snapshot-replay.json).
The original `hover-0000.png` through `hover-0068.png` remain in the run directory.
The `hover-` image prefix is shared by both flight commands.

To rebuild this derived replay under a new filename, using the saved timing list:

```sh
ffmpeg -n -f concat -safe 0 \
  -i runs/exp2-tag-visit42/realsense-snapshot-replay.ffconcat \
  -an -c:v libopenh264 -b:v 3000k -pix_fmt yuv420p \
  -r 10 -fps_mode cfr -movflags +faststart \
  runs/exp2-tag-visit42/realsense-snapshot-replay-copy.mp4
```

Replay creation was an offline export for this checkpoint. The flight CLI still
saves snapshots rather than automatically writing an MP4.

## Hardware and dependencies

| Component | Current configuration |
| --- | --- |
| Drone | Crazyflie 2.1 Brushless with Lighthouse deck |
| Gripper | Passive MSH2-30 steel-pin fork, unloaded |
| Positioning | Two Lighthouse v2 stations, channels 1 and 2 |
| Radio | Crazyradio; this setup uses `radio://0/80/2M` |
| Camera | Overhead Intel RealSense D455, serial `262822301178` |
| Tags | AprilTag 25h9, IDs 0–4 |
| Successful capture profile | 1280×720 color, 848×480 depth, 30 fps |
| Detector | Visually prompted YOLOE-26L, 640 px input, ONNX Runtime on CPU |
| Demo controller | Firmware PID, `stabilizer.controller=1` |

The Python package is [`flight`](../flight/pyproject.toml). `flight-cal` resolves to
`flight.calibration.cli:main`. Core dependencies include pinned `cflib==0.1.33`,
NumPy, and SciPy. Extras add OpenCV (`vision`), `pyrealsense2` (`realsense`), and
ONNX Runtime (`detector`). The separate `yolo` extra is needed for model preparation,
not ordinary inference with the exported model.

The prepared model is `runs/models/drone-yoloe-l.onnx`, with metadata beside it.
Preparation uses the example images and boxes in `flight/configs/drone-reference.json`
to create visual prompts. The class named `NanoDetector` now supports larger
YOLOE exports too; its name does not mean the current runtime model is nano.

Hardware dimensions come from [`presets.py`](../flight/flight/calibration/presets.py),
the drone meshes, and
[`gripper.stl`](../brixzle/docs/exp2/gripper-fork-msh2-30/gripper.stl).
The preset assumes a 20 mm offset from body reference to Lighthouse sample point
and estimates loaded mass at 52.42 g from the existing hardware/CAD configuration.
These are assumptions, not measurements made during setup. PID does not use the
host's `ctrlMel.mass` override.

## Download and rebuild the drone detector

The GitHub branch contains the source code, pinned dependencies, three small
reference JPEGs, and `flight/configs/drone-reference.json`. It excludes the
downloaded `.pt` weights, generated `.onnx` models, run captures, and videos.
Git LFS is not needed for this branch.

From the repository root, using Python 3.11 or newer (3.12 shown):

```sh
uv run --project flight --python 3.12 --extra vision --extra detector --extra yolo \
  flight-cal prepare-detector --size l --image-size 640
```

This uses the pinned Ultralytics package to download the official
`yoloe-26l-seg.pt` checkpoint when absent. It then computes visual prompts from
the three checked-in examples and exports the drone-specific detector. No radio
or camera is opened. The outputs are:

- `runs/models/yoloe-26l-seg.pt`: downloaded base weights, approximately 76 MiB.
- `runs/models/drone-yoloe-l.onnx`: prompted detector, approximately 107 MiB.
- `runs/models/drone-yoloe-l.json`: reference metadata and export SHA-256.

An intermediate `yoloe-26l-seg.onnx` can also be produced during export. Normal
flight uses ONNX Runtime and does not require the `yolo` extra; model preparation
installs PyTorch and needs network access, disk space, and time for the export.
The exported drone-specific ONNX file is generated locally, rather than downloaded
from a separate project-hosted URL.

For an explicit base-weight download, use the official
[Ultralytics release asset](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt),
then run the same preparation command:

```sh
mkdir -p runs/models
curl --fail --location --retry 3 \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt \
  --output runs/models/yoloe-26l-seg.pt
```

The base checkpoint used on this workstation had SHA-256
`a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9`.
The generated ONNX fingerprint may depend on export dependencies; preparation
records the actual fingerprint in its JSON metadata. See the
[official YOLOE guide](https://docs.ultralytics.com/models/yoloe/) for visual prompts.

If using another drone or a substantially different view, update the reference
images and their boxes before rebuilding. A new checkout also needs a fresh
physical `flight-cal setup`: the calibration archives and local video cannot be
recreated by downloading the public model weights.

## Commands to use

Run from the repository root. Use a **new output directory** for every invocation.
The drone starts on its original support, at the original heading, with the flight
area clear. Keep tags fixed relative to the Lighthouse stations, and close cfclient
so it releases the radio.

### 1. Stationary sanity check

```sh
uv run --project flight --extra vision --extra realsense --extra detector \
  flight-cal hover-check --setup runs/exp2-setup37 --demo --arm \
  --out runs/exp2-hover-next
```

`hover-check` takes off vertically, holds over launch for **10 seconds**, then
lands. It does not visit the tags. `--hover-seconds` changes the hold duration
(greater than zero, at most 60 seconds).

### 2. Separate tag-visit flight

```sh
uv run --project flight --extra vision --extra realsense --extra detector \
  flight-cal tag-visit --setup runs/exp2-setup37 --demo --arm \
  --out runs/exp2-tags-next
```

`tag-visit` checks its takeoff hold, visits beside each tag for five seconds,
returns to launch, and lands. Its minimum inward offset is 15 cm; use
`--tag-offset-m` to change that minimum. The planner can enlarge it to keep tags
clear in the camera projection, as described below.

Both commands support these execution modes:

| Mode | Behaviour |
| --- | --- |
| No mode flag | Offline route preview; opens no devices |
| `--camera-check-only` | Opens the camera and checks registration; no radio |
| `--check-only` | Camera, geometry, radio, estimator, battery, and launch checks; no arming |
| `--arm` | Preflight followed by actual flight |

`--demo` includes permission to use an explicitly unverified but successfully
fitted setup. It selects the demo controller and tolerances; it does not relabel
the original calibration as verified. `--allow-unverified` alone permits that
calibration without enabling the other demo changes. `--controller pid` or
`--controller mellinger` overrides controller selection explicitly.

## How calibration works

The compact setup is in [`setup.py`](../flight/flight/calibration/setup.py).
To reproduce the three-reference workflow used for setup37:

```sh
uv run --project flight --extra vision --extra realsense --extra detector \
  flight-cal setup --skip-extra-samples --max-camera-rms-px 10.0 \
  --out runs/exp2-setup-next
```

1. **Discover camera and tags.** Prefer the connected D455. Warm up the camera,
   then collect 20 usable frames showing IDs 0, 1, 2 and at least two other tags.
   Median tag observations and passive stereo depth establish an initial floor
   layout and metric scale.
2. **Discover radio and stations.** Scan for one Crazyflie unless `--uri` is given.
   Check the Lighthouse deck and v2 configuration. Require two stations with
   repeated fresh `bsReceive` and `bsCalCon` observations, rather than relying on
   cached availability. IDs in files are zero-based; displayed channels are one-based.
3. **Prompt for origin.** Drone level on its support, nose along intended +X.
4. **Prompt for +X.** Move the same support exactly one metre along +X, maintaining
   height and heading. The operator supplies this measured distance.
5. **Prompt for XY.** Move to another point on the same plane, away from the X-axis.
6. **Fit both coordinate systems.** At every placement, stable visual detections
   bracket a Lighthouse sweep collection. cflib estimates Lighthouse geometry
   and reference poses. With all extra samples skipped, three depth-derived
   camera points are aligned to those Lighthouse points using a rigid SVD fit.
   Camera reprojection RMS must pass the chosen limit.
7. **Prompt to return to launch.** With depth available, require the observed
   return to be within 2 cm of the original 3D observation. RGB-only uses a 5 px
   check. Save the completed rig and calibration artifacts.

Those are **four placement prompts**. Full setup additionally collects eight
geometry positions and five independent verification positions via cfclient-style
rotate-and-hold gestures. `--skip-verification` omits the five checks;
`--skip-extra-samples` omits both sets and is supported only with RGB-D depth.

The default camera fit limit is 1 px. Setup37 used the explicit 10 px override
and actually fitted at **2.015 px RMS**. Its launch-return observation was 1.51 cm
from origin. The setup's skipped verification status is retained in all reports.
Reference-only setup does not upload geometry; a later authorized live flight
mode uploads and reads it back.

Full Lighthouse acceptance requires maximum held-out error ≤0.4 mm with sample
coverage. This gate is skipped in the demonstrated reference-only workflow.
The successful flight does not establish submillimetre calibration accuracy.

### RGB-D, RGB, and the infrared emitter

`--camera-mode rgbd` uses depth for initial metric layout and reference alignment.
`--camera-mode rgb` uses image bearings; an OpenCV camera additionally needs
intrinsics for its capture profile and a measured tag size. RGB-only setup needs
the extra geometry placements because the three-reference depth alignment is
unavailable. These are separate camera backends; macOS compatibility depends on
the accessible camera and installed backend.

For the D455, auto setup checks the negotiated USB connection. USB 2 uses a lower
bandwidth 640×480/15 fps profile. USB-C describes the connector and does not by
itself establish the negotiated USB speed.

The D455 infrared projector is disabled while RGB-D capture is open because it
interfered with Lighthouse reception in this setup. Depth uses passive stereo.
The adapter restores the prior emitter setting on close. It also checks frame
freshness and color/depth timing, and copies frames before alignment to avoid
reusing camera buffers.

## Frames and camera registration before flight

Positions are in metres; quaternions use `[w, x, y, z]`.
`T_lighthouse_camera` maps camera-coordinate points into the Lighthouse world.
The origin placement defines the Lighthouse sample point at `[0, 0, 0]`.
Body targets are converted to sample-point commands using `body_to_sample_m`;
the current route uses zero yaw. The body launch reference is consequently at
`[0, 0, -0.02]` for this preset.

In demo mode, the camera worker collects ten stable views of at least three
spatially spread fixed tags. [`registration.py`](../flight/flight/calibration/registration.py)
fits the camera pose to their saved Lighthouse coordinates using current camera
intrinsics. It checks both overall reprojection RMS (≤2 px) and fits that leave
each tag out (worst held-out RMS ≤3 px). It preserves the world tag map and writes
the refreshed camera pose only into the new run.

This corrects a camera registration discrepancy without repeating the placement
procedure. It assumes the tags remain fixed relative to the stations; it cannot
detect the entire tag map moving together relative to Lighthouse.

## How tag visits avoid occlusion

[`visits.py`](../flight/flight/calibration/visits.py) builds the tag-specific route:

1. Choose a common altitude with the gripper bottom at least 20 cm above the
   highest saved tag centre.
2. For each tag, point horizontally toward the XY centre of the configured geofence.
3. Start at the requested minimum offset, normally 15 cm.
4. Project the drone/gripper envelope, inflated by 2 cm, into the camera image.
   Reject positions outside the workspace or image, or overlapping any tag with
   its extra 4 px border.
5. Increase the offset in 1 cm increments until a position clears every tag.
   Automatic extension is capped at 40 cm, or at the requested minimum if larger.
   A tag near the centre may require continuing across the centre. If no position
   fits, the command fails before arming.
6. In demo mode, visit the nearest remaining waypoint first. Hold five seconds
   per tag, then return and land.

Live runs calculate these waypoints after camera registration. Offline previews
use the saved pose. `route.json` records the actual offsets and targets;
`route-preview.png` shows predicted drone footprints in camera coordinates.
The projection checks nominal dwell positions. Transitions and tracking errors
can still obscure tags, so actual detections are logged too.

## Flight execution and monitoring

Both commands share [`hover.py`](../flight/flight/calibration/hover.py). Preflight
checks artifact fingerprints, geometry, the deck, estimator convergence, battery,
launch position, and workspace. The uploader queries actual Lighthouse memory
capacity—four slots on this drone—and verifies the written geometry. It preserves
factory calibration and does not attempt the unsupported 16-slot write that
previously failed.

[`preflight.py`](../flight/flight/preflight.py) selects Kalman estimation, enables
the high-level commander, and selects the requested controller. Demo uses PID;
strict mode defaults to Mellinger with restored firmware gains and the preset mass.
The controller runs onboard; Python sends high-level `takeoff`, `go_to`, and
`land` commands. Trajectory durations are increased until sampled seventh-order
profiles satisfy the configured speed and acceleration limits.

| Check | Demo | Strict |
| --- | --- | --- |
| Resting battery before takeoff | ≥3.8 V | ≥3.8 V |
| In-flight battery | Land below 3.35 V for 1 s; immediate below 3.1 V | Immediate below 3.5 V |
| Settled target error | ≤10 cm | ≤5 cm |
| Initial launch tolerance | 10 cm horizontally and vertically | 3 cm horizontally and vertically |
| Roll, pitch, heading limits | 20° | 10° |
| Planned peak speed / acceleration | 0.2 m/s / 0.3 m/s² | 0.1 m/s / 0.2 m/s² |
| Lighthouse reception | At least one configured station within 0.5 s | Both configured stations active |
| Visual disagreement during flight | Logged as advisory | Can terminate flight |

Position logs arrive at 50 ms intervals and health logs at 100 ms. The host checks
them about every 50 ms; telemetry older than 0.5 seconds fails the check. The
rotated drone/gripper envelope, plus 2 cm margin, must stay inside the geofence.
The camera runs in a separate process so blocked image processing cannot block
flight cleanup. Demo visual tracking uses the drone's image bearing, avoiding
unreliable propeller/background depth. Lighthouse controls the flight.

Settled target checks begin 0.5 seconds after the planned movement duration.
On an exception while airborne, the executor requests an in-place landing when
the radio is still connected, then sends stop and disarm. Normal completion also
sends stop and disarm. The final operator question records a visual observation
after flight, not permission to begin the already authorized flight.

## Successful flight evidence

Run42 used setup37, `tag-visit --demo --arm`, PID, and a live camera refresh.

| Measurement | Recorded result |
| --- | --- |
| Route | Takeoff → 0 → 3 → 2 → 4 → 1 → return → land |
| Logged execution duration | 68.00 s |
| Commanded Lighthouse sample-point hover Z | 0.25271 m in the origin frame |
| Body reference hover Z | 0.23271 m |
| Camera RMS before / after refresh | 1.094 / 0.489 px |
| Worst held-out tag reprojection RMS | 0.730 px |
| Preflight / minimum logged flight battery | 3.959 / 3.611 V |
| Largest settled tag-visit position error | 6.67 cm |
| Largest settled takeoff position error | 8.50 cm |
| Visibility during the five settled tag holds | All five tags detected in 65/65 logged camera observations |
| Operator result | Yes; flight and landing looked good |

Actual inward offsets were **34 cm (tag 0), 31 cm (tag 3), 22 cm (tag 2),
26 cm (tag 4), and 25 cm (tag 1)**. Fifteen centimetres was insufficient in the
projection for this camera geometry.

The previous Mellinger attempt, run39, rose about 20 cm above its target and
aborted before visiting tags. PID completed run42 within the demo tolerances.
This comparison does not prove that mass estimation was the sole cause of the
previous error.

The 65 observations above are detector results logged during dwells; they are
distinct from the 69 PNG snapshots saved across the whole run. Visibility of the
tags and agreement with Lighthouse targets do not constitute an independent
measurement of the drone's absolute 3D accuracy. Run42 is an accepted demonstration
with `unverified_calibration: true`. No new automated test suite was run for this
documentation checkpoint.

## Code and artifact map

| File | Responsibility |
| --- | --- |
| [`cli.py`](../flight/flight/calibration/cli.py) | Commands, flags, setup loading, dispatch |
| [`setup.py`](../flight/flight/calibration/setup.py) | Four prompts, paired samples, alignment, completed setup |
| [`discovery.py`](../flight/flight/calibration/discovery.py) | Crazyradio discovery and fresh station reception |
| [`lighthouse.py`](../flight/flight/calibration/lighthouse.py) | Sweep collection, cflib geometry fit, verification, upload/readback |
| [`camera.py`](../flight/flight/calibration/camera.py) | D455/OpenCV adapters, timing, distortion, intrinsics |
| [`layout.py`](../flight/flight/calibration/layout.py), [`vision.py`](../flight/flight/calibration/vision.py) | AprilTag detection and camera/tag geometry |
| [`drone.py`](../flight/flight/calibration/drone.py) | Visual-prompt model export and ONNX drone observations |
| [`registration.py`](../flight/flight/calibration/registration.py) | Live camera pose fit against fixed tags |
| [`visits.py`](../flight/flight/calibration/visits.py) | Inward offsets, projected visibility, route preview |
| [`hover.py`](../flight/flight/calibration/hover.py) | Stationary route, shared trajectory/executor, monitors, visibility report |
| [`preflight.py`](../flight/flight/preflight.py) | Acknowledged parameters, logs, estimator and launch checks |
| [`artifacts.py`](../flight/flight/calibration/artifacts.py) | JSON sessions, reports, provenance fingerprints |

Useful run42 artifacts:

- [`report.json`](../runs/exp2-tag-visit42/report.json): final result and selected policies.
- [`route.json`](../runs/exp2-tag-visit42/route.json): all commands, durations, offsets, and projected footprints.
- [`route-preview.png`](../runs/exp2-tag-visit42/route-preview.png): camera-coordinate planning diagram.
- [`events.jsonl`](../runs/exp2-tag-visit42/events.jsonl): targets, telemetry, camera observations, tracking errors.
- [`camera-registration.json`](../runs/exp2-tag-visit42/camera-registration.json): refreshed transform and residuals.
- [`tag-visibility.json`](../runs/exp2-tag-visit42/tag-visibility.json): per-hold detection counts.
- [`flight-summary.json`](../runs/exp2-tag-visit42/flight-summary.json): offline summary of duration, battery, and errors.
- [`realsense-snapshot-replay.mp4`](../runs/exp2-tag-visit42/realsense-snapshot-replay.mp4): derived color-camera replay.

`runs/` and `flight/runs/` remain local and ignored, including setup37, run42,
the replay, earlier attempts, and prepared detector models. Python environments,
runtime caches, and generated bytecode are also ignored. Preserve experiment
archives separately if needed; no public download is provided for the local
flight recordings. Base detector weights can be downloaded and the prompted
model rebuilt using the instructions above.

Saved calibration reports include local model paths; when moving the checkout to
another machine, review those paths and use a fresh setup for its physical
arrangement. The preset geofence is derived from tag extents, not a measured
obstacle map, and the current planner assumes a clear interior.
