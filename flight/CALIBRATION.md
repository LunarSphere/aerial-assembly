# Experiment 2: four-prompt calibration and tag hover checks

Hardware preset: Crazyflie 2.1 Brushless, Lighthouse deck, two Lighthouse v2 base
stations, Crazyradio, overhead D455 and the passive steel-pin fork. This experiment
calibrates the two coordinate systems and supports stationary hover checks and
separate flights beside the tags.

## Stationary hover and separate tag visits

For a completed setup such as `runs/exp2-setup37`, use `--demo`. No new placement
questions are needed. Keep the tags fixed relative to the Lighthouse stations.

```sh
# Works with the drone switched off; checks only the camera.
uv run --project flight --extra vision --extra realsense --extra detector flight-cal hover-check \
  --setup runs/exp2-setup37 --demo --camera-check-only --out runs/exp2-camera-demo

# With the drone back at its original launch position and heading:
uv run --project flight --extra vision --extra realsense --extra detector flight-cal hover-check \
  --setup runs/exp2-setup37 --demo --arm --out runs/exp2-hover-demo

# Separate flight: five seconds beside each tag, then return and land.
uv run --project flight --extra vision --extra realsense --extra detector flight-cal tag-visit \
  --setup runs/exp2-setup37 --demo --arm --out runs/exp2-tag-visit
```

Use a new output directory each time. Omit `--arm` for an offline route preview;
use `--check-only` instead to check camera and radio without arming.

Demo mode includes `--allow-unverified`. Before connecting the radio, it captures
ten stable views of at least three spread-out tags and fits the camera pose to
their saved Lighthouse coordinates using the current camera intrinsics. Each tag is also predicted from a fit that
excludes it: overall RMS must be at most 2 px and each held-out tag at most 3 px.
It writes `camera-registration.json` in the new run; the original calibration and
tag map are preserved. A consistent camera/view offset no longer requires repeating
the human placements. This cannot detect the entire tag map moving relative to
the Lighthouse stations, and does not independently verify the original setup.

`hover-check` takes off vertically, holds over launch for ten seconds, and lands.
Set `--hover-seconds` to change the hold (greater than zero, at most 60 seconds).
It does not visit tags. `tag-visit` holds beside every tag for five seconds, then
returns and lands. Both keep at least 20 cm between the gripper and the tag plane
at hover height. Nearby tag waypoints are visited first in demo mode.

Tag visits start with a 15 cm horizontal offset toward the geofence centre.
`--tag-offset-m` changes this minimum. Using the camera pose and lens model, the
planner projects the drone/gripper envelope enlarged by 2 cm and rejects overlap
with any tag plus a 4 px border. It increases the offset in 1 cm steps, up to
40 cm (or the specified minimum if larger), and rejects routes without a clear
waypoint. A tag close to the centre may require continuing across the centre.
Live runs rebuild the route after camera registration; offline previews use the
saved calibration. `route.json` records actual offsets, and `route-preview.png`
shows predicted footprints in camera coordinates. This predicts visibility at
nominal dwell positions; transit, attitude and tracking error can still occlude
tags. `tag-visibility.json` records actual detections during completed dwells.

The drone follows Lighthouse coordinates using the firmware high-level commander.
Demo mode selects the firmware PID controller. The previous Mellinger run rose
about 20 cm above its takeoff target; its loaded mass was an estimate. PID avoids
that Mellinger mass setting, but still requires flight validation. Use
`--controller mellinger` to select the previous controller explicitly; strict
mode retains Mellinger by default. See Bitcraze's
[controller documentation](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/controllers/).
Demo peak speed is 0.2 m/s and acceleration 0.3 m/s². Home and settled target
tolerances are 10 cm. Attitude/heading limits are 20 degrees; the rotated gripper
envelope is checked against the configured workspace.

Demo mode selects Lighthouse sweep-angle estimation (`lighthouse.method=1`), which
can operate with one visible station. A loss of all configured stations for more
than 0.5 seconds ends the flight. Fresh radio telemetry, estimator convergence,
battery checks, geometry readback, workspace limits and explicit arming still apply.
Demo flight lands after voltage stays below 3.35 V for one second, or immediately
below 3.1 V. This tolerates motor-start sag (hover38 briefly read 3.48 V from a
4.10 V resting battery). The drone's read-only firmware settings were 3.35 V low
and 3.0 V critical; they are not modified. Preflight still requires 3.8 V at rest.
The selected host thresholds are recorded in `flight_battery_policy`.
YOLO supplies bearing diagnostics; depth on the drone is not required. Its detection
errors and visual position discrepancy are advisory. After arming, camera faults
are logged while flight continues under Lighthouse control; camera registration
must pass before arming. The result remains an unverified demonstration.

The implementation follows Bitcraze's [high-level commander](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/high_level_commander/),
[Lighthouse positioning methods](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/lighthouse/positioning_methods/),
and OpenCV's [PnP camera pose estimation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).

The live D455 camera-only check in `runs/exp2-demo-camera34` reduced the initial
projection discrepancy from 99.29 px to 0.53 px (worst held-out tag 0.77 px).
That establishes consistent registration to the saved tag map; it does not establish
why the old image mapping differed. The drone was off, so flight remains untested.

## Run setup

From the repository root:

```sh
uv run --project flight --extra vision --extra realsense --extra detector flight-cal setup --out runs/exp2-setup
```

Power the drone and both stations, connect Crazyradio and the camera, and close
cfclient so it releases the radio. Keep the drone disarmed. Both stations and the
camera must see the calibration positions; keep hands clear of the top deck.
Clear the flight area and use a stable launch support that leaves the fork free.

The command asks **four questions**, each answered with Enter:

1. Put the drone level on its support at the origin, nose along the intended +X.
2. Move the same support and drone exactly **1 metre along +X**, at the same height
   and heading. Measure this distance; camera estimates do not define the metre.
3. Place it elsewhere on the same plane, away from the X-axis, keeping that height
   and heading. This defines the XY plane, just like cfclient.
4. After sampling and solving, return it to the original launch support and heading.

Between prompts 3 and 4, carry the disarmed drone through the working volume.
Briefly rotate it left–right, return its nose to +X, and hold still at each position.
The cflib gesture detector triggers eight geometry samples, followed by five **new**
verification positions. Vary X, Y and height; repeated positions cannot pass the
coverage check. There are no additional keyboard questions for these samples.

Setup discovers the radio URI, exactly two live Lighthouse stations and their
channels, the D455 serial, tags and visual drone position. Station discovery uses
fresh `bsReceive` and `bsCalCon` bitmasks, as exposed by cfclient, rather than assuming
channels 1 and 2. Each station must appear in at least three samples within a rolling
0.75-second window, with the resulting two-station set stable for one second.
Brief reception gaps do not reset discovery on every individual status sample;
stale or stored-only stations cannot pass. Status history is saved in
`station-discovery.json`. More than one radio target requires an explicit `--uri`.
The D455 profile is selected for the USB connection (640×480 color/depth at 15 fps
on the observed USB 2 connection). Prefer a USB 3 port and cable. Tags 0, 1, 2 and
at least two other AprilTag **25h9** tags must be visible.
The D455 infrared projector is disabled during RGB-D capture: repeated stationary
comparisons showed Lighthouse collection timing out with it on (4–7 usable pairs),
but reaching all 50 with it off. Depth still runs using passive stereo, which needs
scene texture. The emitter's prior setting is restored when the camera closes.
Capture profiles record `ir_emitter_enabled: false`. Angle-stream progress and
packet counts are saved in `lighthouse/drone-*-lighthouse-stream.json`.
Reference drone placements do **not** have to be centred over these tags.

The output contains the tag layout image, intrinsics, paired camera/radio samples,
verification reports, resolved rig and accepted Lighthouse geometry. Calibration
writes geometry to the drone only after both calibration gates pass and launch
placement is checked. It never requests arming, takeoff or flight setpoints.
Use a new output directory when retrying.

### Skip the five verification positions

```sh
uv run --project flight --extra vision --extra realsense --extra detector flight-cal setup \
  --skip-verification --out runs/exp2-unverified
```

This still collects the three reference placements and eight geometry samples,
then returns directly to the launch placement. It saves the fitted Lighthouse and
camera transforms, tag positions and rig with `verification_skipped: true` and
`accepted: false`. Missing verification errors are recorded as unknown, not zero.
Convergence and camera fitting checks still apply. Geometry is not uploaded in
this mode; hover requires a verified setup or explicit `--allow-unverified`. The flag must
be supplied when starting setup; it cannot change a process already running.

To skip **both** the eight geometry samples and five verification positions on the
D455 in RGB-D mode:

```sh
uv run --project flight --extra vision --extra realsense --extra detector flight-cal setup \
  --skip-extra-samples --out runs/exp2-references-only
```

Only origin, +X at 1 metre, XY-plane, and return-to-launch prompts remain. Camera
registration uses depth from those three reference placements. No shake-and-hold
round runs. This produces unverified estimates without uploading geometry or
enabling hover. RGB-only mode still needs the extra geometry samples because three
image points alone do not uniquely resolve camera pose. Poor reference geometry
or a failed fit can still reject estimation.

To try a fit above the default 1-pixel RMS limit, set the limit explicitly:

```sh
uv run --project flight --extra vision --extra realsense --extra detector flight-cal setup \
  --skip-extra-samples --max-camera-rms-px 2.0 --out runs/exp2-relaxed-fit
```

The saved setup and camera reports record the selected limit and whether it was
overridden. The setup28 residual of 1.8773 px passes at 2.0 px. Skipped verification
still produces an unverified result, with no geometry upload or enabled hover.
The Lighthouse and held-out camera error limits remain separate from this RMS flag.

### Frozen camera images during fitting

In `exp2-setup9`, all three saved reference images were byte-identical and all drone
pixel centres matched, although Lighthouse positions changed. This cannot produce
a valid registration. Restart the camera connection, preferably with a USB 3 cable
and port, then capture a new setup. The old run cannot be repaired by increasing
the RMS limit. Camera acquisition now copies original color before depth alignment,
rejects repeated frame numbers/content, checks color/depth timing, and records sensor
frame numbers, timestamps and image hashes in each observation. Reference placement
checks reject an unchanged image position before attempting the fit.

## Drone detector

The local YOLOE-L model is prepared at `runs/models/drone-yoloe-l.onnx` and is
preferred automatically. Nano remains at `runs/models/drone-nano.onnx` and can be
selected with `--detector-model runs/models/drone-nano.onnx`.
It uses several visual prompts showing this actual drone at the origin and +X
positions, so a different background does not depend on one reference alone.
References and their bounding boxes are listed in `flight/configs/drone-reference.json`.
Every prompted class means the same drone; overlapping detections are suppressed
across references. Normal runtime
uses ONNX Runtime on CPU, without PyTorch or a GPU.

For a fresh checkout, prepare the model once using Python 3.11 or newer:

```sh
uv run --project flight --extra vision --extra detector --extra yolo flight-cal prepare-detector --size l
```

Preparation downloads Ultralytics weights and exports the prompted model. This
optional step installs PyTorch; its normal Linux distribution can be large.
Use `--detector-model FILE` to select an existing export during setup.
Weights and run archives are excluded from Git. The checkpoint contains
[download and rebuild instructions](../docs/checkpoint-10-15.md#download-and-rebuild-the-drone-detector),
including the official base-weight URL and expected output files.

On the saved, fully visible drone views, YOLOE-L at 640-pixel input scored roughly
0.95–0.98 with the same three visual reference images; nano at 960 pixels scored
roughly 0.70–0.94 on those views. CPU inference was about 0.6 seconds versus 0.13
seconds. These are example detection scores, not a measured accuracy rate or a
flight validation. The comparison is saved under
`runs/drone-visibility-investigation/large-640-comparison.json`. The larger model's
latency also needs validation against the hover monitor's freshness requirements
before using it for flight monitoring.

The failed +X image in `exp2-setup22` shows the drone clipped at the right edge.
A more capable model cannot recover the missing drone geometry for calibration.
Keep all reference placements fully inside one fixed camera view. Detected boxes
that touch the image edge are rejected explicitly.

YOLOE visual prompts support example-guided detection without training a drone
dataset ([official documentation](https://docs.ultralytics.com/models/yoloe/)).
SAM 3 also supports visual exemplars, but its official setup requires a CUDA GPU
and approved checkpoint access; it was not installed or evaluated here
([official repository](https://github.com/facebookresearch/sam3)).

A detection box estimates the drone body's centre. It does not precisely locate
the Lighthouse sensors, and detection confidence does not establish millimetre
accuracy. Occlusion, uncertain detections and multiple plausible drones stop the
workflow. Held-out measurements must still pass the limits below.

## RGB compatibility

Use an OS-accessible RGB camera and matching calibrated intrinsics, including on
macOS without the RealSense SDK:

```sh
uv run --project flight --extra vision --extra detector flight-cal setup \
  --camera-backend opencv --camera-mode rgb --device 0 \
  --intrinsics measured-intrinsics.json --out runs/exp2-rgb
```

This asks **one extra question** for the measured black-square side in millimetres
(five prompts total). Alternatively pass `--tag-size-mm 70`, replacing 70 with the
actual measured size. RGB assumes equal-size, coplanar tags. It uses drone bearings
for registration and lateral verification; it cannot independently measure drone
height. No depth stream or RealSense SDK is used with the OpenCV backend.

A D455 can also provide only color with `setup --camera-mode rgb`; camera discovery
then still uses the RealSense SDK. `flight-cal intrinsics --help` describes RGB
chessboard calibration. Camera mounting, resolution, focus and intrinsics must
remain fixed after setup.

Hover checks the saved intrinsics file against its recorded fingerprint, then
compares the live and saved lens mappings at 63 points across the image in both
directions. The serial/profile and distortion model must match; numerical mapping
drift must be at most 0.5 pixels. This tolerates small SDK focal-length variations
without requiring identical floating-point hashes. Geometry continues using the
saved calibration intrinsics. Live values and the comparison are saved in the
hover session's `intrinsics-check.json`. Larger changes still require calibration.

## Acceptance and hardware assumptions

- Lighthouse must converge with valid reference placements, varied verification
  positions and maximum held-out crossing-beam error **≤0.4 mm**. Lower error passes;
  this is the solver's consistency residual, not a claim of absolute world accuracy.
- Camera registration fits nonplanar drone/Lighthouse correspondences, reserving
  all five verification positions. Training RMS must be **≤1 pixel**; maximum
  held-out error must be **≤5 mm** (3D with depth, lateral bearing error for RGB).
- Tags are reconstructed from the camera floor plane and transformed into the
  Lighthouse world. They are marked as estimates, not independently surveyed
  ground truth. The camera gate uses held-out drone samples, not those tag estimates.
- Mass and envelope use the existing CF21B/fork CAD preset, including the steel pins.
  The sensor centre is assumed 20 mm above the body frame. These are recorded
  assumptions, not automatic physical measurements. A modified build needs an
  updated preset. Fork exports are in `brixzle/docs/exp2/gripper-fork-msh2-30/`.
- The inferred workspace surrounds the tags and launch position with an 18 cm XY
  margin. This assumes a clear area; it does not detect interior obstacles or prove
  that a landing surface is suitable. Review `rig.json` and `floor.png` before flight.

Strict camera gates may reject a box-centre estimate or noisy depth. Improve the
view and repeat; the workflow does not lower the limits to force acceptance.
At each placement, setup waits up to ten seconds for three consecutive drone
detections within three pixels, with confidence at least 0.40. One rejected frame
does not abort setup. Keep hands clear after pressing Enter. Failed visibility
captures save `drone-*-last.png` and `drone-*-visibility.json`, including the actual
confidence and frame metadata, so a detection failure can be inspected.

## Validate beside every tag

```sh
# Offline route preview: opens no devices.
uv run --project flight --extra vision --extra realsense --extra detector flight-cal tag-visit --setup runs/exp2-setup --out runs/exp2-route

# Camera, radio, geometry readback and estimator checks; motors stay disarmed.
uv run --project flight --extra vision --extra realsense --extra detector flight-cal tag-visit --setup runs/exp2-setup --check-only --out runs/exp2-preflight

# Explicit supervised flight authorization.
uv run --project flight --extra vision --extra realsense --extra detector flight-cal tag-visit --setup runs/exp2-setup --arm --out runs/exp2-tags
```

The route takes off, visits beside every tag, holds five seconds per tag, returns and lands.
It provides at least 20 cm below the gripper, with speed ≤0.1 m/s and acceleration
≤0.2 m/s². Camera images and events include drone detections; depth mode also logs
estimated world positions and offsets from each tag. Visual disagreement above
5 cm, moved camera alignment, stale telemetry or missing stations stops the flight
and requests landing while the radio link remains available. A lost radio link
prevents host commands; keep a human ready for hardware intervention.

### Fly from a completed setup with verification skipped

`hover-check --allow-unverified` accepts a **completed**, successfully fitted setup
whose reports explicitly record that verification was skipped. It does not accept
a failed fit or change the original calibration's `accepted: false` status.

```sh
# Offline route preview; opens no devices and uploads nothing.
uv run --project flight --extra vision --extra realsense --extra detector flight-cal hover-check \
  --setup runs/exp2-setup30 --allow-unverified --out runs/exp2-setup30-route

# Upload/read back the saved geometry and run preflight without arming.
uv run --project flight --extra vision --extra realsense --extra detector flight-cal hover-check \
  --setup runs/exp2-setup30 --allow-unverified --check-only --out runs/exp2-setup30-preflight

# Explicit supervised flight using that unverified calibration.
uv run --project flight --extra vision --extra realsense --extra detector flight-cal hover-check \
  --setup runs/exp2-setup30 --allow-unverified --arm --out runs/exp2-setup30-hover
```

The live commands upload geometry because reference-only setup did not upload it.
The uploader discovers supported station slots from geometry and calibration
memory reads, including unused slots. This drone supports four slots. cflib's
default writer pads to sixteen, which caused `Lighthouse geometry upload/persistence
failed` when it wrote nonexistent slots 5–16. The writer now receives the detected
capacity, preserves factory calibration, and verifies geometry after persistence.
The detected capacity and upload result are recorded under `lighthouse_upload`.
Camera checks, geometry/factory-calibration readback, estimator convergence, launch
position, battery, station reception, envelope and tracking checks still run. The
flight report records `unverified_calibration: true`, the measured camera RMS and
its selected limit. Source setup reports remain unverified. Keep the drone on the
original launch support and leave camera, tags and stations fixed.

YOLOE-L inference now uses up to six CPU threads (about 0.25 seconds per frame on
this host). The camera retains its one-second freshness limit. Visual positions
are compared with interpolated telemetry at image capture time, accounting for
inference delay. These changes have not been validated in an actual flight.

## Read-only discovery and advanced commands

```sh
uv run --project flight flight-cal discover
uv run --project flight --extra vision flight-cal --help
```

The earlier measurement wizard remains available only through `survey --advanced`.
Its `--layout` and `--measure-check-tags` options support a manually measured rig.
Separate `lighthouse`, `camera` and `estimate-tags` commands remain available for
advanced use. Normal setup does not run that questionnaire.

The earlier D455 survey found five tags (0–4), with a 100.8 cm estimated 0-to-1
baseline and approximately 70–71 mm black squares. Those initial estimates remain
in `flight/configs/tag-layout-overhead.json`; compact setup captures a fresh layout.

Implementation references: [Bitcraze cfclient Lighthouse tab](https://github.com/bitcraze/crazyflie-clients-python/blob/master/src/cfclient/ui/tabs/lighthouse_tab.py)
and [Ultralytics YOLOE visual prompting](https://docs.ultralytics.com/models/yoloe/).
