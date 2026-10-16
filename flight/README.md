# Experiment 2: calibration and AprilTag hover validation

After cloning, [download and prepare the detector](../docs/checkpoint-10-15.md#download-and-rebuild-the-drone-detector),
then run setup for your physical arrangement. Model weights and run archives stay
local; the saved setup paths below refer to the original workstation.

For the current completed `runs/exp2-setup37`, run a stationary sanity check:

```sh
uv run --project flight --extra vision --extra realsense --extra detector flight-cal hover-check --setup runs/exp2-setup37 --demo --arm --out runs/exp2-hover-demo
```

`hover-check` takes off, holds over the launch point for ten seconds, and lands.
Use `--hover-seconds` to change the hold. Tag visits are a separate command:

```sh
uv run --project flight --extra vision --extra realsense --extra detector flight-cal tag-visit --setup runs/exp2-setup37 --demo --arm --out runs/exp2-tag-visit
```

`tag-visit` hovers five seconds beside each tag, offset toward the flight area's
centre. The minimum offset is 15 cm (`--tag-offset-m 0.15`); the planner increases
it up to 40 cm when the camera projection predicts tag occlusion. Inspect
`route-preview.png` and `route.json` from a run without `--arm` first.

`--demo` refreshes the camera pose from fixed tags, allows skipped verification,
and uses PID with 10 cm target tolerance. Lighthouse controls flight;
YOLO observations are diagnostics. Use `--camera-check-only` instead of `--arm`
to check the camera while the drone is off, or `--check-only` for full preflight.
Keep the tag map fixed relative to the base stations and return the drone to its
original launch position and heading. Each run needs a new output directory.

Run the cfclient-style setup from the repository root:

```sh
uv run --project flight --extra vision --extra realsense --extra detector flight-cal setup --out runs/exp2-setup
```

There are **four placement prompts**: origin, 1 metre along +X, another position on
that plane, and return to launch. Extra geometry and verification samples use
cfclient's shake-and-hold gesture, without more keyboard questions.

The command discovers the Crazyradio link, two Lighthouse stations, overhead D455,
AprilTags and drone. A CPU visual-prompt detector supplies drone observations;
setup prefers the prepared YOLOE-L model and falls back to nano when unavailable.
RGB-only mode needs one additional tag-size answer. Setup never arms the drone.

Then inspect the route, check the hardware, and explicitly start a supervised flight:

```sh
uv run --project flight --extra vision --extra realsense --extra detector flight-cal hover-check --setup runs/exp2-setup --out runs/exp2-route
uv run --project flight --extra vision --extra realsense --extra detector flight-cal hover-check --setup runs/exp2-setup --check-only --out runs/exp2-preflight
uv run --project flight --extra vision --extra realsense --extra detector flight-cal hover-check --setup runs/exp2-setup --arm --out runs/exp2-hover
```

For the separate tag route, replace `hover-check` with `tag-visit`.
For a completed setup that skipped verification, add `--allow-unverified` to
either command. Live modes upload and read back its geometry; `--arm` is still
required to fly. The output records that the calibration is unverified.
Read the [calibration guide](CALIBRATION.md) for model preparation, RGB support and
acceptance limits. The earlier [pick-and-place mission](MISSION.md) is separate.
