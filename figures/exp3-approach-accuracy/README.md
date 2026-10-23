# Exp 3: Crazyflie X-Y approach accuracy (issue #24)

Raw data and first figures, for discussion. Nothing has been excluded or re-weighted yet.

## What was done

The Crazyflie takes off from its launch pad, hovers 5 s with its Lighthouse deck over the centre
of each of the 5 floor AprilTags, then lands. Each trial is one such route, so 5 approaches. We
flew **5 trials (25 approaches)** on 2026-10-22, in `--demo` mode (PID controller), session
`exp3-s2`.

Two errors are measured over each 5 s hover, in X-Y only:

- **Positioning error:** where the overhead D455 camera sees the deck, minus the tag centre. The
  camera pose is refit to the floor tags every frame, so the cage moving does not bias it. This is
  the error a pickup actually experiences.
- **Control error:** where Lighthouse says the drone is, minus where it was commanded to go. This
  shows how well the controller tracks its own position estimate.

## Results so far

| Data | Positioning: mean / RMSE / max | Control: mean / RMSE / max |
|---|---|---|
| All 5 trials (25 approaches) | 1.54 / 1.75 / 4.2 cm | 0.96 / 1.55 / 5.0 cm |
| Calibration 1: trials 1, 3, 4 | 1.74 / 2.01 / 4.2 cm | 1.40 / 1.98 / 5.0 cm |
| Calibration 2: trials 7, 8 | 1.24 / 1.27 / 1.7 cm | 0.30 / 0.35 / 0.6 cm |

Per-tag numbers are in [`summary.csv`](summary.csv).

![Error vectors per tag](errors-by-tag.png)

![Error per approach in flight order](errors-by-trial.png)

## Points for discussion

1. **The Lighthouse calibration drifted during calibration 1.** Before each takeoff we compare
   where Lighthouse and the camera put the resting drone. That offset grew from 3.5 to 10.9 to
   17.0 mm over trials 1, 3 and 4. The next two attempts could not hold a stable position, and a
   live check then showed the two base stations disagreeing by about 4.5 cm. After recalibrating
   (calibration 2) the offset was 1–2 mm, and control error fell from 1.40 to 0.30 cm. The most
   likely cause is the base stations moving with the cage when someone steps in to reset the
   drone. Since then, every flight checks the stations against their calibration first
   (residuals 0.41 and 0.44 mrad against a 1.5 mrad limit).
2. **What to report.** All 5 trials together (realistic for this rig), calibration 2 only (2
   trials, a clean setup), or more trials on a stable setup.
3. **Tag 3 is the worst** (positioning 2.3 cm, control 2.6 cm mean). It is the far corner from the
   calibration placements, and it was most affected by the drifting calibration.
4. **Positioning error stays at about 1.2 cm even with a good calibration**, while control error
   drops to 0.3 cm. The remaining gap is between where Lighthouse thinks the drone is and where it
   really is, plus camera measurement error (see limitations).
5. **Trial count.** 5 trials, not the planned 20. A charge gives 2–3 flights before the 3.8 V
   preflight minimum, and the cage needs a recalibration whenever it is disturbed. Jitter at rest
   was under 1 mm on the fresh calibration, so the earlier growth in jitter looks like station
   drift rather than battery (see [`../lighthouse-rest-jitter.md`](../lighthouse-rest-jitter.md)).

## Box centre vs SAM mask centroid (#26)

Does measuring the drone with a segmentation mask, rather than the YOLOE box centre, close the
positioning gap? Offline re-run on the same flights: SAM 2.1 (base), prompted with each logged
YOLOE box, gives a whole-drone mask, and its pixel centroid is cast onto the same median-depth
plane as the box centre. Only the frames saved during the flights (1 per second) have images, so
both methods are compared on those frames: 2–5 per approach instead of 9–14.

| Positioning error, same saved frames | Box centre: mean / RMSE / max | Mask centroid: mean / RMSE / max |
|---|---|---|
| All 5 trials (25 approaches) | 1.54 / 1.83 / 4.4 cm | 1.66 / 2.06 / 4.9 cm |
| Calibration 1: trials 1, 3, 4 | 1.80 / 2.14 / 4.4 cm | 2.06 / 2.44 / 4.9 cm |
| Calibration 2: trials 7, 8 | 1.17 / 1.24 / 1.8 cm | 1.07 / 1.29 / 2.1 cm |

The box centre on the saved frames only is within 0.1 cm (mean) of the full-rate result above.
Per approach: [`methods.csv`](methods.csv); summary: [`methods-summary.csv`](methods-summary.csv).

![Box centre vs mask centroid per approach](positioning-methods.png)

**Reading:**

1. **The mask centroid does not close the gap.** It is better on 10 of 25 approaches, and the
   means differ by about 0.1 cm in either direction. Calibration 2 still sits at about 1.1 cm.
2. **The two centres differ by 0.9 cm on average, and the difference depends on the tag.** Tag 1
   shifts +1.3 cm in Y every time (spread 0.07 cm); tag 4 shifts −0.7 cm in Y. So the choice of
   "drone centre" moves the result by up to about 1 cm in a position-dependent way. Possible
   causes, not yet tested: perspective (the camera sees the drone's side away from the image
   centre), shadows inside the box, and the mask failures below.
3. **The masks are imperfect** ([`mask-examples.png`](mask-examples.png), every 10th frame).
   SAM sometimes drops a prop guard next to a tag or the cage frame, which moves the centroid by
   up to about 2 cm (centroid–box distance: median 2.9 px, max 11.9 px; 1 px ≈ 1.9 mm here). It
   sometimes includes the white tag under the drone, which pulls the centroid toward the tag
   centre and so flatters the mask result. No frames are excluded.
4. **The frames have the YOLOE box drawn on them** (#29). Untouched, SAM segments the drawn
   rectangle, so its exact pixels are inpainted first; the prop guards it touched are partly
   repainted. Clean frames from #29 would remove this step.

Rebuild: `segment_frames.py` (needs the frames in `runs/exp3-s2` and the `yolo` extra) writes
`data/trial-NN/masks.json`, then `make_figures.py` picks them up.

## Known limitations

- Drone tilt is ignored; the camera measures the visual centre of the drone, not the deck sensor
  centre. A SAM mask centroid does not change the positioning result (see above).
- Each per-tag value rests on 5 approaches (2 for calibration 2 alone).
- Z is not evaluated.

## Files

- `approaches.csv`: one row per approach, with both errors (cm), positions (m), calibration and
  frame counts.
- `trials.csv`: every attempt (including the 3 that did not fly), with battery voltage, station
  check residual and pad offset.
- `summary.csv`: n, mean, RMSE, std, max and bias for each error, overall and per tag, pooled and
  per calibration.
- `data/`: raw session files and full event logs (gzipped) for every attempt. `.gitattributes` hides
  them from diffs. Routes are not kept; the originals stay in `runs/`.
- `methods.csv`, `methods-summary.csv`, `positioning-methods.png`, `mask-examples.png`: the #26
  box-vs-mask comparison.
- `segment_frames.py`: SAM mask centroids from the saved frames in `runs/` into `data/trial-NN/masks.json`:
  `uv run --project flight --extra vision --extra yolo python figures/exp3-approach-accuracy/segment_frames.py runs/exp3-s2`.
- `make_figures.py`: rebuilds everything above from `data/`:
  `uv run --project flight --extra plot python figures/exp3-approach-accuracy/make_figures.py`.
