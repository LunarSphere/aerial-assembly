# Discrete drop searches

`cad-grid` enumerates the Cartesian product of six axes. Each range is
`[minimum, maximum, step]`, includes both endpoints, and requires a positive
step that divides the range exactly. A fixed axis uses equal endpoints. The
axes are X/Y offsets in millimeters, vertical clearance in millimeters, and
three Euler angles in degrees. Every drop starts at rest.

The current 27-state config is
[`cad-experiment-grid-fast-reference-27.json`](../experiment_configs/cad-experiment-grid-fast-reference-27.json).
It varies X and Y from −5 to +5 mm and side-profile Y rotation from −5 to +5°;
the other axes are fixed. It uses 0.3-second trials, a 0.1-second final dwell,
timestep `0.002`, and contact time constant `0.005`.

The 27-state config uses insertion scoring: at the end of the trial, every leg
tip must be within its socket opening and no deeper than the socket floor.
Success is based only on this tip position; it does not require a stable pose,
dwell, support force, or saved reference drop. The flush-scored single-state config is
[`cad-experiment-grid-fast.json`](../experiment_configs/cad-experiment-grid-fast.json).

The 27-state config retains `reference` in its historical filename, but its
success mode is now `insertion`.

Count states without preparing geometry or simulating:

```bash
aerial cad-grid two_peg_block \
  --config experiment_configs/cad-experiment-grid-fast-reference-27.json \
  --out runs/grid --dry-run
```

Run selected states and retain their trajectories for replay or rendering:

```bash
aerial cad-grid two_peg_block \
  --config experiment_configs/cad-experiment-grid-fast-reference-27.json \
  --out runs/grid --record-trial 0 --record-trial 26
aerial replay runs/grid --trial 0
aerial render runs/grid --trial 26 --out runs/grid/last.mp4
```

Completed states are checkpointed one at a time. Continue an interrupted run
with `--resume`, and optionally limit each invocation with `--max-states N`.
Ranges, physics, scoring, geometry, code, or recorded dependency versions must
match when resuming. Grid outcomes describe the configured simulation only;
they are not hardware success probabilities.
