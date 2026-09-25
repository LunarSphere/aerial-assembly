# Local block drops in MuJoCo

Prepare a local `robot.xml` export, build collision geometry, and simulate a
rigid block dropping onto an identical fixed copy. The current model is in
[`two_peg_block/`](two_peg_block/); it contains the MJCF input and the STL mesh
required by that input. No CAD service connection is needed.

## Install

With `uv` installed, run this from the `simulation/` directory:

```bash
uv sync --all-extras
```

This creates or updates the project environment using `uv.lock`, including the
test and video extras. Run project commands through `uv run`; manual environment
activation is unnecessary. On Windows, use Linux/WSL for `cad-grid`, which uses
Unix file locking.

## Run a drop

```bash
uv run --all-extras aerial cad-drop two_peg_block --config experiment_configs/cad-experiment-fast.json --out runs/drop
uv run --all-extras aerial replay runs/drop
uv run --all-extras aerial render runs/drop --out runs/drop/replay.mp4
```

Each run needs a new output directory. The example runs a 0.3-second trial
with a 0.1-second final dwell. It sets timestep `0.002` s and contact time
constant `0.005` s. The package defaults are `0.1` s and `0.2` s. Contact time
constants must be at least twice the timestep so MuJoCo's `refsafe` clamp does
not silently change the requested value.

## Incremental floor-supported chain

The complement export is a socket-only first block that rests on the floor.
Add two-peg stepping blocks one at a time, keeping the assembly's simulated
state between additions. Each ideal placement puts one peg tip inside the
selected socket and requires the other peg to remain outside the parent's
sockets. The runner settles each addition for one second. Peg disengagement
alone is not collapse; the run stops when any stepping block contacts the
floor. It compiles the requested block capacity once and keeps unused copies
parked with gravity compensation until activation.

```bash
uv run --all-extras aerial cad-chain ../two_peg_block_complement two_peg_block \
  --config experiment_configs/cad-chain-floor.json \
  --out runs/chain --max-blocks 20 \
  --base-mass-grams 480 --block-mass-grams 26
```

To model the complement as bolted to the floor, use
`experiment_configs/cad-chain-bolted.json`. This locks all six degrees of
freedom of the complement; the stepping blocks remain dynamic.

```bash
uv run --all-extras aerial cad-chain ../two_peg_block_complement two_peg_block \
  --config experiment_configs/cad-chain-bolted.json \
  --out runs/chain-bolted --max-blocks 20 \
  --base-mass-grams 480 --block-mass-grams 26
```

to render a chain collapse
```bash
uv run --all-extras aerial render-chain runs/chain-bolted/ \
    --out runs/chain-bolted/collapse.mp4
```

Each stage writes its body states and recorded trajectory to `stage_NNN.json`
and `stage_NNN.npz`; `summary.json` reports the stable count and first failing
addition. Placement is idealized. Results depend on the supplied geometry and
configured contact values and are not hardware success rates.

The grid configs are:

- `experiment_configs/cad-experiment-grid-fast.json` — one flush-scored state.
- `experiment_configs/cad-experiment-grid-fast-reference.json` — one
  reference-scored state.
- `experiment_configs/cad-experiment-grid-fast-reference-27.json` — a
  27-state insertion-scored grid. The historical filename is retained for
  compatibility; it no longer uses a reference drop.

The 27-state grid counts a drop as successful when both leg tips are inside
their respective socket openings at the end of the trial. This geometric check
does not require a dwell, stable pose, or reference drop.

```bash
uv run --all-extras aerial cad-grid two_peg_block \
  --config experiment_configs/cad-experiment-grid-fast-reference-27.json \
  --out runs/grid --dry-run

uv run --all-extras aerial cad-grid two_peg_block \
  --config experiment_configs/cad-experiment-grid-fast-reference-27.json \
  --out runs/grid --record-trial 0 --record-trial 26
```

Completed grid states are checkpointed. Resume an interrupted run with
`--resume`; use `--max-states N` to limit each invocation.

## Local export and model limits

The importer reads MJCF and referenced mesh files from a local directory. It
supports one watertight solid with two vertical peg/socket axes, or three
unmerged solids consisting of two pegs and one body. Unsupported geometry
fails explicitly. Preparation preserves visual geometry, derives mass and
inertia from the source solid, and partitions collision geometry locally.
Default density is provisional at 600 kg/m³; use `--mass-grams` to supply a
measured total mass.

The supplied block's pegs bottom out before flush seating. A physical drop can
therefore settle with a seating error even when the simulation is numerically
valid. Physics values are not calibrated material properties, and a grid
success fraction is not a hardware success rate.

Run directories contain the scene, prepared geometry, validation report,
experiment settings, summary, and trajectory. A failed render does not discard
the simulation results.

## Development

Code is in `src/aerial_assembly/`; tests are in `tests/`. Synthetic test
geometry is only a regression fixture and does not represent the supplied
block. Tests marked `requires_local_model` use the checked-in local model;
three-solid integration cases are skipped unless a separate
`three_part_export/assets/` directory is available.

Run the test suite with `uv run --all-extras pytest`.
