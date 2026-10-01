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


## Feedback 
- angled pegs should be able to build an infinite overhang
- try to find a block design that can generalize to any structure. | generablizable shape could be voxelized
- currently account for left and right error | trick: make sure there is always error in one direction then you can simplify design to rely on the error in one direction
- if we rely on one side of error| one side slides then we can rely on motion that is not verticle | big issue with sliding is friction make it more steep than the friction coefficient suggets
- bouncing behavior from the lego design
- we should shot for being able to stretch out to N number of blocks in sim. | rather than say what design does this ask what is the space of designs that satisfy our conditions. 
- ex: rule if we have an angle and we expect slidng to happen then we we need to ensure we have an angle that will allow the block to slide down. you can use an arctangent for this. 
- ex2: lets say we dont want to tower to fall down could we angle the peg such that it| imagine if you can angle the peg such that it cant tilt out| you saw a reel that describes something similar the other day. check your camera roll
- alternating angles to ensure the arch cant collapse from either direction. 
- ex3: add test cases to make you think about what could cause structure to collapse
- ex4: skinny pegs will break how do we ensure they are struturally sound: pegs decide assembly so imagine a cone begs lead the way but the blocks body will resist forces. 
- ex4: use I-beams to resist structural flexing
- ex6: big picuture come up with rules that justfy the design we have chosen. | this principal is also true for the gripper. 
- we should be able to paint a structure with the block we choose
- Can we assume that the block is extremely heavy/attatched to the ground. yes. its also very interesting if the block is not fixed to the ground.  
- the keystone bridge is called a roman arch
- expansion different paper: how do we disassemble these structures| could we take our assembly apart and build something else. tower to | arch
- Jamming: jamming gripper: bag of sand that complies nicely to structure you are gripping. the sand all jams up when you do something to it. | this would be a cool way to make a gripper that conforms to the object you want to grasp. 
- another perspecive could i make structure permanant by sprinkling grains of sand on top of it. 
- assemble two things and make it pernant or semi permanant by jamming in certain ways | another paper: anyforce you apply causes closing force around whatever you apply. 
- start with one of these ideas | don't be afraid to ask for help. 
- try having gpt 6 draw the cad model for you in blender | no answers but good ideas. | turn a principal into a design
- check related work for the ideas to ensure new papers havent explored this. 
- use computational math to solve for best design by doing it in sim | in-progress | this should come after defining the design. 

