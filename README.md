# Aerial assembly

The parametric drop-assembled brick (CadQuery geometry, MuJoCo assembly trials,
NSGA-III search) is in [`brixzle/`](brixzle/README.md).

Experiment 2's Lighthouse calibration, overhead RGB/RGB-D camera registration,
and supervised tag hover checks are in the [flight calibration guide](flight/CALIBRATION.md).
The [October 15 checkpoint](docs/checkpoint-10-15.md) documents the successful
tag-visit flight and the local RealSense replay. Large captures and model weights
stay local; see [model setup](docs/checkpoint-10-15.md#download-and-rebuild-the-drone-detector)
to download the base weights and rebuild the prompted detector after cloning.

The simulation project, CAD files, saved drops, and Python environment are in
[`simulation/`](simulation/README.md).

From this repository root:

```bash
cd simulation
source .venv/bin/activate
aerial cad-drop goat_mk2 --config examples/cad-experiment.json --out runs/new-drop --video
```

Choose a new output directory for each drop. If your terminal was using the
environment before the move, reactivate it with the command above.

See the [simulation guide](simulation/README.md) for setup and commands, and
[validation evidence](simulation/VALIDATION.md) for the timestep benchmarks.
