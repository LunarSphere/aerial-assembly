# Aerial assembly

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
