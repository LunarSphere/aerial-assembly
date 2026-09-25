# Aerial assembly

The simulation project, CAD files, saved drops, and Python environment are in
[`simulation/`](simulation/README.md).

From this repository root:

```bash
cd simulation
uv sync --all-extras
uv run --all-extras aerial cad-drop two_peg_block --config experiment_configs/cad-experiment-fast.json --out runs/new-drop --video
```

Choose a new output directory for each drop. `uv sync` creates and maintains
the project environment from `simulation/uv.lock`; no manual activation is
needed.

See the [simulation guide](simulation/README.md) for setup, commands, and the
current grid physics settings.
