# Repository Guidelines

## Project Structure & Module Organization

`src/aerial_assembly/` contains the Python package: `cli.py` exposes the `aerial` command; geometry, MJCF, and CAD modules prepare models; `simulation.py` and `experiments.py` execute and score trials; `onshape.py` handles API integration. `tests/` mirrors these areas, with shared fixtures in `conftest.py`. `examples/` holds JSON configurations, `export_example/` contains sample CAD exports, and `docs/` explains import workflows. Generated geometry belongs in `assets/` and experiment outputs in `runs/`. Consult `VALIDATION.md` for numerical evidence and limitations.

## Build, Test, and Development Commands

Use Python 3.11+ from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
python -m pytest -q
```

These commands create an environment, install locked dependencies and the editable package, and run tests. If locked wheels are unavailable, use `python -m pip install -e '.[test]'` and record resolved versions.

For a local smoke test:

```bash
aerial demo --out assets/demo.json
aerial validate assets/demo.json --out runs/geometry-validation.json
aerial drop assets/demo.json --config examples/quick-check.json --out runs/quick
```

Experiment output directories must be new. Simulation runs headlessly; `aerial replay runs/quick --collisions` requires a graphical display.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions/modules, and `PascalCase` for classes. Follow surrounding code and use dataclasses for structured settings. No formatter or linter is configured. Preserve SI units, degree-based CLI release angles, and `[w, x, y, z]` quaternion ordering. Validate inputs explicitly and keep JSON finite.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions. Reuse shared fixtures and temporary directories; mock Onshape responses. Run focused checks with `python -m pytest tests/test_simulation.py -q`. No coverage threshold is configured. Physics changes should check determinism, invalid states, and timestep refinement; distinguish synthetic smoke tests from validated CAD performance.

## Commit & Pull Request Guidelines

History contains only `research project`, so no established commit convention exists. Use concise, imperative subjects describing one coherent change. PRs should explain the behavior change, relevant issues, tests run, and numerical evidence for geometry or physics changes. Include screenshots or videos when visual behavior changes, and document seeds, configurations, and validation limits for experiment claims.
