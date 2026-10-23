# CLAUDE.md

Guidance for Claude Code in this research repo. Humans: see `CONTRIBUTING.md`.

## Layout

Two independent Python projects, each with its own uv environment.

| Dir | Package / CLI | Toolchain |
|---|---|---|
| `brixzle/` | `brixzle` / `brixzle` | uv, Python 3.12 (CadQuery), MuJoCo drop trials, NSGA-III brick search, simulated drone; depends on `../flight` (editable) |
| `flight/` | `flight` / `fly`, `flight-cal` | uv, cflib 0.1.33; real Crazyflie or simulator |

`stl_exports/` holds CAD meshes. `runs/`, `assets/`, `.cache/`, weights and videos are git-ignored. `.claude/REPO_MAP.md` maps the modules: update it in the same change when you add, remove, rename or repurpose one.

## Commands

```bash
cd brixzle && uv sync --all-extras && uv run pytest      # about 80 s
uv run brixzle export --params configs/ab.json --out runs/v0
uv run brixzle fly-assemble tower --bricks 1 --out runs/plan-single

# flight/: run from the repo root so runs/ resolves
uv run --project flight --extra test pytest flight/tests -q   # about 2 s, mocks cflib
uv run --project brixzle fly runs/plan-single/plan.json --sim
```

CI (`.github/workflows/tests.yml`) runs both pytest jobs on every PR and on `master`.

## Workflow

1. Find the GitHub issue (`gh issue view N`). No issue, no branch: ask for one or offer to draft it.
2. Branch from an up-to-date `master` as `<issue#>-short-name`. One issue, one branch, one PR.
3. TDD: write a failing test, run it and see it fail, then make it pass. New behaviour needs a new test.
4. Run the affected project's tests; they must pass before the PR.
5. Open a PR whose body says `Closes #N`, lists the tests run, and cites evidence for any result. The maintainer reviews every PR; merge only when they ask, once CI is green.

`/pick-up-issue N` runs steps 1–5. `/grill-me` stress-tests an open-ended design issue before any code.

## Hardware safety

- Never arm the drone, open a radio link, or run anything with `--arm` (`fly`, `flight-cal`). Only a human flies.
- Allowed: tests, `--sim`, dry runs (the default) and offline checks. Ask before `--check-only`, since it opens the camera and radio.
- If a task needs a real flight, stop and hand the human the exact command.

## Reporting results (anti-hallucination)

- Simulated results are not hardware rates. Label them "simulated, uncalibrated contact".
- Never state a number, success rate or timing without a source to cite: a run directory, seed and config, a test, or a `STATUS.md` line.
- Say when a result was not produced in this session ("from `STATUS.md`, not re-run").
- If you did not run it, do not say it passed. Report failing tests with their output.
- Don't invent datasheet values or cflib/crazyflow APIs. Read `third_party/`, the installed package source, or the docs, and say when a value is a guess.

## Conventions

- SI units. Quaternions are `[w, x, y, z]` everywhere; convert to scalar-last only via `flight/flight/frames.py` (`wxyz_to_xyzw`).
- Every experiment writes to a new output directory (`--out runs/<new-dir>`); never overwrite an earlier run.
- When an Experiment 2 decision changes (thrust, sensing, gripper, controller, fixture), update the decision table in `brixzle/docs/exp2/STATUS.md` in the same PR.
- Four-space indent, `snake_case`, `PascalCase` classes, dataclasses for settings, explicit input validation, finite JSON. No formatter or linter; match the surrounding code.
- Physics changes: check determinism and invalid states, and keep synthetic smoke tests separate from validated CAD results.
- What to commit: hand-written configs, test fixtures, and only the raw data a figure script actually reads (under `figures/<name>/data/`, which `.gitattributes` hides from diffs). Don't commit regenerable outputs such as `route.json`, `route-preview.png` or `metrics.json`; they stay in git-ignored `runs/`.
- Commits: short imperative subject, one coherent change.

## Architecture

- **Mission vs. hardware (`flight/flight/`)**: `mission.py` talks only to the cflib API, an injected clock and a pose source, so one `plan.json` flies in MuJoCo (`--sim` lazily imports `brixzle`) or on a real Crazyflie. `fake.py` records cflib calls for dry runs; `checks.py` (offline) and `preflight.py` (live) gate real flights.
- **Calibration (`flight/flight/calibration/`)**: Lighthouse geometry → overhead D455 registered to the Lighthouse frame via fixed AprilTags → `hover.py` (hover-check) and `visits.py` (tag-visit). `flight-cal` defaults to a dry-run route preview; `--check-only` runs preflight without arming; only `--arm` flies.
- **brixzle**: `params.py` → `profile.py`/`cad.py` geometry → `scene.py`/`trials.py` MuJoCo trials → `optimize.py` NSGA-III. `drone.py`, `cf/` and `simcf.py` simulate the Crazyflie for `fly-assemble`.

Dependabot bumps uv and Actions weekly; minor and patch bumps auto-merge once checks pass.
