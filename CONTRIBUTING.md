# Contributing

This is a small research repo maintained by three or four people. These rules keep us from breaking each other's work and keep claims honest. There is no production tooling: no linter, no formatter, just tests.

## Setup

You need [uv](https://docs.astral.sh/uv/) and `gh`. Each project has its own environment.

```bash
cd brixzle && uv sync --all-extras && uv run pytest        # Python 3.12, CadQuery + MuJoCo, about 80 s
cd .. && uv run --project flight --extra test pytest flight/tests -q   # about 2 s, mocks cflib
```

Run `flight` commands from the repository root so `runs/` paths resolve. The hardware extras (`vision`, `realsense`, `detector`) are only needed for calibration and flights; see `flight/README.md`.

## Workflow

1. **Start from an issue.** Every change has a GitHub issue. If there isn't one, open it first.
2. **Branch from `master`**, named `<issue#>-short-name` (for example `17-gradient-optimizer`). One issue, one branch, one PR.
3. **Test first.** Write a failing test, make it pass, then clean up. New behaviour needs a new test. Bug fixes need a test that reproduced the bug.
4. **Open a PR** with `Closes #N` in the body. Say what changed, which tests you ran, and the evidence behind any numbers (see below).
5. **Review.** The maintainer reviews and merges, or asks Claude to merge. CI (`flight` and `brixzle` pytest) must be green.

Keep commits small, with an imperative subject ("Add lip gene to NSGA bounds"). When you add, move or repurpose a module, update `.claude/REPO_MAP.md` in the same PR. When an Experiment 2 decision changes, update `brixzle/docs/exp2/STATUS.md`.

## What not to commit

- `runs/`, `assets/`, `.cache/`: experiment outputs and generated geometry stay local.
- `results_*/` directories, `MUJOCO_LOG.TXT`, `.DS_Store`.
- Model weights (`*.onnx`, `*.pt`, …), videos and bag files, except small curated media under `brixzle/docs/`.
- `.env` and any credentials.

These are all in `.gitignore`. If `git status` shows one of them, don't force-add it.

## What counts as a validated result

A number (success rate, error in mm, timing) can go in a PR, README or `STATUS.md` only if it comes with:

- the command, config file and seed that produced it,
- the run directory it came from (local `runs/<dir>` is fine, as long as it's named), and
- a label saying whether it is **simulated** (uncalibrated contact and aerodynamics, not a hardware rate) or **measured on hardware** (date, setup, number of trials).

Simulated success rates are hypotheses about hardware, not results. Datasheet values need a link or a page reference.

## Hardware

Only a human arms the Crazyflie. `fly` and `flight-cal` default to dry runs; `--check-only` opens the camera and radio without arming; `--arm` flies. Use a new `--out` directory for every run. Claude Code is told never to arm or open a radio link (see `CLAUDE.md`).

## Working with Claude Code

`CLAUDE.md` holds the rules Claude follows. Two project skills live in `.claude/skills/`:

- `/pick-up-issue N`: reads the issue, branches, writes a failing test, implements it and opens the PR.
- `/grill-me`: questions you about an open-ended design issue (for example #17 or #18) before any code is written.
